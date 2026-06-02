"""DM-FoundAD trainer + few-shot evaluator (argparse, no Hydra).

Designed for hand-off via bash scripts. Trains DMFoundAD on a few-shot folder
and evaluates on MVTec/VisA test set with TopK image score and pixel L2 score.

Usage (see scripts/main_dm_foundad.sh):
    python -m foundad.src.dm_train \
        --mode train --dataset mvtec \
        --train_root /path/to/mvtec_4shot \
        --seg_layers 1,4,7,10,13 --cls_layers 1 \
        --epochs 2000 --diy_name dm_full

    python -m foundad.src.dm_train \
        --mode eval --dataset mvtec \
        --test_root /path/to/mvtec --diy_name dm_full --ckpt_step 12000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.datasets.dataset import build_dataloader
from src.dm_foundad import DMFoundAD
from src.utils.cls_perturb import CLSPerturb
from src.utils.synthesis import CutPasteUnion

MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# ============================================================================ #
# Train
# ============================================================================ #

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seg_layers = parse_int_list(args.seg_layers)
    cls_layers = parse_int_list(args.cls_layers) if args.cls_layers else []

    model = DMFoundAD(
        model_name=args.model,
        seg_layers=seg_layers,
        cls_layers=cls_layers,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
        cls_hidden=args.cls_hidden,
        cls_depth=args.cls_depth,
        share_seg_backbone=args.share_seg_backbone,
        if_pe=args.if_pe,
        feat_normed=args.feat_normed,
    )
    print(f"Trainable params: {sum(p.numel() for p in model.trainable_parameters()) / 1e6:.3f} M")

    _, loader, sampler = build_dataloader(
        mode="train",
        root=args.train_root,
        batch_size=args.batch_size,
        pin_mem=True,
        resize=args.crop,
        use_hflip=True, use_vflip=True, use_rotate90=True,
        use_color_jitter=True, use_gray=True, use_blur=True,
    )

    cutpaste = CutPasteUnion(colorJitter=0.5)
    cls_perturb = CLSPerturb(mixup_alpha=args.mixup_alpha)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    log_dir = Path("logs") / Path(args.train_root).name / f"dm{args.diy_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    csv_path = log_dir / "train.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss_total", "wall_ms"])

    gstep = 0
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        ep_loss, ep_n = 0.0, 0
        for itr, batch in enumerate(loader):
            imgs, labels, paths = batch
            imgs = imgs.to(device, non_blocking=True)

            _, imgs_patch_aug = cutpaste(imgs, labels)
            _, imgs_cls_aug = cls_perturb(imgs, labels)

            t0 = time.time()
            loss, loss_dict = model.training_step(
                imgs_clean=imgs,
                imgs_patch_aug=imgs_patch_aug,
                imgs_cls_aug=imgs_cls_aug,
                paths=paths,
                gate_p=args.gate_p,
                gate_c=args.gate_c,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            wall_ms = (time.time() - t0) * 1000
            gstep += 1
            ep_loss += float(loss.item()); ep_n += 1

            if gstep % 100 == 0:
                comp = " ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != "total")
                print(f"[E{ep+1:04d} I{itr:04d} g{gstep:06d}] loss={loss.item():.4f} ({comp}) {wall_ms:.0f}ms")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep + 1, gstep, f"{loss.item():.6f}", f"{wall_ms:.1f}"])

            if args.save_every_steps > 0 and gstep % args.save_every_steps == 0:
                _save_ckpt(model, log_dir, gstep)

        print(f"Epoch {ep+1} avg-loss={ep_loss / max(ep_n, 1):.6f}")

    _save_ckpt(model, log_dir, gstep)
    print(f"Done. Last checkpoint at step {gstep}.")


def _save_ckpt(model: DMFoundAD, log_dir: Path, gstep: int) -> None:
    state = {
        "seg_projectors": model.seg_projectors.state_dict(),
        "cls_projectors": model.cls_projectors.state_dict(),
        "step": gstep,
    }
    torch.save(state, log_dir / f"dm-step{gstep}.pth.tar")


# ============================================================================ #
# Eval
# ============================================================================ #

@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seg_layers = parse_int_list(args.seg_layers)
    cls_layers = parse_int_list(args.cls_layers) if args.cls_layers else []

    model = DMFoundAD(
        model_name=args.model,
        seg_layers=seg_layers,
        cls_layers=cls_layers,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
        cls_hidden=args.cls_hidden,
        cls_depth=args.cls_depth,
        share_seg_backbone=args.share_seg_backbone,
        if_pe=args.if_pe,
        feat_normed=args.feat_normed,
    )
    model.eval()

    log_dir = Path("logs") / Path(args.train_root).name / f"dm{args.diy_name}"
    ckpt = log_dir / f"dm-step{args.ckpt_step}.pth.tar"
    state = torch.load(ckpt, map_location="cpu")
    model.seg_projectors.load_state_dict(state["seg_projectors"])
    if cls_layers:
        model.cls_projectors.load_state_dict(state["cls_projectors"])
    model.to(device)
    print(f"Loaded {ckpt}")

    if args.classnames_override:
        classnames = [c.strip() for c in args.classnames_override.split(",") if c.strip()]
    else:
        classnames = MVTEC_CLASSES if args.dataset == "mvtec" else VISA_CLASSES
    K = args.K_top_mvtec if args.dataset == "mvtec" else args.K_top_visa

    out_dir = log_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"eval_step{args.ckpt_step}_alpha{args.alpha}.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "class", "I_AUROC_topk", "I_AUROC_cls", "I_AUROC_fused",
            "I_AUPR_fused", "P_AUROC", "P_AUPR",
        ])

    metrics_acc: Dict[str, List[float]] = {k: [] for k in
                                            ["I_AUROC_topk", "I_AUROC_cls", "I_AUROC_fused",
                                             "I_AUPR_fused", "P_AUROC", "P_AUPR"]}

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test", root=args.test_root, batch_size=1,
            classname=cls, resize=args.crop, datasetname=args.dataset,
        )

        topk_scores: List[float] = []
        cls_scores: List[float] = []
        labels: List[int] = []
        pix_buf, mask_buf = [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"]
            paths = batch["image_path"]
            labels.extend([int(x) for x in batch["is_anomaly"]])

            res = model.score(img, paths)
            patch_l = res["patch_agg"]                     # [B, P]
            topk = patch_l.topk(K, dim=1).values.mean(dim=1)
            topk_scores.extend(topk.cpu().numpy().tolist())

            cls_s = res["cls_agg"]                          # [B]
            cls_scores.extend(cls_s.cpu().numpy().tolist())

            P = patch_l.size(1)
            h = w = int(math.sqrt(P))
            pix = F.interpolate(patch_l.view(-1, 1, h, w),
                                size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu())
            mask_buf.append(mask)

        topk_arr = np.array(topk_scores); cls_arr = np.array(cls_scores)
        labels_arr = np.array(labels)
        pix_all = torch.cat(pix_buf).numpy()
        mask_all = torch.cat(mask_buf).squeeze(1).numpy()

        topk_n = (topk_arr - topk_arr.min()) / (topk_arr.max() - topk_arr.min() + 1e-8)
        cls_n = (cls_arr - cls_arr.min()) / (cls_arr.max() - cls_arr.min() + 1e-8)
        fused = args.alpha * cls_n + (1 - args.alpha) * topk_n if cls_layers else topk_n

        I_AUROC_topk = roc_auc_score(labels_arr, topk_n)
        I_AUROC_cls = roc_auc_score(labels_arr, cls_n) if cls_layers else 0.0
        I_AUROC_fused = roc_auc_score(labels_arr, fused)
        I_AUPR_fused = average_precision_score(labels_arr, fused)

        gmin, gmax = pix_all.min(), pix_all.max()
        pix_n = (pix_all - gmin) / (gmax - gmin + 1e-8)
        P_AUROC = roc_auc_score(mask_all.flatten() > 0, pix_n.flatten())
        P_AUPR = average_precision_score(mask_all.flatten() > 0, pix_n.flatten())

        row = [cls,
               f"{I_AUROC_topk:.4f}", f"{I_AUROC_cls:.4f}", f"{I_AUROC_fused:.4f}",
               f"{I_AUPR_fused:.4f}", f"{P_AUROC:.4f}", f"{P_AUPR:.4f}"]
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        for k, v in zip(metrics_acc.keys(),
                        [I_AUROC_topk, I_AUROC_cls, I_AUROC_fused, I_AUPR_fused, P_AUROC, P_AUPR]):
            metrics_acc[k].append(v)
        print(f"[{cls}] I_topk={I_AUROC_topk:.4f}  I_cls={I_AUROC_cls:.4f}  "
              f"I_fused={I_AUROC_fused:.4f}  P_AUROC={P_AUROC:.4f}  P_AUPR={P_AUPR:.4f}")

    means = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow(["MEAN"] + [f"{means[k]:.4f}" for k in
            ["I_AUROC_topk", "I_AUROC_cls", "I_AUROC_fused", "I_AUPR_fused", "P_AUROC", "P_AUPR"]])
    print("\n=== MEAN ===")
    for k, v in means.items():
        print(f"  {k} = {v:.4f}")
    print(f"\nSaved -> {out_csv}")


# ============================================================================ #
# CLI
# ============================================================================ #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    ap.add_argument("--train_root", default="")
    ap.add_argument("--test_root", default="")
    ap.add_argument("--diy_name", default="dm_full")
    ap.add_argument("--ckpt_step", type=int, default=0)

    ap.add_argument("--model", default="dinov3")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--seg_layers", default="1,4,7,10,13",
                    help="negative-from-last block indices for segmentation projectors")
    ap.add_argument("--cls_layers", default="1",
                    help="empty string '' disables CLS head; '1' = last block; '1,4' = last + 4th-from-last")
    ap.add_argument("--pred_depth", type=int, default=6)
    ap.add_argument("--pred_emb_dim", type=int, default=384)
    ap.add_argument("--cls_hidden", type=int, default=384)
    ap.add_argument("--cls_depth", type=int, default=2)
    ap.add_argument("--share_seg_backbone", action="store_true",
                    help="share the seg projector body across layers (DHF via score-end aggregation)")
    ap.add_argument("--if_pe", action="store_true")
    ap.add_argument("--feat_normed", action="store_true")

    ap.add_argument("--gate_p", type=float, default=0.5)
    ap.add_argument("--gate_c", type=float, default=0.5)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_every_steps", type=int, default=1000)

    ap.add_argument("--alpha", type=float, default=0.5,
                    help="image score fusion weight: S_image = alpha*cls + (1-alpha)*topk")
    ap.add_argument("--K_top_mvtec", type=int, default=10)
    ap.add_argument("--K_top_visa", type=int, default=6)
    ap.add_argument("--classnames_override", default="",
                    help="comma-separated class list for non-default datasets (e.g. MVTec LOCO)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.mode == "train":
        assert a.train_root, "--train_root required for train"
        train(a)
    else:
        assert a.train_root and a.test_root, "--train_root + --test_root required for eval"
        evaluate(a)
