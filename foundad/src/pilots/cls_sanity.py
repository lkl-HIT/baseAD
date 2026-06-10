"""Pilot 1: zero-shot CLS-token cosine AUROC on MVTec / VisA.

Goal: verify that a frozen DINOv3 ViT-L/16 CLS token, with k normal references,
can separate normal vs anomaly at image level. This is the GO/NO-GO gate for
the CLS-head branch of DM-FoundAD.

No training. ~1 GPU·h. Reuses VisionModule for backbone loading; only requires
return_class_token=True (added below via a forked _extract path).

Pass/fail (per AGENTS.md, machine-side will report; planning-side decides):
    CLS_AUROC_mean >= 75 -> GO  (Idea 1 / DM-FoundAD)
    60 <= CLS_AUROC_mean < 75 -> DEGRADE (Idea 3, pooled global token)
    CLS_AUROC_mean < 60 -> DROP CLS head (Idea 2, DHF-only)

Outputs CSV: logs/pilot1_cls_sanity/<dataset>_k<K>_seed<S>.csv
    columns: class, cls_auroc, patchmean_auroc, n_normal, n_test, n_anomaly
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _gather_class_split(test_root: Path, cls: str, dataset: str) -> Tuple[List[Path], List[Path], List[Path]]:
    """Return (normals_for_reference_pool, normal_test, anomaly_test).

    Reference pool comes from `train/good` (or `train/ok`); test split comes from
    `test/<subdir>` where `good`/`ok` is the normal label.
    """
    cls_dir = test_root / cls
    train_dir = cls_dir / "train"
    test_dir = cls_dir / "test"

    ref_pool: List[Path] = []
    for sub in ("good", "ok"):
        ref_pool = _list_images(train_dir / sub)
        if ref_pool:
            break

    normal_test: List[Path] = []
    anomaly_test: List[Path] = []
    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            imgs = _list_images(sub)
            if sub.name in ("good", "ok"):
                normal_test.extend(imgs)
            else:
                anomaly_test.extend(imgs)
    return ref_pool, normal_test, anomaly_test


def _build_transform(crop: int):
    return transforms.Compose([
        transforms.Resize((crop, crop), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_dinov3(device: torch.device):
    enc = torch.hub.load(
        "/root/.cache/torch/hub/facebookresearch_dinov3_main",
        "dinov3_vitb16",
        source="local",
    ).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device)


@torch.inference_mode()
def _extract_cls_and_patchmean(enc, img_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cls_emb [B,D], patchmean_emb [B,D]) from last block."""
    out = enc.get_intermediate_layers(img_batch, n=1, return_class_token=True)[0]
    if isinstance(out, tuple):
        patches, cls = out
    else:
        patches = out[:, 1:, :]
        cls = out[:, 0, :]
    patch_mean = patches.mean(dim=1)
    return cls, patch_mean


def _embed(enc, paths: List[Path], crop: int, device: torch.device, batch: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    tfm = _build_transform(crop)
    cls_all, pm_all = [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        cls, pm = _extract_cls_and_patchmean(enc, imgs)
        cls_all.append(cls.float().cpu().numpy())
        pm_all.append(pm.float().cpu().numpy())
    return np.concatenate(cls_all, 0), np.concatenate(pm_all, 0)


def _cosine_score(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """1 - max cosine similarity to any ref. Higher = more anomalous."""
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    sim = qn @ rn.T
    return 1.0 - sim.max(axis=1)


def run(dataset: str, test_root: Path, K: int, seed: int, crop: int, out_csv: Path) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    classes = MVTEC_CLASSES if dataset == "mvtec" else VISA_CLASSES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = _load_dinov3(device)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [["class", "cls_auroc", "patchmean_auroc", "n_normal_ref", "n_test_normal", "n_test_anomaly"]]

    cls_aurocs, pm_aurocs = [], []
    for cls in classes:
        ref_pool, normal_test, anomaly_test = _gather_class_split(test_root, cls, dataset)
        if not ref_pool or not anomaly_test:
            print(f"[skip] {cls}: ref_pool={len(ref_pool)} anomaly_test={len(anomaly_test)}")
            continue

        random.shuffle(ref_pool)
        ref_paths = ref_pool[:K]
        test_paths = normal_test + anomaly_test
        labels = np.array([0] * len(normal_test) + [1] * len(anomaly_test))

        cls_ref, pm_ref = _embed(enc, ref_paths, crop, device)
        cls_test, pm_test = _embed(enc, test_paths, crop, device)

        cls_score = _cosine_score(cls_test, cls_ref)
        pm_score = _cosine_score(pm_test, pm_ref)

        cls_auroc = roc_auc_score(labels, cls_score)
        pm_auroc = roc_auc_score(labels, pm_score)
        cls_aurocs.append(cls_auroc)
        pm_aurocs.append(pm_auroc)

        rows.append([cls, f"{cls_auroc:.4f}", f"{pm_auroc:.4f}",
                     len(ref_paths), len(normal_test), len(anomaly_test)])
        print(f"[{cls}] CLS-AUROC={cls_auroc:.4f}  PatchMean-AUROC={pm_auroc:.4f}")

    rows.append(["MEAN",
                 f"{np.mean(cls_aurocs):.4f}",
                 f"{np.mean(pm_aurocs):.4f}",
                 K, "-", "-"])
    print(f"\n=== {dataset} K={K} seed={seed} ===")
    print(f"Mean CLS-AUROC      = {np.mean(cls_aurocs):.4f}")
    print(f"Mean PatchMean-AUROC = {np.mean(pm_aurocs):.4f}")

    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved -> {out_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    ap.add_argument("--test_root", required=True, help="dataset root, e.g. /root/autodl-tmp/dataset/mvtec")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--out_csv", default="logs/pilot1_cls_sanity/result.csv")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.dataset, Path(a.test_root), a.K, a.seed, a.crop, Path(a.out_csv))
