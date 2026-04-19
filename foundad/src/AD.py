import csv
import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image

from src.datasets.dataset import build_dataloader
from src.utils.metrics import compute_ad_metrics_gpu
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule
from src.utils.pca import PCAScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _load_completed_eval_rows(
    csv_path: Path, ckpt_name: str
) -> Dict[str, Tuple[float, float, float, float]]:
    """Parse AD_eval.csv: rows for this checkpoint with per-class metrics (latest row wins)."""
    if not csv_path.is_file():
        return {}
    completed: Dict[str, Tuple[float, float, float, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            if row[0] == "checkpoint":
                continue
            if row[0] != ckpt_name or row[1] == "Mean":
                continue
            try:
                completed[row[1]] = (
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            except ValueError:
                continue
    return completed


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        multi_layer_agg=meta.get("multi_layer_agg", "none"),
    )


def _collect_normal_features(model, train_root, cls, crop, n_layer, device, n_aug=3):
    """Extract normal patch features for PCA fitting with rotation augmentation."""
    import torchvision.transforms as T
    from PIL import Image as PILImage

    img_dir = os.path.join(train_root, "train", cls)
    if not os.path.isdir(img_dir):
        logger.warning("PCA: train dir not found: %s", img_dir)
        return None

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = sorted(
        p for p in (os.path.join(img_dir, f) for f in os.listdir(img_dir))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    )
    if not img_paths:
        logger.warning("PCA: no images in %s", img_dir)
        return None

    transform = T.Compose([
        T.Resize((crop, crop)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    angles = [0.0]
    if n_aug > 0:
        step = 360.0 / (n_aug + 1)
        angles += [step * i for i in range(1, n_aug + 1)]

    all_feats = []
    for angle in angles:
        batch = []
        for p in img_paths:
            pil = PILImage.open(p).convert("RGB")
            if angle > 0:
                pil = pil.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            batch.append(transform(pil))
        imgs = torch.stack(batch).to(device)
        enc = model.target_features(imgs, img_paths, n_layer=n_layer)
        all_feats.append(enc.reshape(-1, enc.size(-1)).cpu())

    features = torch.cat(all_feats, dim=0)
    logger.info(
        "PCA [%s]: collected %d patch features (n_img=%d, n_aug=%d)",
        cls, features.size(0), len(img_paths), n_aug,
    )
    return features

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    top_ratio = cfg["testing"].get("top_ratio")
    pca_cfg = cfg.get("testing", {}).get("pca", {})
    use_pca = pca_cfg.get("enabled", False)
    pca_ev_ratio = pca_cfg.get("ev_ratio", 0.99)
    pca_weight = pca_cfg.get("score_weight", 0.3)
    pca_n_aug = pca_cfg.get("n_aug", 3)
    train_root = cfg.get("data", {}).get("train_root")

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = cfg["data"]["mvtec_classnames"] 
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    skip_done = cfg["testing"].get("skip_evaluated_classes", True)
    completed: Dict[str, Tuple[float, float, float, float]] = {}
    if skip_done:
        completed = _load_completed_eval_rows(csv_path, ckpt.name)
        if len(classnames) > 0 and all(c in completed for c in classnames):
            ia = [completed[c][0] for c in classnames]
            ip = [completed[c][1] for c in classnames]
            pa = [completed[c][2] for c in classnames]
            pr = [completed[c][3] for c in classnames]
            logger.info(
                "All %d classes already evaluated for %s in %s.",
                len(classnames), ckpt.name, csv_path,
            )
            logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                        np.mean(ia), np.mean(ip), np.mean(pa), np.mean(pr))
            return

    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        if skip_done and cls in completed:
            ia, ip, pa, pr = completed[cls]
            inst_auc.append(ia)
            inst_aupr.append(ip)
            pix_auc.append(pa)
            pro_auc.append(pr)
            logger.info(
                "Skipping %s (already in %s) | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                cls,
                csv_path.name,
                ia,
                ip,
                pa,
                pr,
            )
            continue

        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        pca_scorer = None
        if use_pca and train_root:
            feats = _collect_normal_features(model, train_root, cls, crop, n_layer, device, n_aug=pca_n_aug)
            if feats is not None:
                pca_scorer = PCAScorer(ev_ratio=pca_ev_ratio, device=device).fit(feats)

        foundad_img_scores, pca_img_scores, labels = [], [], []
        foundad_pix_buf, pca_pix_buf, img_buf, mask_buf, name_buf = [], [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            pred = model.predict(enc)

            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)

            n_top = max(1, int(l.size(1) * top_ratio)) if top_ratio is not None else K
            topk = torch.topk(l, n_top, dim=1).values.mean(dim=1)
            foundad_img_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            foundad_pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(img.cpu()); mask_buf.append(mask.cpu())

            if pca_scorer is not None:
                l_pca = pca_scorer.score(enc)
                topk_pca = torch.topk(l_pca, n_top, dim=1).values.mean(dim=1)
                pca_img_scores.extend(topk_pca.cpu())
                pix_pca = F.interpolate(l_pca.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
                pca_pix_buf.append(pix_pca.squeeze(1).cpu())

        fi = torch.tensor(foundad_img_scores).numpy()
        fi = (fi - fi.min()) / (fi.max() - fi.min() + 1e-8)
        fp_all = torch.cat(foundad_pix_buf)
        fp_min, fp_max = fp_all.min(), fp_all.max()
        fp_norm = ((fp_all - fp_min) / (fp_max - fp_min + 1e-8)).numpy()

        if pca_scorer is not None:
            pi = torch.tensor(pca_img_scores).numpy()
            pi = (pi - pi.min()) / (pi.max() - pi.min() + 1e-8)
            pp_all = torch.cat(pca_pix_buf)
            pp_min, pp_max = pp_all.min(), pp_all.max()
            pp_norm = ((pp_all - pp_min) / (pp_max - pp_min + 1e-8)).numpy()
            alpha = pca_weight
            p_np = (1 - alpha) * fi + alpha * pi
            pix_norm = (1 - alpha) * fp_norm + alpha * pp_norm
        else:
            p_np = fi
            pix_norm = fp_norm

        mask_np = torch.cat(mask_buf).squeeze(1).numpy()

        met = compute_ad_metrics_gpu(
            p_np, np.array(labels), pix_norm, mask_np,
            nstrips=cfg["testing"]["max_steps"],
        )

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, met["inst_auroc"], met["inst_aupr"], met["pix_auroc"], met["pro_auc"])
        csv_logger.log(ckpt.name, cls, met["inst_auroc"], met["inst_aupr"], met["pix_auroc"], met["pro_auc"])

        inst_auc.append(met["inst_auroc"]); inst_aupr.append(met["inst_aupr"])
        pix_auc.append(met["pix_auroc"]);   pro_auc.append(met["pro_auc"])

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("=" * 60)
    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    logger.info("=" * 60)
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        import cv2
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1, P, D]
        pred = model.predict(enc)                                       # [1, P, D]

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)         # [1, P]

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        save_dir = (out_root / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


@torch.inference_mode()
def evaluate_model(model, cfg, csv_logger=None, epoch=None):
    """Evaluate model in-place (no checkpoint I/O). Returns dict of mean metrics."""
    device = next(model.parameters()).device
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    top_ratio = cfg["testing"].get("top_ratio")
    pca_cfg = cfg.get("testing", {}).get("pca", {})
    use_pca = pca_cfg.get("enabled", False)
    pca_ev_ratio = pca_cfg.get("ev_ratio", 0.99)
    pca_weight = pca_cfg.get("score_weight", 0.3)
    pca_n_aug = pca_cfg.get("n_aug", 3)
    train_root = cfg.get("data", {}).get("train_root")

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == "mvtec":
        classnames = cfg["data"]["mvtec_classnames"]
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == "visa":
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    inst_auc, inst_aupr, pix_auc, pro_auc_list = [], [], [], []

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        pca_scorer = None
        if use_pca and train_root:
            feats = _collect_normal_features(model, train_root, cls, crop, n_layer, device, n_aug=pca_n_aug)
            if feats is not None:
                pca_scorer = PCAScorer(ev_ratio=pca_ev_ratio, device=device).fit(feats)

        foundad_img_scores, pca_img_scores, labels = [], [], []
        foundad_pix_buf, pca_pix_buf, mask_buf = [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]
            labels.extend(batch["is_anomaly"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            pred = model.predict(enc)

            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)
            n_top = max(1, int(l.size(1) * top_ratio)) if top_ratio is not None else K
            topk = torch.topk(l, n_top, dim=1).values.mean(dim=1)
            foundad_img_scores.extend(topk.cpu())

            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(
                l.view(-1, 1, h, w), size=img.shape[2:],
                mode="bilinear", align_corners=False,
            )
            foundad_pix_buf.append(pix.squeeze(1).cpu())
            mask_buf.append(mask.cpu())

            if pca_scorer is not None:
                l_pca = pca_scorer.score(enc)
                topk_pca = torch.topk(l_pca, n_top, dim=1).values.mean(dim=1)
                pca_img_scores.extend(topk_pca.cpu())
                pix_pca = F.interpolate(
                    l_pca.view(-1, 1, h, w), size=img.shape[2:],
                    mode="bilinear", align_corners=False,
                )
                pca_pix_buf.append(pix_pca.squeeze(1).cpu())

        fi = torch.tensor(foundad_img_scores).numpy()
        fi = (fi - fi.min()) / (fi.max() - fi.min() + 1e-8)
        fp_all = torch.cat(foundad_pix_buf)
        fp_min, fp_max = fp_all.min(), fp_all.max()
        fp_norm = ((fp_all - fp_min) / (fp_max - fp_min + 1e-8)).numpy()

        if pca_scorer is not None:
            pi = torch.tensor(pca_img_scores).numpy()
            pi = (pi - pi.min()) / (pi.max() - pi.min() + 1e-8)
            pp_all = torch.cat(pca_pix_buf)
            pp_min, pp_max = pp_all.min(), pp_all.max()
            pp_norm = ((pp_all - pp_min) / (pp_max - pp_min + 1e-8)).numpy()
            alpha = pca_weight
            p_np = (1 - alpha) * fi + alpha * pi
            pix_norm = (1 - alpha) * fp_norm + alpha * pp_norm
        else:
            p_np = fi
            pix_norm = fp_norm

        mask_np = torch.cat(mask_buf).squeeze(1).numpy()

        met = compute_ad_metrics_gpu(
            p_np, np.array(labels), pix_norm, mask_np,
            nstrips=cfg["testing"]["max_steps"],
        )

        logger.info(
            "  %s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
            cls, met["inst_auroc"], met["inst_aupr"], met["pix_auroc"], met["pro_auc"],
        )
        if csv_logger is not None and epoch is not None:
            csv_logger.log(epoch, cls, met["inst_auroc"], met["inst_aupr"], met["pix_auroc"], met["pro_auc"])

        inst_auc.append(met["inst_auroc"])
        inst_aupr.append(met["inst_aupr"])
        pix_auc.append(met["pix_auroc"])
        pro_auc_list.append(met["pro_auc"])

    means = {
        "inst_auroc": float(np.mean(inst_auc)),
        "inst_aupr": float(np.mean(inst_aupr)),
        "pix_auroc": float(np.mean(pix_auc)),
        "pro_auc": float(np.mean(pro_auc_list)),
    }

    logger.info(
        "  Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
        means["inst_auroc"], means["inst_aupr"], means["pix_auroc"], means["pro_auc"],
    )
    if csv_logger is not None and epoch is not None:
        csv_logger.log(epoch, "Mean",
                       means["inst_auroc"], means["inst_aupr"],
                       means["pix_auroc"], means["pro_auc"])

    return means


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()