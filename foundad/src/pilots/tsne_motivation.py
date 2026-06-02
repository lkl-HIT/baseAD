"""Motivation t-SNE figure: visualize CLS / patch-l manifolds on normal vs anomaly.

For the paper §3 motivation. No training. ~1 GPU·h.

Produces three subplots per class (default: hazelnut, bottle):
  1. CLS-token (last block) — normal blue, anomaly red
  2. patch-mean (last block)
  3. patch-mean (block 13 = 12-from-last for ViT-L/16)

Saves a single PNG: assets/motivation_tsne_<class>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_dinov3(device: torch.device):
    enc = torch.hub.load(
        "/root/.cache/torch/hub/facebookresearch_dinov3_main",
        "dinov3_vitl16", source="local",
    ).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device)


@torch.inference_mode()
def embed(enc, paths: List[Path], crop: int, layers: List[int], device, batch: int = 8):
    """Returns dict: {f'cls_l{l}': [N,D], f'patch_l{l}': [N,D]} (patch is mean-pooled)."""
    tfm = transforms.Compose([
        transforms.Resize((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    n = max(layers + [1])
    out = {f"cls_l{l}": [] for l in layers}
    out.update({f"patch_l{l}": [] for l in layers})
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        outs = enc.get_intermediate_layers(imgs, n=n, return_class_token=True)
        for l in layers:
            patches, cls = outs[-l]
            out[f"cls_l{l}"].append(cls.float().cpu().numpy())
            out[f"patch_l{l}"].append(patches.float().mean(dim=1).cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def plot_tsne(emb_norm: np.ndarray, emb_ano: np.ndarray, ax, title: str):
    n_n, n_a = len(emb_norm), len(emb_ano)
    X = np.concatenate([emb_norm, emb_ano], 0)
    perp = max(5, min(30, (n_n + n_a) // 4))
    Z = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca").fit_transform(X)
    ax.scatter(Z[:n_n, 0], Z[:n_n, 1], s=12, c="#1f77b4", alpha=0.7, label=f"normal ({n_n})")
    ax.scatter(Z[n_n:, 0], Z[n_n:, 1], s=12, c="#d62728", alpha=0.7, label=f"anomaly ({n_a})")
    ax.set_title(title); ax.legend(loc="best", fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def run(test_root: Path, classnames: List[str], crop: int, layers: List[int],
        max_normal: int, max_anomaly: int, out_dir: Path) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = load_dinov3(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls in classnames:
        cls_dir = test_root / cls / "test"
        normal_paths: List[Path] = []
        anomaly_paths: List[Path] = []
        if not cls_dir.is_dir():
            print(f"[skip] {cls}"); continue
        for sub in sorted(cls_dir.iterdir()):
            if not sub.is_dir():
                continue
            imgs = list_images(sub)
            if sub.name in ("good", "ok"):
                normal_paths.extend(imgs)
            else:
                anomaly_paths.extend(imgs)
        normal_paths = normal_paths[:max_normal]
        anomaly_paths = anomaly_paths[:max_anomaly]
        if not normal_paths or not anomaly_paths:
            print(f"[skip] {cls}: insufficient samples"); continue

        emb_n = embed(enc, normal_paths, crop, layers, device)
        emb_a = embed(enc, anomaly_paths, crop, layers, device)

        n_panels = 2 * len(layers)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.0))
        if n_panels == 1:
            axes = [axes]
        for i, l in enumerate(layers):
            plot_tsne(emb_n[f"cls_l{l}"], emb_a[f"cls_l{l}"], axes[2 * i],
                      f"{cls} — CLS L-{l}")
            plot_tsne(emb_n[f"patch_l{l}"], emb_a[f"patch_l{l}"], axes[2 * i + 1],
                      f"{cls} �� PatchMean L-{l}")
        fig.tight_layout()
        out_png = out_dir / f"motivation_tsne_{cls}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"[ok] {out_png}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="MVTec root")
    ap.add_argument("--classes", default="hazelnut,bottle,transistor,cable")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--layers", default="1,13",
                    help="negative-from-last block indices to visualize")
    ap.add_argument("--max_normal", type=int, default=100)
    ap.add_argument("--max_anomaly", type=int, default=80)
    ap.add_argument("--out_dir", default="assets/motivation_tsne")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    classnames = [c.strip() for c in a.classes.split(",") if c.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    run(Path(a.test_root), classnames, a.crop, layers, a.max_normal, a.max_anomaly, Path(a.out_dir))
