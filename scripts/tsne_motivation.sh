#!/usr/bin/env bash
# Motivation t-SNE figure for the paper §3.
# Visualizes CLS-token vs patch-mean manifolds at last block AND at block-13-from-last,
# on normal vs anomaly samples from a few representative MVTec classes.
# No training. ~1 GPU·h total.

set -euo pipefail
cd "$(dirname "$0")/../foundad"

MVTEC_ROOT="${MVTEC_ROOT:-/root/autodl-tmp/dataset/mvtec}"
CLASSES="${CLASSES:-hazelnut,bottle,transistor,cable,zipper}"
LAYERS="${LAYERS:-1,13}"

python -m src.pilots.tsne_motivation \
    --test_root "$MVTEC_ROOT" \
    --classes "$CLASSES" \
    --layers "$LAYERS" \
    --max_normal 100 \
    --max_anomaly 80 \
    --out_dir assets/motivation_tsne

echo "=== t-SNE done. Figures in assets/motivation_tsne/*.png ==="
echo "Use these as paper Fig 1 (or §3 motivation) — CLS / patch / per-layer manifold separation."
