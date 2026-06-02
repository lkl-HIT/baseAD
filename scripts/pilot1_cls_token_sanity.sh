#!/usr/bin/env bash
# Pilot 1 — Zero-shot CLS-token cosine AUROC on MVTec/VisA.
# Goal: verify a frozen DINOv3 CLS token can separate normal vs anomaly given K normal references.
# Cost: ~1 GPU·h, no training.
#
# Edit DATASET_ROOT / VISA_ROOT below to match your machine layout. Default seed=42, K=4.
# AGENTS.md convention: this script is hand-off; the compute machine runs `bash scripts/pilot1_cls_token_sanity.sh`.

set -euo pipefail
cd "$(dirname "$0")/../foundad"

# === EDIT ME (per machine) =====================================================
MVTEC_ROOT="${MVTEC_ROOT:-/root/autodl-tmp/dataset/mvtec}"
VISA_ROOT="${VISA_ROOT:-/root/autodl-tmp/dataset/visa/visa_pytorch}"
# ==============================================================================

K="${K:-4}"
SEED="${SEED:-42}"
CROP="${CROP:-512}"
OUT_DIR="logs/pilot1_cls_sanity"

mkdir -p "$OUT_DIR"

echo "=== Pilot 1: MVTec K=$K seed=$SEED crop=$CROP ==="
python -m src.pilots.cls_sanity \
    --dataset mvtec \
    --test_root "$MVTEC_ROOT" \
    --K "$K" \
    --seed "$SEED" \
    --crop "$CROP" \
    --out_csv "$OUT_DIR/mvtec_k${K}_seed${SEED}.csv"

echo
echo "=== Pilot 1: VisA K=$K seed=$SEED crop=$CROP ==="
python -m src.pilots.cls_sanity \
    --dataset visa \
    --test_root "$VISA_ROOT" \
    --K "$K" \
    --seed "$SEED" \
    --crop "$CROP" \
    --out_csv "$OUT_DIR/visa_k${K}_seed${SEED}.csv"

echo
echo "=== Pilot 1 done. Inspect $OUT_DIR/*.csv ==="
echo "Decision gate (per IDEA_REPORT.md §5):"
echo "  CLS-AUROC mean >= 75  -> GO Idea 1 (DM-FoundAD)"
echo "  60 <= CLS-AUROC < 75  -> DEGRADE to Idea 3 (pooled global token)"
echo "  CLS-AUROC < 60        -> DROP CLS head, keep DHF only (Idea 2)"
