#!/usr/bin/env bash
# Pilot 2 — DCS-only ablation (single-layer DM-FoundAD).
# L_seg = L_cls = {last block}; isolates the DCS lever from DHF.
# Cost: ~4 GPU·h on MVTec 4-shot (15 classes × 2000 epochs × small model).

set -euo pipefail
cd "$(dirname "$0")/../foundad"

# === EDIT ME ==================================================================
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/fewshot}"   # contains mvtec_4shot/
DATA_NAME="${DATA_NAME:-mvtec_4shot}"
TEST_ROOT="${TEST_ROOT:-/root/autodl-tmp/dataset/mvtec}"
DATASET="${DATASET:-mvtec}"
# ==============================================================================

DIY="${DIY:-pilot2_dcs_only}"
EPOCHS="${EPOCHS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"

# Step 1: prep few-shot folder if missing
if [ ! -d "$DATA_PATH/$DATA_NAME/train" ]; then
    echo "[setup] sampling few-shot folder ..."
    python -m src.sample \
        source="$TEST_ROOT" \
        target="$DATA_PATH/$DATA_NAME" \
        seed="$SEED" \
        num_samples=4
fi

# Step 2: train DCS-only (single layer, both heads on layer 1)
python -m src.dm_train \
    --mode train --dataset "$DATASET" \
    --train_root "$DATA_PATH/$DATA_NAME" \
    --diy_name "$DIY" \
    --seg_layers "1" --cls_layers "1" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --save_every_steps "$SAVE_EVERY" \
    --seed "$SEED"

# Step 3: evaluate at last checkpoint, sweep fusion weight α
LAST_STEP=$((EPOCHS * 2))   # epochs * (4_shot * 15_classes / batch=8) ≈ varies; replace below if needed
LAST_CKPT=$(ls -1 "logs/$DATA_NAME/dm$DIY/" | grep '^dm-step' | sed 's/[^0-9]//g' | sort -n | tail -1)
echo "[eval] using ckpt step $LAST_CKPT"

for ALPHA in 0.3 0.5 0.7; do
    python -m src.dm_train \
        --mode eval --dataset "$DATASET" \
        --train_root "$DATA_PATH/$DATA_NAME" \
        --test_root "$TEST_ROOT" \
        --diy_name "$DIY" --ckpt_step "$LAST_CKPT" \
        --seg_layers "1" --cls_layers "1" \
        --alpha "$ALPHA" \
        --seed "$SEED"
done

echo "=== Pilot 2 done. Inspect logs/$DATA_NAME/dm$DIY/eval/*.csv ==="
echo "Decision gate (per IDEA_REPORT.md §5):"
echo "  I_AUROC_fused − I_AUROC_topk >= +1.5 -> DCS lever validated, proceed Pilot 3"
echo "  +0.3 .. +1.5                          -> marginal, need DHF to amplify"
echo "  < +0.3                                -> DCS not working, fall back to Idea 4 (memory-side DCS)"
