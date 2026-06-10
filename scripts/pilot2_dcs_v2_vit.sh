#!/usr/bin/env bash
# Pilot 2 v2 — DCS ablation with ClsViTProjector (ViT-style, multi-layer CLS).
# L_seg = {1}; L_cls = {1,4} (2-layer CLS tokens → Self-Attention across layers).
# Goal: verify that ViT-style CLS projector + multi-layer CLS improves over old MLP.
# Quick run: 300 epochs (~2100 steps), eval at last ckpt.

set -euo pipefail
cd "$(dirname "$0")/../foundad"

# === EDIT ME ==================================================================
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/fewshot}"
DATA_NAME="${DATA_NAME:-mvtec_4shot}"
TEST_ROOT="${TEST_ROOT:-/root/autodl-tmp/dataset/mvtec}"
DATASET="${DATASET:-mvtec}"
# ==============================================================================

DIY="${DIY:-pilot2_dcs_v2_vit}"
EPOCHS="${EPOCHS:-300}"
SAVE_EVERY="${SAVE_EVERY:-300}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"

SEG_LAYERS="1"
CLS_LAYERS="1,4"

if [ ! -d "$DATA_PATH/$DATA_NAME/train" ]; then
    echo "[setup] sampling few-shot folder ..."
    python -m src.sample \
        source="$TEST_ROOT" \
        target="$DATA_PATH/$DATA_NAME" \
        seed="$SEED" \
        num_samples=4
fi

# Train: DCS with ViT-style CLS projector, 2-layer CLS tokens
echo "=== Training DCS v2 (ViT CLS projector, cls_layers=$CLS_LAYERS) ==="
python -m src.dm_train \
    --mode train --dataset "$DATASET" \
    --train_root "$DATA_PATH/$DATA_NAME" \
    --diy_name "$DIY" \
    --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
    --cls_depth 2 \
    --epochs "$EPOCHS" --batch_size "$BATCH" --save_every_steps "$SAVE_EVERY" \
    --seed "$SEED"

# Find last checkpoint
LAST_CKPT=$(ls -1 "logs/$DATA_NAME/dm$DIY/" | grep '^dm-step' | sed 's/[^0-9]//g' | sort -n | tail -1)
echo "[eval] using ckpt step $LAST_CKPT"

# Evaluate with multiple alpha values
for ALPHA in 0.3 0.5 0.7; do
    echo "=== Evaluating alpha=$ALPHA ==="
    python -m src.dm_train \
        --mode eval --dataset "$DATASET" \
        --train_root "$DATA_PATH/$DATA_NAME" \
        --test_root "$TEST_ROOT" \
        --diy_name "$DIY" --ckpt_step "$LAST_CKPT" \
        --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
        --alpha "$ALPHA" \
        --seed "$SEED"
done

echo "=== Pilot 2 v2 done. Inspect logs/$DATA_NAME/dm$DIY/eval/*.csv ==="
echo "Compare with old Pilot2:"
echo "  Old: I_AUROC_cls=70.51%, I_AUROC_fused=93.54% (alpha=0.5)"
echo "  Target: I_AUROC_cls > 80%, I_AUROC_fused > 95.5%"