#!/usr/bin/env bash
# Pilot 3 — DHF-only ablation (no CLS head; per-layer patch projectors).
# L_seg = {1, 4, 7, 10, 13} (last 5 blocks at strides ~3); L_cls = {} (disabled).
# Cost: ~4 GPU·h. Uses --share_seg_backbone to keep params reasonable (DHF via score-end agg).

set -euo pipefail
cd "$(dirname "$0")/../foundad"

# === EDIT ME ==================================================================
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/fewshot}"
DATA_NAME="${DATA_NAME:-mvtec_4shot}"
TEST_ROOT="${TEST_ROOT:-/root/autodl-tmp/dataset/mvtec}"
DATASET="${DATASET:-mvtec}"
# ==============================================================================

DIY="${DIY:-pilot3_dhf_only}"
EPOCHS="${EPOCHS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"
SEG_LAYERS="${SEG_LAYERS:-1,4,7,10,12}"

if [ ! -d "$DATA_PATH/$DATA_NAME/train" ]; then
    echo "[setup] sampling few-shot folder ..."
    python -m src.sample \
        source="$TEST_ROOT" \
        target="$DATA_PATH/$DATA_NAME" \
        seed="$SEED" \
        num_samples=4
fi

# Train: DHF-only, share backbone to keep param budget close to FoundAD baseline
python -m src.dm_train \
    --mode train --dataset "$DATASET" \
    --train_root "$DATA_PATH/$DATA_NAME" \
    --diy_name "$DIY" \
    --seg_layers "$SEG_LAYERS" --cls_layers "" \
    --share_seg_backbone \
    --epochs "$EPOCHS" --batch_size "$BATCH" --save_every_steps "$SAVE_EVERY" \
    --seed "$SEED"

LAST_CKPT=$(ls -1 "logs/$DATA_NAME/dm$DIY/" | grep '^dm-step' | sed 's/[^0-9]//g' | sort -n | tail -1)
echo "[eval] using ckpt step $LAST_CKPT"

# alpha=0 forces score = topk(patch_agg) since cls head disabled
python -m src.dm_train \
    --mode eval --dataset "$DATASET" \
    --train_root "$DATA_PATH/$DATA_NAME" \
    --test_root "$TEST_ROOT" \
    --diy_name "$DIY" --ckpt_step "$LAST_CKPT" \
    --seg_layers "$SEG_LAYERS" --cls_layers "" \
    --share_seg_backbone \
    --alpha 0.0 \
    --seed "$SEED"

echo "=== Pilot 3 done. Inspect logs/$DATA_NAME/dm$DIY/eval/*.csv ==="
echo "Decision gate (per IDEA_REPORT.md §5):"
echo "  P_AUPR over FoundAD baseline >= +3 -> DHF lever validated"
echo "  Image-only gain, no pixel gain      -> DHF reduces to score-end averaging; reconsider granularity"
