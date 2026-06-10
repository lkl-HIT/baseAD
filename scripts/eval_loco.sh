#!/usr/bin/env bash
# MVTec LOCO logical-anomaly evaluation for DM-FoundAD.
# Reuses the dm_train.py eval mode but points at LOCO. The dataset folder layout is
#   <loco_root>/<class>/test/{logical_anomalies,structural_anomalies,good}/...
# We rely on dataset.py treating any subdir != good/ok as anomaly (per build_dataloader).
# Cost: ~4 GPU·h.

set -euo pipefail
cd "$(dirname "$0")/../foundad"

LOCO_ROOT="${LOCO_ROOT:-/root/autodl-tmp/dataset/mvtec_loco}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/fewshot}"
DATA_NAME="${DATA_NAME:-mvtec_loco_4shot}"
DIY="${DIY:-dm_full_loco}"
SEG_LAYERS="${SEG_LAYERS:-1,4,7,10,12}"
CLS_LAYERS="${CLS_LAYERS:-1}"
EPOCHS="${EPOCHS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"
LOCO_CLASSES="breakfast_box,juice_bottle,pushpins,screw_bag,splicing_connectors"

if [ ! -d "$DATA_PATH/$DATA_NAME/train" ]; then
    echo "[setup] sampling LOCO few-shot folder ..."
    python -m src.sample \
        source="$LOCO_ROOT" \
        target="$DATA_PATH/$DATA_NAME" \
        seed="$SEED" \
        num_samples=4
fi

# Train DM-FoundAD on LOCO normals
python -m src.dm_train \
    --mode train --dataset mvtec \
    --train_root "$DATA_PATH/$DATA_NAME" \
    --diy_name "$DIY" \
    --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --save_every_steps "$SAVE_EVERY" \
    --seed "$SEED"

LAST_CKPT=$(ls -1 "logs/$DATA_NAME/dm$DIY/" | grep '^dm-step' | sed 's/[^0-9]//g' | sort -n | tail -1)
echo "[eval] using ckpt step $LAST_CKPT"

# Evaluate at α ∈ {0.0, 0.5} so we can attribute LOCO gain to the CLS head specifically
for ALPHA in 0.0 0.5; do
    python -m src.dm_train \
        --mode eval --dataset mvtec \
        --train_root "$DATA_PATH/$DATA_NAME" \
        --test_root "$LOCO_ROOT" \
        --diy_name "$DIY" --ckpt_step "$LAST_CKPT" \
        --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
        --classnames_override "$LOCO_CLASSES" \
        --alpha "$ALPHA" \
        --seed "$SEED"
done

echo "=== LOCO done. Compare α=0.0 (patch only) vs α=0.5 (with CLS head). ==="
echo "If CLS head genuinely helps logical anomalies, α=0.5 should beat α=0.0 by >= +5 I-AUROC on LOCO."
