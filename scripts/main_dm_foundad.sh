#!/usr/bin/env bash
# Main run — full DM-FoundAD: DCS + DHF on DINOv3 ViT-L/16, normal-only few-shot.
# L_seg = {1,4,7,10,13}, L_cls = {1}. Sweep K ∈ {1, 2, 4} on MVTec, VisA.
# Cost: ~8 GPU·h.

set -euo pipefail
cd "$(dirname "$0")/../foundad"

# === EDIT ME ==================================================================
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/dataset/fewshot}"
MVTEC_TEST="${MVTEC_TEST:-/root/autodl-tmp/dataset/mvtec}"
VISA_TEST="${VISA_TEST:-/root/autodl-tmp/dataset/visa/visa_pytorch}"
# ==============================================================================

DIY="${DIY:-dm_full}"
EPOCHS="${EPOCHS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"
SEG_LAYERS="${SEG_LAYERS:-1,4,7,10,13}"
CLS_LAYERS="${CLS_LAYERS:-1}"
ALPHA="${ALPHA:-0.5}"

run_one () {
    local DATASET="$1"
    local TEST_ROOT="$2"
    local K="$3"

    local DATA_NAME="${DATASET}_${K}shot"
    if [ ! -d "$DATA_PATH/$DATA_NAME/train" ]; then
        echo "[setup] sampling $DATA_NAME ..."
        python -m src.sample \
            source="$TEST_ROOT" \
            target="$DATA_PATH/$DATA_NAME" \
            seed="$SEED" \
            num_samples="$K"
    fi

    python -m src.dm_train \
        --mode train --dataset "$DATASET" \
        --train_root "$DATA_PATH/$DATA_NAME" \
        --diy_name "${DIY}_${DATASET}_${K}shot" \
        --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --save_every_steps "$SAVE_EVERY" \
        --seed "$SEED"

    local CKPT
    CKPT=$(ls -1 "logs/$DATA_NAME/dm${DIY}_${DATASET}_${K}shot/" | grep '^dm-step' | sed 's/[^0-9]//g' | sort -n | tail -1)

    python -m src.dm_train \
        --mode eval --dataset "$DATASET" \
        --train_root "$DATA_PATH/$DATA_NAME" \
        --test_root "$TEST_ROOT" \
        --diy_name "${DIY}_${DATASET}_${K}shot" --ckpt_step "$CKPT" \
        --seg_layers "$SEG_LAYERS" --cls_layers "$CLS_LAYERS" \
        --alpha "$ALPHA" \
        --seed "$SEED"
}

for K in 1 2 4; do
    run_one mvtec "$MVTEC_TEST" "$K"
    run_one visa  "$VISA_TEST"  "$K"
done

echo "=== Main DM-FoundAD done. Inspect logs/*/dm${DIY}_*/eval/*.csv ==="
echo "Compare against FoundAD baseline checkpoints (same K) — see refine-logs/EXPERIMENT_PLAN.md."
