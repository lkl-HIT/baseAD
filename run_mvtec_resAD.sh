#!/bin/bash
# ================================================================
# FoundAD feature_resAD — MVTec 全量实验脚本
#
# 内容：
#   1) 训练 baseline (single-layer)       2000 epochs
#   2) 训练 multi-layer (mean-pool)        2000 epochs
#   3) 在 step 10000 / 14000 上用 4 种推理配置评估
#      - 原始 FoundAD
#      - + top-ratio (1%)
#      - + PCA (weight=0.3)
#      - + PCA + top-ratio
#
# MVTec 4-shot: 15类 × 4张 = 60张, bs=8 → 7 itr/epoch
# 2000 epochs × 7 = 14000 steps, 每 2000 步存一个 ckpt
#
# 预计总时间: 6-8 小时 (取决于 GPU)
# 用法: bash run_mvtec_resAD.sh
# ================================================================

set -uo pipefail

# ============ 公共配置 ============
EPOCHS=2000
SAVE_EVERY=2000
BS=8
DATASET=mvtec
DATA_NAME=mvtec_4shot
DATA_PATH=/root/autodl-tmp/dataset/fewshot
TEST_ROOT=/root/autodl-tmp/dataset/mvtec

TS=$(date +%m%d_%H%M)
LOG="experiment/resAD_mvtec_${TS}"
mkdir -p "$LOG"

log() { echo ""; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "日志目录: $LOG"
log "开始时间: $(date)"

# ============ 阶段 1: 训练 baseline ============
log "========== [1/3] 训练 baseline (single-layer, 2000 epochs) =========="
python foundad/main.py mode=train \
  data.batch_size=$BS \
  data.dataset=$DATASET \
  data.data_name=$DATA_NAME \
  data.data_path=$DATA_PATH \
  data.test_root=$TEST_ROOT \
  app=train_dinov3 \
  diy_name=mvtec_4_base \
  optimization.epochs=$EPOCHS \
  optimization.save_every_steps=$SAVE_EVERY \
  2>&1 | tee "$LOG/train_base.log"

# ============ 阶段 2: 训练 multi-layer ============
log "========== [2/3] 训练 multi-layer (mean-pool, 2000 epochs) =========="
python foundad/main.py mode=train \
  data.batch_size=$BS \
  data.dataset=$DATASET \
  data.data_name=$DATA_NAME \
  data.data_path=$DATA_PATH \
  data.test_root=$TEST_ROOT \
  app=train_dinov3 \
  app.meta.multi_layer_agg=mean \
  diy_name=mvtec_4_ml \
  optimization.epochs=$EPOCHS \
  optimization.save_every_steps=$SAVE_EVERY \
  2>&1 | tee "$LOG/train_ml.log"

# ============ 阶段 3: 评估 ============
log "========== [3/3] 评估所有配置 =========="

EVAL_STEPS=(10000 14000)

run_eval() {
  local diy=$1 step=$2 tag=$3; shift 3
  local ckpt_file="logs/${DATA_NAME}/dinov3${diy}/train-step${step}.pth.tar"
  if [ ! -f "$ckpt_file" ]; then
    log "  跳过 ${diy} step=${step} — ckpt 不存在: ${ckpt_file}"
    return 0
  fi
  log "  eval ${diy} step=${step} [${tag}]"
  python foundad/main.py mode=AD \
    data.dataset=$DATASET \
    data.data_name=$DATA_NAME \
    diy_name=$diy \
    data.test_root=$TEST_ROOT \
    app=test \
    app.ckpt_step=$step \
    testing.skip_evaluated_classes=false \
    testing.segmentation_vis=False \
    "$@" 2>&1 | tee "$LOG/eval_${diy}_s${step}_${tag}.log"
}

for STEP in "${EVAL_STEPS[@]}"; do
  for DIY in mvtec_4_base mvtec_4_ml; do
    log "====== ${DIY} @ step ${STEP} ======"

    # A) 原始 FoundAD
    run_eval "$DIY" "$STEP" "foundad"

    # B) + top-ratio 1%
    run_eval "$DIY" "$STEP" "topR" \
      testing.top_ratio=0.01

    # C) + PCA (weight=0.3)
    run_eval "$DIY" "$STEP" "pca" \
      testing.pca.enabled=true \
      testing.pca.score_weight=0.3

    # D) + PCA + top-ratio
    run_eval "$DIY" "$STEP" "pca_topR" \
      testing.pca.enabled=true \
      testing.pca.score_weight=0.3 \
      testing.top_ratio=0.01
  done
done

# ============ 汇总 ============
log "全部完成！结束时间: $(date)"
echo ""
echo "============================================================"
echo "  各配置 Mean 指标汇总"
echo "============================================================"
printf "%-50s %s\n" "实验" "AUROC_i | AUPR_i | AUROC_p | PRO-AUC"
echo "------------------------------------------------------------"
for f in "$LOG"/eval_*.log; do
  [ -f "$f" ] || continue
  tag=$(basename "$f" .log | sed 's/^eval_//')
  mean_line=$(grep "Mean |" "$f" | tail -1 || true)
  if [ -n "$mean_line" ]; then
    printf "%-50s %s\n" "$tag" "$mean_line"
  fi
done
echo "============================================================"
