# FoundAD 命令备忘

## 训练 (experiment 模式，边训练边评估)

```bash
# VisA 4-shot
python foundad/main.py mode=train \
  data.batch_size=8 \
  data.dataset=visa \
  data.data_name=visa_4shot \
  data.data_path=/root/autodl-tmp/dataset/fewshot \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=train_dinov3 \
  diy_name=visa_4_test \
  experiment.enabled=true \
  experiment.eval_every_n_epochs=1 \
  optimization.epochs=40
```

日志输出在 `experiment/logs/visa_4shot/dinov3visa_4_test/` 下：
- `train.csv` — 训练 loss
- `eval.csv` — 每次评估的各类别指标

---

## 评估 (已有权重)

### MVTec 4-shot — dinov3mvtec_4_test (step 11000~14000)

```bash
# 评估最优 step (14000)
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=14000 \
  testing.segmentation_vis=False
# 评估其他 step（按需替换）
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=12000
```

### MVTec 4-shot — dinov3_pretrained (官方预训练权重)

```bash
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=_pretrained \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=pretrained
```

### VisA 4-shot — dinov3visa_4_test (step 1000~12000)

```bash
# 评估最优 step (12000)
python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=12000

# 评估其他 step（按需替换）
python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=8000
```

### 带可视化的评估（生成 segmentation heatmap）

```bash
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=14000 \
  testing.segmentation_vis=True
```

---

## feature_resAD 分支：增强实验

> 三个改动独立可组合：multi-layer 需重训练，PCA 和 top-ratio 可直接用在已有权重上。

### 多层特征聚合训练 (multi_layer_agg=mean)

```bash
# VisA 4-shot，多层 mean-pool（需重训练）
python foundad/main.py mode=train \
  data.batch_size=8 \
  data.dataset=visa \
  data.data_name=visa_4shot \
  data.data_path=/root/autodl-tmp/dataset/fewshot \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=train_dinov3 \
  app.meta.multi_layer_agg=mean \
  diy_name=visa_4_ml \
  experiment.enabled=true \
  experiment.eval_every_n_epochs=1 \
  optimization.epochs=500

# MVTec 4-shot，多层 mean-pool（需重训练）
python foundad/main.py mode=train \
  data.batch_size=8 \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  data.data_path=/root/autodl-tmp/dataset/fewshot \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=train_dinov3 \
  app.meta.multi_layer_agg=mean \
  diy_name=mvtec_4_ml \
  experiment.enabled=true \
  experiment.eval_every_n_epochs=1 \
  optimization.epochs=40
```

### 在已有权重上启用 PCA 辅助打分 + top-ratio（无需重训练）

```bash
# VisA — baseline + PCA + top-ratio
python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=12000 \
  testing.pca.enabled=true \
  testing.pca.score_weight=0.3 \
  testing.pca.ev_ratio=0.99 \
  testing.pca.n_aug=3 \
  testing.top_ratio=0.01

# MVTec — baseline + PCA + top-ratio
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=14000 \
  testing.pca.enabled=true \
  testing.pca.score_weight=0.3 \
  testing.pca.ev_ratio=0.99 \
  testing.pca.n_aug=3 \
  testing.top_ratio=0.01
```

### 仅 top-ratio（无需重训练，最快验证）

```bash
# VisA — 仅 top-ratio
python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=12000 \
  testing.top_ratio=0.01

# MVTec — 仅 top-ratio
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=14000 \
  testing.top_ratio=0.01
```

### 仅 PCA 辅助打分（无需重训练）

```bash
# VisA — 仅 PCA
python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=12000 \
  testing.pca.enabled=true \
  testing.pca.score_weight=0.3

# MVTec — 仅 PCA
python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=/root/autodl-tmp/dataset/mvtec \
  app=test \
  app.ckpt_step=14000 \
  testing.pca.enabled=true \
  testing.pca.score_weight=0.3
```

### 消融参考：PCA 超参调节

```bash
# score_weight 消融: 0.1 / 0.3 / 0.5
testing.pca.score_weight=0.1
testing.pca.score_weight=0.3
testing.pca.score_weight=0.5

# ev_ratio 消融: 0.95 / 0.99 / 0.995
testing.pca.ev_ratio=0.95
testing.pca.ev_ratio=0.99
testing.pca.ev_ratio=0.995

# top_ratio 消融: 0.005 / 0.01 / 0.02
testing.top_ratio=0.005
testing.top_ratio=0.01
testing.top_ratio=0.02
```

### 一键过夜脚本

```bash
# MVTec 全量实验 (2×训练 + 16×评估, 约 6-8 小时)
bash run_mvtec_resAD.sh
```

日志输出在 `experiment/resAD_mvtec_{日期}/`，结束后自动打印汇总表。
