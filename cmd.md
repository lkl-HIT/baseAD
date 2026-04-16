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
