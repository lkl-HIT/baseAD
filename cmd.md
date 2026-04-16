# FoundAD 实验模式 (VisA 4-shot)

```bash
conda activate foundad

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
