# FoundAD <img src="./assets/icon.png" alt="FoundAD logo showing a stylized magnifying glass over abstract shapes, representing anomaly detection in visual data. The logo is set against a neutral background and does not contain any text. The tone is professional and focused." width="30" height="30">

The implementation of the paper **Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors** ([arXiv](http://arxiv.org/abs/2510.01934
), [OpenReview](https://openreview.net/forum?id=YRrlJ8oVEH)).

  <a href="https://ymxlzgy.com/">Guangyao Zhai</a>, <a href="https://karolinezhy.github.io/">Yue Zhou</a>, <a href="">Xinyan Deng</a>, <a href="https://scholar.google.com/citations?user=f5DnPiEAAAAJ&hl=de">Lars Heckler</a>, <a href="https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/">Nassir Navab</a>, and <a href="https://www.cs.cit.tum.de/camp/members/benjamin-busam/">Benjamin Busam</a>
<br>
  Technical University of Munich <span style="margin: 0 10px;">•</span> MVTec Software GmbH

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Training and Inference](#train-infer)
   - [Dataset Preparation](#dataset-preparation)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
4. [Acknowledgement](#acknowledgement)
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n foundad python=3.10
conda activate foundad
git clone git@github.com:ymxlzgy/FoundAD.git
cd FoundAD
pip install -r requirements.txt
pip install -e .
```


## Quick Start
Before we start, please make sure you have the rights to use [DINOv3](https://github.com/facebookresearch/dinov3). Download our trained manifold projectors, and put them to `./logs/`. 
|DINOv3-based|1-shot|2-shot|4-shot|
|---------|:---------:|:---------:|:---------:|
|**MVTec AD**|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_4shot.zip)|
|**VisA**  |[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_4shot.zip)|


Run a demo on MVTec-AD 
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=mvtec data.data_name=mvtec_1shot data.test_root=assets/mvtec
```

Or a demo on VisA
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=visa data.data_name=visa_4shot data.test_root=assets/visa
```


## Training and Inference

### Dataset Preparation

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: [<u>Here</u>](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | We use the structured dataset of [<u>RealNet</u>](https://github.com/cnulab/RealNet). |

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/media/ymxlzgy/Data21/xinyan/visa target=/media/ymxlzgy/Data21/xinyan/visa_tmp seed=42 num_samples=2
```
where `source` is the dataset folder, `target` is the folder of few-shot samples, and `num_samples` is the number of samples training models, e.g., 2 for 2-shot learning. `seed` can be adjusted to have multiple rounds of experiment.

### Model Training

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/media/ymxlzgy/Data21/xinyan app=train_dinov3 diy_name=dbug
```
where `data.dataset` is "mvtec" or "visa", `data.data_name` is the folder name of few-shot samples, `data.data_path` is the path where the few-shot folder is at, `app` is "train_dinov3" or other model configs under `configs/app/`, and `diy_name` (optionally) is the post-fix name of the model saving directory. To adjust the layer, please specify `app.meta.n_layer`.

**Path resolution.** Hydra sets `train_root` to `data.data_path/data.data_name`. Training code loads images from **`train_root/train/`** using `ImageFolder`, so each class must be a subdirectory of `train`, for example:

```text
${data.data_path}/${data.data_name}/train/<classname>/*.png
```

**Example (custom root, e.g. AutoDL).** If your few-shot folder lives under `/root/autodl-tmp/dataset` and is named `mvtec_4shot`, run from the repo root:

```bash
python foundad/src/sample.py source=/root/autodl-tmp/dataset/mvtec target=/root/autodl-tmp/dataset/fewshot/mvtec_4shot seed=42 num_samples=4
```

```bash
python foundad/main.py mode=train \
  data.batch_size=8 \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  data.data_path=/root/autodl-tmp/dataset/fewshot \
  app=train_dinov3 \
  diy_name=mvtec_4_test
```
```bash

python foundad/main.py mode=AD \
  data.dataset=mvtec \
  data.data_name=mvtec_4shot \
  diy_name=mvtec_4_test \
  data.test_root=<你的MVTec完整数据集路径> \
  app=test \
  app.ckpt_step=<你保存的步数>


python foundad/main.py mode=AD \
  data.dataset=visa \
  data.data_name=visa_4shot \
  diy_name=visa_4_test \
  data.test_root=/root/autodl-tmp/dataset/visa/visa_pytorch \
  app=test \
  app.ckpt_step=12000


```

The name `data.data_name` must contain the dataset tag (`mvtec` or `visa`) because the code checks `dataset in data_name`. Checkpoints and logs go under `logs/<data_name>/dinov3<diy_name>/` (see `foundad/configs/app/train_dinov3.yaml`).

### Anomaly Detection / Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot diy_name=dbug data.test_root=/media/ymxlzgy/Data21/xinyan/mvtec app=test app.ckpt_step=1950
```
where `data.test_root` is the dataset folder, and `app` is test_dinov2 or test_dinov3 under `configs/app/`. To adjust sample number K, please specify `testing.K_top_mvtec` and `testing.K_top_visa`.

## Acknowledgement
This repo utilizes [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2](https://github.com/facebookresearch/dinov2), [DINO](https://github.com/facebookresearch/dino), [SigLIP](https://github.com/google-research/big_vision), [CLIP](https://github.com/openai/CLIP) and [DINOSigLIP](https://github.com/tri-ml/prismatic-vlms). We also thank [I-JEPA](https://github.com/facebookresearch/ijepa) for the inspiration.
