<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=global-structure-aware-diffusion-process-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lolv2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2?p=global-structure-aware-diffusion-process-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lolv2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2-1?p=global-structure-aware-diffusion-process-for-1)

<h2 align="center">[NeurIPS 2023] Global Structure-Aware Diffusion Process for Low-Light Image Enhancement</h2>

<p align="center">
    <a href="https://arxiv.org/abs/2310.17577">ArXiv</a>
    ·
    <a href="https://nips.cc/virtual/2023/poster/71121">NeurIPS23</a>
    ·
    <a href="https://jinnh.github.io/GlobalDiff/">Project Page</a>
  </p>

<a href="https://arxiv.org/abs/2310.17577">
    <img src="https://github.com/jinnh/jinnh.github.io/blob/main/GlobalDiff/static/images/globaldiff_arxiv_page.png?raw=true" alt="Logo" width="140" height="142">
  </a>

<a href="https://nips.cc/virtual/2023/poster/71121">
    <img src="https://github.com/jinnh/jinnh.github.io/blob/main/GlobalDiff/static/images/globaldiff_nips_page.png?raw=true" alt="Logo" width="140" height="143">
  </a>

<a href="https://jinnh.github.io/GlobalDiff/">
    <img src="https://raw.githubusercontent.com/jinnh/jinnh.github.io/main/GlobalDiff/static/images/globaldiff_project_page.png" alt="Logo" width="150" height="150">
  </a>

</div>

<!-- <details close>
  <summary>Evaluation on LOLv1 and LOLv2</b></summary>

  <div align="center">
    <img src="./images/quantitative%20results.png" alt="Logo" width="700" height="300">
    <img src="./images/visual%20results.png" alt="Logo" width="700" height="320">
  </div>
</details> -->

## Get Started

### Dependencies and Installation

- Python 3.8
- Pytorch 1.11

1. Create Conda Environment

```
conda create --name GlobalDiff python=3.8
conda activate GlobalDiff
```

2. Install PyTorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Clone Repo

```
git clone https://github.com/jinnh/GSAD.git
```

4. Install Dependencies

```
cd GSAD
pip install -r requirements.txt
```

### Data Preparation

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

Then, put them in the following folder:

<details open> <summary>dataset (click to expand)</summary>

```
├── dataset
    ├── LOLv1
        ├── our485
            ├──low
            ├──high
	├── eval15
            ├──low
            ├──high
├── dataset
   ├── LOLv2
       ├── Real_captured
           ├── Train
	   ├── Test
       ├── Synthetic
           ├── Train
	   ├── Test
```

</details>

### Testing

Note: Following LLFlow and KinD, we have also adjusted the brightness of the output image produced by the network, based on the average value of Ground Truth (GT). ``It should be noted that this adjustment process does not influence the texture details generated; it is merely a straightforward method to regulate the overall illumination.`` Moreover, it can be easily adjusted according to user preferences in practical applications.

Visual results on LOLv1 and LOLv2 can be downloaded from [Google drive](https://drive.google.com/drive/folders/1UIBn5Wle8FySag5Fby6PBm3zcxmN3qmY?usp=sharing).

You can also refer to the following links to download the [pretrained model](https://drive.google.com/drive/folders/1KLPm2oOg2Fx4WlbnOXMjN2rbyzzG8Hd-?usp=sharing) and put it in the following folder:

```
├── checkpoints
    ├── lolv1_gen.pth
    ├── lolv2_real_gen.pth
    ├── lolv2_syn_gen.pth
```

```
# LOLv1
python test.py --dataset ./config/lolv1.yml --config ./config/lolv1_test.json

# LOLv2-real
python test.py --dataset ./config/lolv2_real.yml --config ./config/lolv2_real_test.json

#LOLv2-synthetic
python test.py --dataset ./config/lolv2_syn.yml --config ./config/lolv2_syn_test.json
```

### Testing on unpaired data

```
python test_unpaired.py  --config config/test_unpaired.json --input unpaired_image_folder
```

You can use any one of these three pre-trained models, and employ different sampling steps and noise levels to obtain visual-pleasing results by modifying these terms in the 'test_unpaired.json'.

```
"resume_state": "./checkpoints/lolv2_syn_gen.pth"

"val": {
    "schedule": "linear",
    "n_timestep": 10,
    "linear_start": 2e-3,
    "linear_end": 9e-1
}
```


### Training

```
bash train.sh
```
Note: Pre-trained uncertainty models are available in [Google Drive](https://drive.google.com/drive/folders/139LvNvVv0ATp3-mIcSipnl1YauDcaVyf).

### Training on the customized dataset

1. We provide the dataset and training configs for both LOLv1 and LOLv2 benchmarks in the 'config' folder. You can create your configs for your dataset. You can also write your dataloader for the customized dataset before going to the 'diffusion.feed_data()'.

```
./config/customized_dataset.yml # e.g., lolv1.yml
./config/customized_dataset_train.json # e.g., lolv1_train.json
```

2. Specify the following terms in 'customized_dataset.yml'.

```
datasets.train.root # the path of training data
datasets.val.root # the path of testing data
```

3. Modify the following config path in 'train.sh', then run 'train.sh'.

```
## train uncertainty model
python train.py -uncertainty --config ./config/llie_train_u.json --dataset ./config/customized_dataset.yml 

## train global structure-aware diffusion
python train.py --config ./config/customized_dataset_train.json --dataset ./config/customized_dataset.yml
```

## Citation

If you find our work useful for your research, please cite our paper

```
@article{hou23global,
  title={Global Structure-Aware Diffusion Process for Low-Light Image Enhancement},
  author={Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, and Hui Yuan},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgement

Our code is built upon [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Thanks to the contributors for their great work.
