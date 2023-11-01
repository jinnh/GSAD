<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=global-structure-aware-diffusion-process-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lol-v2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2?p=global-structure-aware-diffusion-process-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-structure-aware-diffusion-process-for-1/low-light-image-enhancement-on-lol-v2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2-1?p=global-structure-aware-diffusion-process-for-1)

<h2 align="center">[NeurIPS 2023] Global Structure-Aware Diffusion Process for Low-Light Image Enhancement</h2>

<p align="center">
    <a href="https://arxiv.org/abs/2310.17577">arXiv</a>
    ·
    <a href="https://arxiv.org/abs/2310.17577">NeurIPS23</a>
    ·
    <a href="https://github.com/jinnh/GSAD">Project Page</a>
  </p>

<a href="https://github.com/jinnh/GSAD">
    <img src="./images/framework.png" alt="Logo" width="480" height="300">
  </a>

</div>

<details close>
  <summary>Evaluation on LOLv1 and LOLv2</b></summary>

<div align="center">
<img src="./images/quantitative%20results.png" alt="Logo" width="700" height="300">
</div>

<img src="./images/visual%20results.png" alt="Logo" width="700" height="320">

</details>

## Get Started

### Dependencies and Installation

- Python 3.8
- Pytorch 1.10

1. Create Conda Environment

```
conda create --name GlobalDiff python=3.8
conda activate GlobalDiff
```

2. Install pytorch

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

### Data preparation

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

### Training

```
bash train.sh
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

Our code is partly built upon [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Thanks to the contributors of their great work.
