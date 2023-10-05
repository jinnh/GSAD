# [NeurIPS 2023] Global Structure-Aware Diffusion Process for Low-Light Image Enhancement

## Overall

This paper studies a diffusion-based framework to address the image reconstruction conundrum. Although current diffusion models enjoy superiority in the image generation field, it's still lacking in image reconstruction. To further boost the reconstruction performance of the diffusion model, we proposed to regularize the inherent ODE-trajectory  Observing that a na\"ive implementation of the diffusion model alone is inadequate for effectively resolving this issue, we go deep into the diffusion process and incorporate global structure-aware regularization, which leverages the intrinsic non-local structural constituents of image data, gradually facilitating the preservation of intricate details and the augmentation of contrast during the diffusion process. This incorporation mitigates the adverse effects of noise and artifacts stemming from the diffusion process, culminating in a more precise and resilient enhancement. To additionally promote learning in challenging regions, we introduce an uncertainty-guided regularization technique, which judiciously relaxes constraints on the most extreme portions of the image. Experimental evaluations reveal that the proposed diffusion-based framework, complemented by rank-informed regularization, attains exceptional performance in the realm of low-light enhancement. The outcomes indicate substantial advancements in image quality, noise suppression, and contrast amplification in comparison with SOTA techniques. This avant-garde approach catalyzes further exploration and advancement in low-light image processing, with potential ramifications for other applications of diffusion models.

![Framework](images/framework.png)

## Evaluation

![Quantitative results](images/quantitative%20results.png)

![Visual results](images/visual%20results.png)

## Citation

If you find our work useful for your research, please cite our paper

```
@article{zhou2023pyramid,
  title={Global Structure-Aware Diffusion Process for Low-Light Image Enhancement},
  author={Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, and Hui Yuan},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgement

Our code is partly built upon [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Thanks to the contributors of their great work.
