<div align="center">
<h1>FLDM-VTON: Faithful Latent Diffusion Model for Virtual Try-on</h1>

<a href='https://xiangji-ai.github.io/fldm-vton'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2404.14162'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://arxiv.org/abs/2404.14162'><img src='https://img.shields.io/badge/Code-Github-blue'></a> <a href='https://www.xiangji.ai'><img src='https://img.shields.io/badge/Web-xiangji.ai-purple'></a>

</div>

Official PyTorch implementation of the paper ["FLDM-VTON: Faithful Latent Diffusion Model for Virtual Try-on"](https://arxiv.org/abs/2404.14162) (IJCAI2024)

<p align="center"> <a href='https://ijcai24.org/'><img src=https://ijcai24.org/wp-content/uploads/2023/11/cropped-logo-1.png width=15% /></a> <a href='https://www.fudan.edu.cn/'><img src=https://pic.baike.soso.com/ugc/baikepic2/11345/20200825160633-1825645712_jpeg_1200_1198_285650.jpg/0 width=6% /></a>  <a href='https://www.xiangji.ai'><img src=https://static.xiangjifanyi.com/portal-new/assets/logo2-b014c15d.png width=17% /></a></p>


## TODO LIST

- [x] Demo images
- [ ] Inference code
- [ ] Training code


## Installation

```bash
git clone https://github.com/xiangji-ai/fldm-vton
cd fldm-vton

pipenv install -f requirements.txt
```

## Data preparation

### VITON-HD
You can download VITON-HD dataset from [VITON-HD](https://github.com/shadow2496/VITON-HD).

### DressCode
You can download DressCode dataset from [DressCode](https://github.com/aimagelab/dress-code).



## License

All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.


## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{wang2024fldm,
  title={FLDM-VTON: Faithful Latent Diffusion Model for Virtual Try-on},
  author={Wang, Chenhui and Chen, Tao and Chen, Zhihao and Huang, Zhizhong and Jiang, Taoran and Wang, Qi and Shan, Hongming},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2024}
}
```