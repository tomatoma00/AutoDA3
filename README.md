# AutoDA3
**Auto3R for Depth-Anything-3**

Official implementation of **Auto3R: Automated 3D Reconstruction and Scanning via Data-driven Uncertainty Quantification** based on **Depth-Anything-3**

[![arXiv](https://img.shields.io/badge/arXiv-2512.04528-b31b1b.svg)](https://arxiv.org/abs/2512.04528)


## Install & Requirements

CUDA>=11.8, recommand >=12.1

Below is the sample installation:

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install -e .

install the gsplat==1.5.3 (use git clone and pip install --no-build-isolation gsplat/ )
