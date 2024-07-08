# RainyScape
Code for 'RainyScape: Unsupervised Rainy Scene Reconstruction using Decoupled Neural Rendering'


# Get the RainyScape_dataset

Please download the dataset via the following Onedirve link.

https://portland-my.sharepoint.com/:u:/g/personal/xianqialv2-c_my_cityu_edu_hk/EWLAW9WZ99lPq1j7OPA-1kEBwvq7k4MMwWpgXkTMh5pmnA?e=NnpnvZ

# Training and Testing the RainyScape

For NeRF rendering,

CUDA_VISIBLE_DEVICES=0 python torch_diffnerf_derain_rgbfeature_v0308.py --config configs/scene01_deraining_unsup_ver1.txt

For 3DGS rendering, you need to configure the basic environment of 3DGS first. The configuration process refers to the official 3DGS project page.

CUDA_VISIBLE_DEVICES=0 python torch_diffnerf_derain_rgbfeature_v0308.py --config configs/scene01_deraining_unsup_ver1.txt

# Acknowledgments

This code is based on the following implementation:

\item
The pytorch implementation of NeRF: https://github.com/yenchenlin/nerf-pytorch

\item
The 3DGS official code: https://github.com/graphdeco-inria/gaussian-splatting
