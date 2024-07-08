# RainyScape
Code for 'RainyScape: Unsupervised Rainy Scene Reconstruction using Decoupled Neural Rendering'


# Get the RainyScape_dataset
Please download the dataset via the following Onedirve link.
https://portland-my.sharepoint.com/:u:/g/personal/xianqialv2-c_my_cityu_edu_hk/EWLAW9WZ99lPq1j7OPA-1kEBwvq7k4MMwWpgXkTMh5pmnA?e=NnpnvZ

# Training and Testing the RainyScape
For NeRF rendering,
CUDA_VISIBLE_DEVICES=3 python torch_diffnerf_derain_rgbfeature_v0308.py --config configs/scene01_deraining_unsup_ver1.txt

For 3DGS rendering,

