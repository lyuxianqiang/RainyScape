# RainyScape
Code for 'RainyScape: Unsupervised Rainy Scene Reconstruction using Decoupled Neural Rendering'


# Get the RainyScape_dataset

Please download the dataset via the following Onedirve link.

https://portland-my.sharepoint.com/:u:/g/personal/xianqialv2-c_my_cityu_edu_hk/EWLAW9WZ99lPq1j7OPA-1kEBwvq7k4MMwWpgXkTMh5pmnA?e=NnpnvZ

After get the dataset, put the dataset into ```data``` folder.

# Training and Testing RainyScape

## NeRF Rendering

### Setting

Install the environment.
```
cd RainyScape-NeRF
pip install -r requirements.txt
```

Changing the data path and log path in the ```configs/Yard_deraining_unsup.txt```

### Training
To train and test the RainyScape-NeRF model, use the following command:

```
CUDA_VISIBLE_DEVICES=0 python torch_nerf_rainyscape.py --config configs/Yard_deraining_unsup.txt
```

## 3DGS Rendering

### Setting

Install the environment.
Before running the 3DGS rendering, make sure to configure the basic environment for 3DGS. Refer to the official [3DGS project page](https://github.com/graphdeco-inria/gaussian-splatting) for the setup instructions.

Convert the dataset by COLMAP for 3DGS input.

### Training

Once the environment is configured, you can run the rendering with:

```
CUDA_VISIBLE_DEVICES=0 python torch_3dgs_rainyscape.py -s data/Yard/
```

# Acknowledgments

If you use the code or the dataset in you own paper, please cite

   ```sh
@misc{lyu2024rainyscapeunsupervisedrainyscene,
      title={RainyScape: Unsupervised Rainy Scene Reconstruction using Decoupled Neural Rendering}, 
      author={Xianqiang Lyu and Hui Liu and Junhui Hou},
      year={2024},
      eprint={2404.11401},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.11401}, 
}
   ```

This project is based on the following implementations:

- The pytorch implementation of NeRF: https://github.com/yenchenlin/nerf-pytorch

- The 3DGS official code: https://github.com/graphdeco-inria/gaussian-splatting

