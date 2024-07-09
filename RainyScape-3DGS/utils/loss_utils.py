#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import math
import os
# from skimage import metrics
from torchvision import transforms

convert_to_gray = transforms.Grayscale(num_output_channels=1)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gradient(input_tensor):
    input_tensor = input_tensor.unsqueeze(0)
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(input_tensor.device).type_as(input_tensor)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(input_tensor.device).type_as(input_tensor)
    sobel_x = sobel_x.repeat(input_tensor.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(input_tensor.size(1), 1, 1, 1)
    grad_x = F.conv2d(input_tensor, sobel_x, padding=1, groups=input_tensor.size(1))
    grad_y = F.conv2d(input_tensor, sobel_y, padding=1, groups=input_tensor.size(1))
    return grad_x, grad_y

def gradinet_rotate_loss_semi2(pred, target, rain_gen, binnum = 60, topn = 1):
    '''
    : Adaptive Gradient Rotation Loss
    '''
    C, W, H = pred.shape
    minus_rain = target - rain_gen
    minus_rain = convert_to_gray(minus_rain)
    residual = target - pred
    # target = convert_to_gray(target)
    residual = convert_to_gray(residual)
    pred = convert_to_gray(pred)
    rain_gen = convert_to_gray(rain_gen)

    pred_grad_x, pred_grad_y = gradient(pred)
    # target_grad_x, target_grad_y = gradient(target)
    res_grad_x, res_grad_y = gradient(residual)
    rain_grad_x, rain_grad_y = gradient(rain_gen)
    minus_grad_x, minus_grad_y = gradient(minus_rain)

    gradient_orientation = torch.atan2(res_grad_y, res_grad_x)
    gradient_orientation = torch.remainder(gradient_orientation, torch.pi)  # 保持在[0, π)范围内

    magnitude = torch.sqrt(res_grad_x**2 + res_grad_y**2 + 1e-8)
    mask = magnitude > 0.1
    filtered_orientation = gradient_orientation[mask]

    histogram = torch.histc(filtered_orientation, bins=binnum, min=0, max=torch.pi)
    values, indices = torch.topk(histogram, topn)

    rain_theta_loss = 0
    derain_theta_loss = 0
    for theta_radians in indices:
        rain_grad_theta_y = rain_grad_x * torch.cos(theta_radians + 0.5*torch.pi) + rain_grad_y * torch.sin(theta_radians + 0.5*torch.pi)   # 沿着角度 优化越小越好
        rain_grad_theta_x = rain_grad_x * torch.cos(theta_radians) + rain_grad_y * torch.sin(theta_radians)  # 垂直于角度  
        rain_theta_loss += rain_grad_theta_y.abs().sum()
        rain_theta_loss -= rain_grad_theta_x.abs().sum()
        pred_grad_theta = pred_grad_x * torch.cos(theta_radians) + pred_grad_y * torch.sin(theta_radians)
        minus_grad_theta = minus_grad_x * torch.cos(theta_radians) + minus_grad_y * torch.sin(theta_radians)
        derain_theta_loss += pred_grad_theta.abs().sum()
        derain_theta_loss += minus_grad_theta.abs().sum() 

    combined_loss = rain_theta_loss / (topn*W*H) + 0.5 * derain_theta_loss /(topn*W*H)
    return combined_loss

def tv1_norm3d(x, weight):
    '''
    Tv norm.
    :param x: p*p*3
    :param weight: list with length 3
    '''
    W, H, C = x.shape[:3]
    x_tv = (x[1:, :, :] - x[:-1, :, :]).abs().sum() * weight[0]
    y_tv = (x[:, 1:, :] - x[:, :-1, :]).abs().sum() * weight[1]
    # z_tv = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().sum() * weight[2]
    tv_loss = (x_tv + y_tv) / (W*H)
    return tv_loss

def get_loss_rainyscape(args, Y, back_pre, rain_gen, epoch): 
    '''
    : consider the direction of rain drop!
    ：unsupervised
    '''
    sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
    likelihood = 0.1 / (sigma**2 + 1e-6) * (Y - back_pre - rain_gen).square().mean()

    mse_scale = args.epsilon2 * (Y - back_pre - rain_gen).square().mean()
    tv_loss = args.rho * (tv1_norm3d(back_pre, [1.0,1.0]))
    gd_loss = args.wgd * gradinet_rotate_loss_semi2(back_pre,Y,rain_gen) 
    loss = likelihood + mse_scale + tv_loss + gd_loss
    return loss

