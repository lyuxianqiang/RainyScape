import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from skimage import metrics
from torchvision import transforms

convert_to_gray = transforms.Grayscale(num_output_channels=1)

def gradient(input_tensor):
    input_tensor = input_tensor.unsqueeze(0)
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(input_tensor.device).type_as(input_tensor)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(input_tensor.device).type_as(input_tensor)
    sobel_x = sobel_x.repeat(input_tensor.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(input_tensor.size(1), 1, 1, 1)
    grad_x = F.conv2d(input_tensor, sobel_x, padding=1, groups=input_tensor.size(1))
    grad_y = F.conv2d(input_tensor, sobel_y, padding=1, groups=input_tensor.size(1))
    return grad_x, grad_y


def gradinet_rotate_loss_semi2(pred, target, rain_gen, topn = 1):
    '''
    Adaptive Gradient Rotation Loss.
    : pred: The predicated clean image
    : target: The input rainy image 
    : rain_gen: The predicated rain
    '''
    C, W, H = pred.shape
    minus_rain = target - rain_gen
    minus_rain = convert_to_gray(minus_rain)
    residual = target - pred
    residual = convert_to_gray(residual)
    pred = convert_to_gray(pred)
    rain_gen = convert_to_gray(rain_gen)

    pred_grad_x, pred_grad_y = gradient(pred)
    res_grad_x, res_grad_y = gradient(residual)
    rain_grad_x, rain_grad_y = gradient(rain_gen)
    minus_grad_x, minus_grad_y = gradient(minus_rain)

    gradient_orientation = torch.atan2(res_grad_y, res_grad_x)
    gradient_orientation = torch.remainder(gradient_orientation, torch.pi)  # 保持在[0, π)范围内

    # Calculate the gradient magnitude and filter out gradients smaller than the threshold
    magnitude = torch.sqrt(res_grad_x**2 + res_grad_y**2 + 1e-8)
    mask = magnitude > 0.1
    filtered_orientation = gradient_orientation[mask]

    # Statistical histogram, radian step size π/60, find the top k in the histogram
    histogram = torch.histc(filtered_orientation, bins=60, min=0, max=torch.pi)
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


def get_loss_MStep_all(args, Y, back_pre, rain_gen, gt, epoch):  # back_pre p*p*3
    '''
    : Consider the direction of rain drop! Unsupervised.
    :param Y: B x p x p x 3 tensor, input rainy images
    :param back_pre: B x p x p x 3 tensor, derained images
    :rain_gen: B x p x p x 3 tensor, generated rain
    :param gt: B x p x p x 3 tensor, groundtruth images, if any.
    '''
    sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
    likelihood = 0.1 / (sigma**2 + 1e-6) * (Y - back_pre - rain_gen).square().mean()

    mse_scale = args.epsilon2 * (Y - back_pre - rain_gen).square().mean()
    tv_loss = args.rho * (tv1_norm3d(back_pre, [1.0,1.0]))
    gd_loss = args.wgd * gradinet_rotate_loss_semi2(back_pre.permute(2,0,1),Y.permute(2,0,1),rain_gen.permute(2,0,1)) 
    loss = likelihood + mse_scale + tv_loss + gd_loss
    return loss, likelihood, mse_scale, tv_loss, gd_loss
