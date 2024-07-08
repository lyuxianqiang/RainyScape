import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.io import loadmat, savemat
import fnmatch

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from NeRF import *
from run_nerf_helpers import *
from generators import *
from metrics import *

from load_maya import load_maya_data
from load_llff import load_llff_data
from load_blender import load_blender_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)
DEBUG = False



def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_derain", type=float, default=1e-6, 
                        help='learning rate after rain generator')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=20000,
                        help='number of iteration')
    # Setting of EM algorithm
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--state_size", type=int, default=64)
    parser.add_argument("--motion_size", type=int, default=64)
    parser.add_argument("--generator_path", type=str, default=None, 
                    help='specific generator_path file to save S,Z,M')
    parser.add_argument("--max_iter_EM", type=int, default=2,
                        help='number of EM iteration')
    parser.add_argument("--langevin_steps", type=int, default=5,
                        help='number of EM iteration')
    parser.add_argument("--delta", type=float, default=0.03,
                        help='step size for E-step')
    parser.add_argument("--rho", type=float, default=0.5,
                    help='step size for E-step')
    parser.add_argument("--wgd", type=float, default=1,
                    help='weight of gd loss')
    parser.add_argument("--epsilon2", type=float, default=0.01,
                        help='step size for E-step, 1e-6 for 256, 0.1 for float')
    parser.add_argument("--pretrain_nerf", type=int, default=1000,
                        help='number of pretrain_nerf iteration, 50 * 20')

# region
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--feature_channel", type=int, default=64, 
                        help='input feature channel numbers, 128 recommended')
    parser.add_argument("--embedding_mode", type=str, default='PE', 
                        help='embedding_mode for nerf rendering, PE, PeRgb, PeRgbF, PeF')
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='maya', 
                        help='options: maya / llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--mayahold", type=int, default=0, 
                        help='will take every 1/N images as LLFF test set, paper uses 0')
# endregion


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print(args)

    # Load data
    K = None
    if args.dataset_type == 'maya':
        images, images_wo_rains, poses, bds, render_poses, i_test = load_maya_data(args.datadir, args.factor,recenter=True, bd_factor=.75,spherify=args.spherify)
        generator_path = args.generator_path
        if not os.path.exists(generator_path):
            os.makedirs(generator_path)
        latent_path = generator_path + 'latent.mat'
        state_path = generator_path + 'state.mat'
        Z = np.random.randn(1, images.shape[0], args.latent_size).astype(np.float32)
        savemat(latent_path, {'Z': Z})
        S = np.random.randn(1, args.state_size).astype(np.float32)
        savemat(state_path, {'S': S})

        ########################let render_poses be the training cameras ##############################
        render_poses = poses
        ########################let render_poses be the training cameras ##############################
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded maya', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.mayahold > 0:
            print('Auto LLFF holdout,', args.mayahold)
            i_test = np.arange(images.shape[0])[::args.mayahold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        ########################let i_train, i_test, i_val all be the same  ##############################
        i_train = np.arange(images.shape[0])
        i_test = np.arange(images.shape[0])
        i_val = np.arange(images.shape[0])
        # i_train = np.arange(images.shape[0])
        # i_test = np.arange(images.shape[0])
        # i_val = np.arange(images.shape[0])
        ########################let i_train, i_test, i_val all be the same  ##############################
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        ########################let render_poses be the training cameras ##############################
        render_poses = poses
        ########################let render_poses be the training cameras ##############################
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        ########################let i_train, i_test, i_val all be the same  ##############################
        i_train = np.arange(images.shape[0])
        i_test = np.arange(images.shape[0])
        i_val = np.arange(images.shape[0])
        ########################let i_train, i_test, i_val all be the same  ##############################
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    current_time = time.strftime('%m-%d_%H-%M-%S', time.localtime())
    writer = SummaryWriter(f'runs/experiment_{expname}_{current_time}')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)


    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    kernelnet = None
    # Create nerf model
    nerf = NeRFAll(args, kernelnet).cuda()
    optimizer_nerf = torch.optim.Adam(params=nerf.parameters(),lr=args.lrate, betas=(0.9, 0.999))
    GStateNet = GeneratorState(latent_size=args.latent_size,
                                state_size=args.state_size,
                                motion_size=args.motion_size,
                                num_feature=128).cuda()
    GRainNet = GeneratorRain(im_size=[512,512],
                                out_channels=3,
                                state_size=args.state_size,
                                num_feature=64).cuda()
    optimizer_G = torch.optim.Adam([{'params': GStateNet.parameters(),'lr': 1e-3,'weight_decay': 0},
                                {'params': GRainNet.parameters(),'lr': 1e-4,'weight_decay': 0}],
                                betas = (0.5, 0.999))


    
    start = 0
    # Load Checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        unsorted_ckpts = [f for f in os.listdir(os.path.join(basedir, expname)) if fnmatch.fnmatch(f, 'nerf*.tar')]
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(unsorted_ckpts)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        checkpoint_nerf = torch.load(ckpt_path)
        start = checkpoint_nerf['global_step']
        optimizer_nerf.load_state_dict(checkpoint_nerf['optimizer_nerf_state_dict'])
        smart_load_state_dict(nerf, checkpoint_nerf)
        # Split the path from the right at the first 'nerf' occurrence
        parts = ckpt_path.rsplit('nerf', 1)
        checkpoint_path_G = 'generator'.join(parts)
        print('Reloading from', checkpoint_path_G)
        checkpoint_generator = torch.load(checkpoint_path_G)
        GStateNet.load_state_dict(checkpoint_generator['GState'])
        GRainNet.load_state_dict(checkpoint_generator['GRain'])
        optimizer_G = torch.optim.Adam([{'params': GStateNet.parameters(),'lr': 1e-3,'weight_decay': 0},
                            {'params': GRainNet.parameters(),'lr': 1e-4,'weight_decay': 0}],
                            betas = (0.5, 0.999))

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc: 
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = 0.
    render_kwargs_test['raw_noise_std'] = 0.
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            ########################################################render images and disparity maps #################################################################
            nerf.eval()
            rgbs, depths = nerf(H, W, K, args.chunk, 
                                input_imgs=images[i_test],
                                poses=poses[i_test],
                                **render_kwargs_test)
            
            for idx, (rgb, depth) in enumerate(zip(rgbs, depths)):
                rgb8 = to8b(rgb.cpu().numpy())
                depth_np = depth.cpu().numpy()
                max_depth = np.max(depth_np)
                # Avoid division by zero
                if max_depth > 0:
                    depth8 = to8b(depth_np / max_depth)
                else:
                    print(f"Warning: depth image at index {idx} is all zero.")
                    depth8 = np.zeros_like(depth_np)  # or whatever you want to do in this case

                filename_rgb = os.path.join(testsavedir, f'rgb_{idx:03d}.png')
                filename_disp = os.path.join(testsavedir, f'depth_{idx:03d}.png')
                imageio.imwrite(filename_rgb, rgb8)
                imageio.imwrite(filename_disp, depth8)
            # rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, input_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            ########################################################render images and disparity maps #################################################################

            return

    # Prepare raybatch tensor if batching random rays
    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
        print('done, concats', rays.shape)
        rays_rgb = np.concatenate([rays, images[:,None]], 1)
        rays_rgb = np.concatenate([rays_rgb, images_wo_rains[:,None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1,4,3])
        rays_rgb = rays_rgb.astype(np.float32)

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    images_gt = torch.Tensor(images_wo_rains).to(device)
    poses = torch.Tensor(poses).to(device)

    if use_batching:
        print('shuffle rays')
        perm = np.random.permutation(rays_rgb.shape[0])
        rays_rgb_shuffle = rays_rgb[perm]
        print('done')

    i_batch = 0
    N_rand = args.N_rand
    N_iters = args.N_iters + 1
    print('Begin')

    start = start + 1
    epoch = 0 
    for i in trange(start, N_iters):
        input_M = poses[:,:,:4]
        input_M = input_M.reshape(input_M.size(0), -1)
        input_S = torch.from_numpy(S).cuda()
        input_Z = torch.from_numpy(Z).cuda()

        lossM_EM = likelihood_EM = mse_EM = tv_EM = gd_EM =  0
        mean_norm_grad_EM_nerf = 0

        if i < args.pretrain_nerf:
            for _ in range(10):
                optimizer_nerf.zero_grad()
                # # Sample random ray batch
                batch = torch.Tensor(rays_rgb_shuffle[i_batch:i_batch+N_rand]).to(device)
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:3], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    perm = np.random.permutation(rays_rgb.shape[0])
                    rays_rgb_shuffle = rays_rgb[perm]
                    i_batch = 0

                nerf.train()
                rgb, rgb0, extras = nerf(H, W, K, chunk=args.chunk, 
                                        rays=batch_rays, 
                                        rays_feature = None,
                                        retraw=False, **render_kwargs_train)
                
                img_loss = img2mse(rgb, target_s)
                img_loss0 = img2mse(rgb0, target_s)
                lossM = img_loss + img_loss0

                lossM.backward()
                mean_norm_grad_EM_nerf = nn.utils.clip_grad_norm_(nerf.parameters(), 1e4)
                optimizer_nerf.step()
                # accumulate loss of M-Step
                lossM_EM = lossM.item()

        else:
            #######  EM-algorithm  ########
            for _ in range(args.max_iter_EM):
                # M-Step　
                optimizer_G.zero_grad()
                optimizer_nerf.zero_grad()

                rain_gen_M, state_next = G_forward_truncate(GStateNet, GRainNet,
                                                            input_Z, input_S, input_M)
                rain_gen_M = rain_gen_M.squeeze().permute(1,2,3,0)

                #####  Core optimization loop  #####
                nerf.train()
                lossM_avg = 0
                writer.add_images('Grain_images', rain_gen_M[24:25,:,:,:].permute(0,3,1,2), i)
                for _ in range(2):
                    #####  Random from one image and patchs 64  #####
                    img_i = torch.randint(low=0, high=len(i_train), size=(1,)).item()
                    start_x = torch.randint(0, H - 64, size=(1,)).item()
                    start_y = torch.randint(0, W - 64, size=(1,)).item()
                    target_s = images_gt[img_i,start_x:start_x+64, start_y:start_y+64,:]
                    image_patch = images[img_i,start_x:start_x+64, start_y:start_y+64,:]
                    pose = poses[img_i, :3,:4]
                    rays_o, rays_d = get_rays(H, W, K, pose)
                    batch_rays = torch.stack([rays_o[start_x:start_x+64, start_y:start_y+64,:], rays_d[start_x:start_x+64, start_y:start_y+64,:],image_patch],dim=0)

                    rgb, rgb0, extras = nerf(H, W, K, chunk=args.chunk, 
                                            rays=batch_rays, 
                                            rays_feature = None,
                                            input_imgs = image_patch,
                                            retraw=False, **render_kwargs_train)
                    lossM, likelihood, mse_scale, tv, gd_loss = get_loss_MStep_all(args,
                                                                    image_patch,
                                                                    rgb,
                                                                    rain_gen_M[img_i,start_x:start_x+64, start_y:start_y+64,:],
                                                                    target_s,
                                                                    i)
                    lossM_avg += lossM / 2.0
                writer.add_scalar('Loss/lossM', lossM_avg, i)
                writer.add_scalar('Loss/likelihood', likelihood, i)
                writer.add_scalar('Loss/mse_scale', mse_scale, i)
                writer.add_scalar('Loss/tv', tv, i)
                writer.add_scalar('Loss/gd_loss', gd_loss, i)

                lossM_avg.backward()
                current_norm_grad_nerf = nn.utils.clip_grad_norm_(nerf.parameters(), 1e4)
                optimizer_nerf.step()
                optimizer_G.step()

                # accumulate loss of M-Step
                lossM_EM += lossM_avg.item()
                likelihood_EM += likelihood.item()
                mse_EM += mse_scale.item()
                tv_EM += tv.item()
                gd_EM += gd_loss.item()
                mean_norm_grad_EM_nerf += current_norm_grad_nerf

                # E-Step　
                freeze_Generator(GStateNet,GRainNet)
                rain_gt = image_patch - rgb.detach()
                for ss in range(args.langevin_steps):
                    input_S.requires_grad = True
                    input_Z.requires_grad = True
                    # input_M.requires_grad = True
                    rain_gen_E, state_next = G_forward_truncate(GStateNet,GRainNet,input_Z, input_S, input_M)
                    rain_gen_E = rain_gen_E.squeeze().permute(1,2,3,0)
                    lossE, likelihood, mse_scale, tv, gd_loss = get_loss_MStep_all(args,
                                                                    image_patch,
                                                                    rgb.detach(),
                                                                    rain_gen_E[img_i,start_x:start_x+64, start_y:start_y+64,:],
                                                                    target_s,
                                                                    i)
                    lossE.backward()

                    input_S = input_S - 0.5 * (args.delta**2) * (input_S.grad + input_S/50)
                    input_Z = input_Z - 0.5 * (args.delta**2) * (input_Z.grad + input_Z/50)
                    if ss < (args.langevin_steps/3):
                        input_S = input_S + args.delta * torch.randn_like(input_S)
                        input_Z = input_Z + args.delta * torch.randn_like(input_Z)
                    input_S.detach_()
                    input_Z.detach_()
                unfreeze_Generator(GStateNet,GRainNet)
            # update Z_rank and S_rank
            if (i+1) > args.pretrain_nerf:
                Z = input_Z.data.cpu().numpy()
                S = input_S.data.cpu().numpy()
                if (i+1) % 100 == 0:
                    savemat(latent_path, {'Z':Z})
                    savemat(state_path,  {'S':S})
        
        mse_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(mse_loss)


        # NOTE: IMPORTANT!
        ###   update learning rate of nerf   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        if i < args.pretrain_nerf:
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        else:
            new_lrate = args.lrate_derain
        for param_group in optimizer_nerf.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, 'nerf{:06d}.tar'.format(i))
            path_generator = os.path.join(basedir, expname, 'generator{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_nerf_state_dict': optimizer_nerf.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            torch.save({'GState': GStateNet.state_dict(),
                        'GRain': GRainNet.state_dict()}, path_generator)


        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)  
            if i >= args.pretrain_nerf:
                rain_save = to8b(rain_gen_E.detach().cpu().numpy())
                for idx_ge in range(50):
                    rain_save_img = rain_save[idx_ge]
                    filename_rain = os.path.join(testsavedir, f'rain_{idx_ge:03d}.png')
                    imageio.imwrite(filename_rain,rain_save_img)

            with torch.no_grad():
                nerf.eval()
                rgbs, depths = nerf(H, W, K, args.chunk, 
                                    input_imgs=images[i_test],
                                    poses=poses[i_test],
                                    **render_kwargs_test)
                
                for idx, (rgb, depth) in enumerate(zip(rgbs, depths)):
                    rgb8 = to8b(rgb.cpu().numpy())
                    depth_np = depth.cpu().numpy()
                    max_depth = np.max(depth_np)
                    rain_em8 = to8b(torch.abs(images[idx]-rgb).cpu().numpy())
                    # Avoid division by zero
                    if max_depth > 0:
                        depth8 = to8b(depth_np / max_depth)
                    else:
                        print(f"Warning: depth image at index {idx} is all zero.")
                        depth8 = np.zeros_like(depth_np)  # or whatever you want to do in this case

                    filename_rgb = os.path.join(testsavedir, f'rgb_{idx:03d}.png')
                    filename_disp = os.path.join(testsavedir, f'depth_{idx:03d}.png')
                    filename_emrain = os.path.join(testsavedir, f'emrain_{idx:03d}.png')
                    imageio.imwrite(filename_rgb, rgb8)
                    imageio.imwrite(filename_disp, depth8)
                    imageio.imwrite(filename_emrain, rain_em8)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} LossM: {lossM_EM} ({likelihood_EM}/{mse_EM}/{tv_EM},{gd_EM})  PSNR: {psnr.item()}")
            # print log
            lr_nerf = optimizer_nerf.param_groups[0]['lr']
            lr_GState = optimizer_G.param_groups[0]['lr']
            lr_GRain = optimizer_G.param_groups[1]['lr']
            log_str = '\nM-Step: Epoch:{:03d}/{:03d}, ' + \
                            'LossM:{:.2e}({:.2e}/{:.2e}/{:.2e}/{:.2e}), GradD:{:.2e}/{:.2e}, ' + \
                                                                'lrSRD:{:.2e}/{:.2e}/{:.2e}'
            print(log_str.format(i+1, args.N_iters,
                    lossM_EM, likelihood_EM, mse_EM, tv_EM, gd_EM, mean_norm_grad_EM_nerf,1e4,
                    lr_GState, lr_GRain, lr_nerf))

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
