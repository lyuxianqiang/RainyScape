import os
import torch
import datetime
from random import randint
import numpy as np
from scipy.io import loadmat, savemat
from pprint import pprint
from utils.loss_utils import *
from gaussian_renderer import render, network_gui
import sys
import torchvision
from scene import Scene, GaussianModel
from generator import GeneratorState, GeneratorRain,G_forward_perview,freeze_Generator,unfreeze_Generator
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, args, debug_from):
    print("Training configuration:")
    pprint(vars(dataset))
    pprint(vars(opt))
    pprint(vars(pipe))
    first_iter = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = os.path.basename(os.path.normpath(dataset.source_path))
    dataset.model_path = os.path.join("./output/", f"{data_name}_v1")
    args.model_path = dataset.model_path
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    ####
    GStateNet = GeneratorState(latent_size=128,
                                state_size=64,
                                motion_size=16,
                                num_feature=128).cuda()
    GRainNet = GeneratorRain(im_size=[512,512],
                                out_channels=3,
                                state_size=64,
                                num_feature=64).cuda()
    optimizer_G = torch.optim.Adam([{'params': GStateNet.parameters(),'lr': 1e-3,'weight_decay': 0},
                            {'params': GRainNet.parameters(),'lr': 1e-4,'weight_decay': 0}],
                            betas = (0.5, 0.999))

    latent_path = os.path.join(dataset.model_path , 'latent.mat')
    state_path  = os.path.join(dataset.model_path , 'state.mat')
    Z = np.random.randn(1, 50, 128).astype(np.float32)
    savemat(latent_path, {'Z': Z})
    S = np.random.randn(1, 64).astype(np.float32)
    savemat(state_path, {'S': S})


    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):  
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # print('GUI Running')   ## 
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        input_S = torch.from_numpy(S).cuda()
        input_Z = torch.from_numpy(Z).cuda()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() 

        if iteration < 4000:
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, 
                                scene, render, (pipe, background), GStateNet,GRainNet,input_Z,input_S)
                if (iteration in args.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in args.checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        else:
            #######  EM-algorithm  ########
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            idx = viewpoint_cam.colmap_id # 1...50
            input_M = viewpoint_cam.full_proj_transform # 4*4  3d世界坐标系到2d投影坐标系

            # M-Step　
            rain_gen_M = G_forward_perview(GStateNet, GRainNet,input_Z[:,idx-1,:], input_S, input_M) #B x 3 x p x p
            rain_gen_M = rain_gen_M.squeeze()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            ### 3
            image_all = image + rain_gen_M.detach()
            Ll1 = l1_loss(image_all, gt_image) + l1_loss(image, gt_image)
            lossM_gaussians = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_all, gt_image))

            # Loss for Generator 
            lossM_generator = get_loss_rainyscape(args, gt_image, image.detach(), rain_gen_M, iteration)

            # Generator part  
            lossM_generator.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # 3dgs rendering part
            lossM_gaussians.backward()
            for group in gaussians.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, lossM_generator, lossM_gaussians, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, 
                                scene, render, (pipe, background), GStateNet,GRainNet,input_Z,input_S)
                if (iteration in args.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in args.checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            torch.cuda.empty_cache()
                
            # E-Step　
            freeze_Generator(GStateNet,GRainNet)
            for ss in range(5):
                input_S.requires_grad = True
                input_Z.requires_grad = True
                rain_gen_E = G_forward_perview(GStateNet, GRainNet,input_Z[:,idx-1,:], input_S, input_M)
                rain_gen_E = rain_gen_E.squeeze()
                lossE = get_loss_rainyscape(args, gt_image, image.detach(), rain_gen_E, iteration)

                grad_S, grad_Z = torch.autograd.grad(lossE, [input_S, input_Z], retain_graph=False)
                input_S = input_S - 0.5 * (args.delta**2) * (grad_S + input_S/50)
                input_Z = input_Z - 0.5 * (args.delta**2) * (grad_Z + input_Z/50)


                if ss < (5/3):
                    input_S = input_S + args.delta * torch.randn_like(input_S)
                    input_Z = input_Z + args.delta * torch.randn_like(input_Z)
                input_S.detach_()
                input_Z.detach_()

            unfreeze_Generator(GStateNet,GRainNet)

            # update Z_rank and S_rank
            if (iteration+1) > 4000:
                Z = input_Z.data.cpu().numpy()
                S = input_S.data.cpu().numpy()
                if (iteration+1) % 100 == 0:
                    savemat(latent_path, {'Z':Z})
                    savemat(state_path,  {'S':S})
            torch.cuda.empty_cache()
        
        # print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        # print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")   


def prepare_output_and_logger(args):    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, GStateNet,GRainNet,camera_Z,scene_S):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})   # [5, 10, 15, 20, 25]

        ##
        # print(validation_configs)
        test_savepath = os.path.join(args.model_path,str(iteration))
        # print(test_savepath)
        if not os.path.exists(test_savepath):
            os.makedirs(test_savepath)

        for config in validation_configs:
            print(f"Name: {config['name']}, Number of cameras: {len(config['cameras'])}")
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    camera_id = viewpoint.colmap_id - 1
                    camera_M = viewpoint.full_proj_transform
                    rain_image = torch.clamp(G_forward_perview(GStateNet, GRainNet,camera_Z[:,camera_id,:], scene_S, camera_M),0.0,1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    torchvision.utils.save_image(image, os.path.join(test_savepath, viewpoint.image_name + ".png"))
                    torchvision.utils.save_image(rain_image, os.path.join(test_savepath, viewpoint.image_name + "_rain.png"))
                    # torchvision.utils.save_image(image, os.path.join(test_savepath, '{0:03d}'.format(idx) + ".png"))
                    # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8880)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000,10_000,20_000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000,10_000,20_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--rho", type=float, default=0.5,
                    help='step size for E-step')
    parser.add_argument("--wgd", type=float, default=1,
                    help='weight of gd loss')
    parser.add_argument("--epsilon2", type=float, default=500,
                        help='step size for E-step, 1e-6 for 256, 0.1 for float')
    parser.add_argument('--delta', type=float, default=0.03, help='delta for MCMC in E-step')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.model_path = os.path.basename(args.source_path)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, args.debug_from)

    # All done
    print("\nTraining complete.")
