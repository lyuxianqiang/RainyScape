import torch
import torch.nn as nn
import numpy as np
from run_nerf_helpers import *
import os
import imageio
import time
from feature_encoder import FeatureExtractor

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)



class NeRFAll(nn.Module):
    def __init__(self, args, kernelsnet=None):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)
        print('Position encoding embeddding number 1: ',self.input_ch)
        if args.embedding_mode == 'PE':
            pass
        elif args.embedding_mode == 'PeRgb':
            self.input_ch = self.input_ch + 3
        elif args.embedding_mode == 'PeRgbF':
            self.input_ch = self.input_ch + 3 + args.feature_channel
        elif args.embedding_mode == 'PeF': 
            self.input_ch = self.input_ch + args.feature_channel
        else:
            print(f"Invalid embedding mode: {args.embedding_mode}")
        self.input_ch_views = 0
        self.kernelsnet = kernelsnet
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        self.mlp_coarse = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        self.mlp_fine = None
        if args.N_importance > 0:
            self.mlp_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        # self.rgb_activate = activate[args.rgb_activate]
        # self.sigma_activate = activate[args.sigma_activate]
        # self.tonemapping = ToneMapping(args.tone_mapping_type)

    def mlpforward(self, inputs, viewdirs, mlp_type, input_f=None, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            mlp_type, int, 1 for mlp_coarse, 2 for mlp_fine.
        """
        ###### input.shape = [N_rays, N_samples, 3] #######
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if input_f is not None:
            ############## add input feature information to embending ################
            input_f = input_f[:,None].expand(-1,inputs.shape[1],-1)
            # print('network input information:',input_f.shape) # torch.Size([1024, 64, 128])
            input_f_flat = torch.reshape(input_f, [-1, input_f.shape[-1]])  # [rays*samples,128]
            embedded = torch.cat([embedded, input_f_flat], -1)  

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if mlp_type == 1:
            outputs_flat = torch.cat([self.mlp_coarse(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)
        elif mlp_type == 2:
            outputs_flat = torch.cat([self.mlp_fine(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)
        else:
            print(f"Invalid mlp mode !!!")
        
        # if netchunk is None:
        #     outputs_flat = mlp(embedded)
        # else:
        #     outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def raw2outputs(self,raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        ####### ray_batch [b,3+3+2+3+3] rays_o,ray_d,bounds,viewdirs,rays_c
        # print('render rays each step:',ray_batch.shape) # torch.Size([1024, 78])

        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
        if self.args.embedding_mode == 'PE':
            rays_f = None
        elif self.args.embedding_mode == 'PeRgb' or 'PeRgbF' or 'PeF':
            rays_f = ray_batch[:,11:] 
        else:
            print(f"Invalid rays embedding mode in render_rays : {self.args.embedding_mode}")
        

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        ##### 1 for mlp_coarse, 2 for mlp_fine  ######
        raw = self.mlpforward(pts, viewdirs, 1, rays_f)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                             white_bkgd, pytest=pytest)

        if N_importance > 0:
            rgb_map_0, disp_map0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            # mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            # raw = self.mlpforward(pts, viewdirs, mlp, rays_f)
            raw = self.mlpforward(pts, viewdirs, 2, rays_f)

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'disp_map': disp_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            # ret['acc0'] = acc_map_0
            # ret['disp0'] = disp_map0
            # ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_feature = None, rays_info=None, input_imgs=None, poses=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the kernelsnet is specified

        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"
            rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, rays_feature, **kwargs)
            return rgb, extras['rgb0'], extras
                # return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), {}

        #  evaluation
        else:
            assert poses is not None, "Please specify poses when in the eval model"
            if "render_point" in kwargs.keys():
                rgbs, depths, weights = self.render_subpath(H, W, K, chunk, poses, **kwargs)
                depths = weights * 2
            else:
                # print(kwargs)
                rgbs, depths = self.render_path(H, W, K, chunk, poses, input_imgstest = input_imgs, **kwargs)
            return rgbs, depths

    def render(self, H, W, K, chunk, rays=None, rays_feature = None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              K: 3*3 intrinsic matrix of camera
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [3, batch_size, 3]. Ray origin, direction, and rgb for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d, rays_c = rays
        # rays_o, rays_d = rays


        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        rays_c = torch.reshape(rays_c, [-1, 3]).float()
        if rays_feature is not None:
            rays_f = rays_feature.float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        
        if self.args.embedding_mode == 'PE':
            pass
        elif self.args.embedding_mode == 'PeRgb':
            rays = torch.cat([rays, rays_c], -1) 
        elif self.args.embedding_mode == 'PeRgbF':
            # print(rays.shape,rays_f.shape)
            rays = torch.cat([rays, rays_c, rays_f], -1) 
        elif self.args.embedding_mode == 'PeF': 
            rays = torch.cat([rays, rays_f], -1) 
        else:
            print(f"Invalid rays embedding mode in Render: {self.args.embedding_mode}")     

        # Batchfy and Render and reshape
        # print('rays shape:', rays.shape)  # torch.Size([262144, 78]) 8+3+3+64
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            # print(torch.cuda.memory_allocated() / (1024 ** 3), 'GB')
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            # 输出 ret 中每个元素的 key 和 shape，而不输出具体的 tensor
            # for key, value in ret.items():
            #     print(f"Key: {key}, Shape: {value.shape}, Grad: {value.requires_grad}")
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, input_feature = None, render_factor=0, input_imgstest=None, **render_kwargs):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor
            #### differ from original !!!! ###
            K = K / render_factor

        rgbs = []
        depths = []

        ####  feature extractor using resnet 34  #####
        feat_extractor = FeatureExtractor().cuda()
        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays_o, rays_d = get_rays(H, W, K, c2w)
            # print('Test shape of rays_o', rays_o.shape)  # [H,W,3]
            batch_rays =  torch.stack([rays_o, rays_d, input_imgstest[i]], dim=0)   # [3,H,W,3]
            # print('Test shape of batch_rays', batch_rays.shape)

            feat_extractor_input = input_imgstest[i:i+1].permute(0,3,1,2)
            # print('Input shape of feature extractor', feat_extractor_input.shape)
            batch_feature = feat_extractor(feat_extractor_input)
            # print('Output shape of feature extractor', batch_feature.shape)   # [n,128,x,y]
            batch_feature = batch_feature.permute(0,2,3,1).reshape(-1,self.args.feature_channel)

            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=batch_rays,rays_feature=batch_feature, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_subpath(self, H, W, K, chunk, render_poses, render_point, images_indices, render_kwargs,
                       render_factor=0):
        """
        
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        weights = []

        t = time.time()

        rayx, rayy = torch.meshgrid(torch.linspace(0, W - 1, W),
                                    torch.linspace(0, H - 1, H))
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):

            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}

            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).type(torch.long) * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info)

            new_rays = new_rays[:, render_point]
            weight = weight[:, render_point]
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=new_rays.reshape(-1, 3, 2),
                                                  c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb.reshape(H, W, 3))
            depths.append(depth.reshape(H, W))
            weights.append(weight.reshape(H, W))
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        weights = torch.stack(weights, 0)

        return rgbs, depths, weights
