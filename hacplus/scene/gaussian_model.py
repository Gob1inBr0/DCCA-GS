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

import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from hacplus.utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from hacplus.utils.graphics_utils import BasicPointCloud
from hacplus.utils.system_utils import mkdir_p
from hacplus.utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2

from hacplus.utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from hacplus.utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from hacplus.utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i

class Channel_CTX_fea(nn.Module):
    def __init__(self, feat_dim=50, channel_group=None):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.channel_group = (
            pick_channel_group(self.feat_dim)
            if channel_group is None
            else int(channel_group)
        )
        assert self.feat_dim % self.channel_group == 0
        self.n_groups = self.feat_dim // self.channel_group
        g = self.channel_group
        for i in range(self.n_groups):
            setattr(
                self,
                f"MLP_d{i}",
                nn.Sequential(
                    nn.Linear(self.feat_dim * 3 + g * i, g * 4),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(g * 4, g * 3),
                ),
            )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, feat_dim]
        g = self.channel_group
        parts = torch.split(fea_q, split_size_or_sections=[g] * self.n_groups, dim=-1)
        outs = []
        prev = []
        for i in range(self.n_groups):
            mlp = getattr(self, f"MLP_d{i}")
            out = mlp(torch.cat(prev + [mean_scale], dim=-1))
            mean_i, scale_i, prob_i = torch.chunk(out, chunks=3, dim=-1)
            outs.append((mean_i, scale_i, prob_i))
            prev.append(parts[i])
            if to_dec == i:
                return mean_i, scale_i, prob_i
        mean_adj = torch.cat([o[0] for o in outs], dim=-1)
        scale_adj = torch.cat([o[1] for o in outs], dim=-1)
        prob_adj = torch.cat([o[2] for o in outs], dim=-1)
        return mean_adj, scale_adj, prob_adj


def pick_channel_group(feat_dim):
    """Largest supported channel-group size that divides ``feat_dim``.

    The official code uses 10 channels per group for feat_dim=50; for other
    dims fall back to 8, then 5, 4, 2, 1.
    """
    for g in (10, 8, 5, 4, 2, 1):
        if feat_dim % g == 0:
            return g
    return 1

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=50,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 hierarchical_context: bool=False,
                 hierarchical_context_start_iter: int=12000,
                 content_aware_quant: bool=True,
                 content_aware_q_mode: str="formula",
                 complexity_scale: float=0.35,
                 content_aware_start_iter: int=20000,
                 content_aware_ramp_iters: int=10000,
                 mlp_complexity_hidden: int=None,
                 mlp_complexity_layers: int=1,
                 level_threshold_low: float=0.33,
                 level_threshold_high: float=0.66,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.feat_channel_group = pick_channel_group(feat_dim)
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.hierarchical_context = bool(hierarchical_context)
        self.hierarchical_context_start_iter = int(hierarchical_context_start_iter)
        self.content_aware_quant = bool(content_aware_quant)
        if content_aware_q_mode != "formula":
            raise ValueError("PHG v1 only supports content_aware_q_mode='formula'")
        self.content_aware_q_mode = str(content_aware_q_mode)
        self.complexity_scale = float(complexity_scale)
        self.content_aware_start_iter = int(content_aware_start_iter)
        self.content_aware_ramp_iters = max(int(content_aware_ramp_iters), 1)
        self.mlp_complexity_hidden = mlp_complexity_hidden
        self.mlp_complexity_layers = max(int(mlp_complexity_layers), 1)
        self.level_threshold_low = float(level_threshold_low)
        self.level_threshold_high = float(level_threshold_high)
        self.current_step = 0
        self.current_iter = 0

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        # I6 per-anchor sensitivity EMA state (training-only).
        self.sensitivity_feat = torch.empty(0)
        self.sensitivity_scaling = torch.empty(0)
        self.sensitivity_offsets = torch.empty(0)
        self.sensitivity_mean = torch.zeros(3, device="cuda")
        self.sensitivity_var = torch.ones(3, device="cuda")

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.base_grid_context_dim = self.encoding_xyz.output_dim
        if self.hierarchical_context:
            self.grid_context_dim = self.base_grid_context_dim * 2 + 3
        else:
            self.grid_context_dim = self.base_grid_context_dim
        self.mlp_grid = nn.Sequential(
            nn.Linear(self.grid_context_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+feat_dim+1+1+1),
        ).cuda()

        hidden = (
            feat_dim // 2
            if self.mlp_complexity_hidden is None
            else int(self.mlp_complexity_hidden)
        )
        layers = [nn.Linear(8, hidden), nn.ReLU(True)]
        for _ in range(self.mlp_complexity_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(True)]
        layers.append(nn.Linear(hidden, 3))
        self.mlp_complexity = nn.Sequential(*layers).cuda()

        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea(
                feat_dim=self.feat_dim, channel_group=self.feat_channel_group
            ).cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()
        self.mlp_complexity.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()
        self.mlp_complexity.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._mask,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask[:, :10, :]
        mask_sig = torch.sigmoid(self._mask[:, :10, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        mask = self.get_mask  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    # ------------------------------------------------------------------
    # PHG I1: decoder-recomputable scale-aware hierarchical context
    # ------------------------------------------------------------------

    def get_scene_center(self):
        return (self.x_bound_min + self.x_bound_max) * 0.5

    def get_scene_extent(self):
        return torch.clamp(self.x_bound_max - self.x_bound_min, min=1e-6)

    def is_hierarchical_context_active(self):
        return bool(self.hierarchical_context) and (
            self.current_iter >= self.hierarchical_context_start_iter
        )

    def compute_anchor_level_ids(self, anchor):
        """Spatial-distance-only level ids (decoder-recomputable from grid coords)."""
        center = self.get_scene_center()
        extent = self.get_scene_extent()
        diag = torch.norm(extent, dim=-1, keepdim=True).clamp(min=1e-6)
        spatial = torch.norm(anchor - center, dim=-1, keepdim=True) / diag
        level = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
        level[spatial[:, 0] >= self.level_threshold_low] = 1
        level[spatial[:, 0] >= self.level_threshold_high] = 2
        return level

    def compute_anchor_level_onehot(self, anchor):
        level = self.compute_anchor_level_ids(anchor)
        return torch.nn.functional.one_hot(level, num_classes=3).to(anchor.dtype)

    def calc_context_feat(self, x, anchor_indices=None, caller=""):
        """I1 context = concat(base, parent, level); no view context is stored.

        ``anchor_indices``/``caller`` are accepted for adapter compatibility but
        are not needed because every input is recomputable from grid coords.
        """
        del anchor_indices, caller
        base = self.calc_interp_feat(x)
        if not self.is_hierarchical_context_active():
            return base
        parent_stride = max(self.voxel_size * self.update_hierachy_factor, 1e-6)
        parent_anchor = torch.round(x / parent_stride) * parent_stride
        parent = self.calc_interp_feat(parent_anchor)
        level = self.compute_anchor_level_onehot(x).to(base.dtype)
        return torch.cat([base, parent, level], dim=-1)

    # ------------------------------------------------------------------
    # PHG I2: content-aware formula quantization (decoder-recomputable)
    # ------------------------------------------------------------------

    @property
    def get_complexity_mlp(self):
        return self.mlp_complexity

    def get_complexity_chunk_size(self):
        return 4096

    def is_content_aware_quant_active(self):
        return bool(self.content_aware_quant) and (
            self.current_step >= self.content_aware_start_iter
        )

    def _content_aware_ramp_progress(self):
        if self.current_step < self.content_aware_start_iter:
            return 0.0
        return float(
            min(
                max(
                    (self.current_step - self.content_aware_start_iter)
                    / float(self.content_aware_ramp_iters),
                    0.0,
                ),
                1.0,
            )
        )

    def _estimate_formula_local_density(self, anchor):
        n = anchor.shape[0]
        if n <= 1:
            return torch.full(
                (n, 1), 0.5, device=anchor.device, dtype=anchor.dtype
            )
        with torch.no_grad():
            if n <= 4096:
                dists = torch.cdist(anchor, anchor)
                dists.fill_diagonal_(float("inf"))
                nn_dist = dists.min(dim=1)[0].unsqueeze(-1)
            else:
                sample_idx = (
                    torch.linspace(0, n - 1, steps=4096, device=anchor.device)
                    .round()
                    .long()
                )
                sample_anchors = anchor[sample_idx]
                chunks = []
                step = self.get_complexity_chunk_size()
                for start in range(0, n, step):
                    end = min(start + step, n)
                    dists = torch.cdist(anchor[start:end], sample_anchors)
                    chunks.append(dists.min(dim=1)[0].unsqueeze(-1))
                nn_dist = torch.cat(chunks, dim=0)
            voxel = max(self.voxel_size, 1e-6)
            return torch.exp(-nn_dist / voxel)

    def build_formula_complexity_input(
        self, anchor, predicted_mean_scaling, predicted_mean_offsets, masks
    ):
        from hacplus.utils.codec_consistency import build_formula_complexity_features

        local_density = self._estimate_formula_local_density(anchor)
        return build_formula_complexity_features(
            local_density,
            predicted_mean_scaling,
            predicted_mean_offsets,
            masks,
            self.n_offsets,
        )

    def estimate_formula_content_complexity(
        self, anchor, predicted_mean_scaling, predicted_mean_offsets, masks
    ):
        formula_input = self.build_formula_complexity_input(
            anchor, predicted_mean_scaling, predicted_mean_offsets, masks
        )
        return self.get_complexity_mlp(formula_input), formula_input

    def _q_base_value(self, name):
        return {"feat": 1.0, "scaling": 0.001, "offsets": 0.2}[name]

    def _make_field_q_params(self, name, q_adj):
        adj = 1 + torch.tanh(q_adj)
        return float(self._q_base_value(name)) * adj, None

    def _apply_field_complexity(self, name, Q, QS, complexity_field):
        del QS  # formula path only; no inv_scale mode in PHG v1
        return Q * complexity_field.to(dtype=Q.dtype, device=Q.device), None

    def _codec_apply_content_aware_quant_params(
        self,
        tag,
        anchor,
        masks,
        Q_feat,
        Q_scaling,
        Q_offsets,
        QS_feat,
        QS_scaling,
        QS_offsets,
        mean_scaling,
        mean_offsets,
        exact_scaling=None,
        exact_offsets=None,
        photo_stats=None,
        q_mode=None,
        complexity_strength=None,
    ):
        del tag, exact_scaling, exact_offsets, photo_stats, q_mode
        if not self.content_aware_quant:
            return (
                Q_feat,
                Q_scaling,
                Q_offsets,
                QS_feat,
                QS_scaling,
                QS_offsets,
                None,
            )
        from hacplus.utils.codec_consistency import formula_complexity_multiplier

        complexity_adj, formula_input = self.estimate_formula_content_complexity(
            anchor, mean_scaling, mean_offsets, masks
        )
        strength = (
            self.complexity_scale * self._content_aware_ramp_progress()
            if complexity_strength is None
            else float(complexity_strength)
        )
        multiplier = formula_complexity_multiplier(complexity_adj, strength)
        Q_feat = Q_feat * multiplier[:, 0:1].reshape(
            -1, *([1] * (Q_feat.dim() - 1))
        )
        Q_scaling = Q_scaling * multiplier[:, 1:2].reshape(
            -1, *([1] * (Q_scaling.dim() - 1))
        )
        Q_offsets = Q_offsets * multiplier[:, 2:3].reshape(
            -1, *([1] * (Q_offsets.dim() - 1))
        )
        self._last_formula_complexity_input = formula_input.detach()
        return (
            Q_feat,
            Q_scaling,
            Q_offsets,
            QS_feat,
            QS_scaling,
            QS_offsets,
            complexity_adj,
        )

    def _field_quantize(self, name, x, Q, QS, center):
        del name, QS  # formula path only; q_step/STE_multistep in PHG v1
        return STE_multistep.apply(x, Q, center)

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets+1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
                {'params': self.mlp_complexity.parameters(), 'lr': training_args.mlp_complexity_lr_init, "name": "mlp_complexity"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
                {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
                {'params': self.mlp_complexity.parameters(), 'lr': training_args.mlp_complexity_lr_init, "name": "mlp_complexity"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, foreach=True)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_complexity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_complexity_lr_init,
            lr_final=training_args.mlp_complexity_lr_final,
            lr_delay_mult=training_args.mlp_complexity_lr_delay_mult,
            max_steps=training_args.mlp_complexity_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_complexity":
                lr = self.mlp_complexity_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.detach(), requires_grad=True)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0].detach(), extension_tensor), dim=0),
                    requires_grad=True,
                )
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0].detach(), extension_tensor), dim=0),
                    requires_grad=True,
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0].detach()[mask], requires_grad=True
                )
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    with torch.no_grad():
                        scales[:, 3:].clamp_(max=0.05)
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0].detach()[mask], requires_grad=True
                )
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    with torch.no_grad():
                        scales[:, 3:].clamp_(max=0.05)
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask
        n_target = int(valid_points_mask.shape[0])
        for name in ("sensitivity_feat", "sensitivity_scaling", "sensitivity_offsets"):
            tensor = getattr(self, name)
            if tensor.numel() == 0:
                continue
            if tensor.shape[0] < n_target:
                pad = torch.zeros(
                    (n_target - tensor.shape[0], 1), device=tensor.device
                )
                setattr(self, name, torch.cat([tensor, pad], dim=0))
            elif tensor.shape[0] > n_target:
                setattr(self, name, tensor[:n_target])

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        for name in ("sensitivity_feat", "sensitivity_scaling", "sensitivity_offsets"):
            tensor = getattr(self, name)
            if tensor.numel() > 0:
                setattr(self, name, tensor[valid_points_mask])


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            # Local memory fix: dedup against existing anchors with O(G+U)
            # memory instead of the O(U*G*3) broadcast comparison, which can
            # allocate >20GB once U and G both reach hundreds of thousands.
            occupied = set(map(tuple, grid_coords.cpu().tolist()))
            keep = torch.tensor(
                [
                    tuple(c.tolist()) not in occupied
                    for c in selected_grid_coords_unique.cpu()
                ],
                dtype=torch.bool,
                device="cuda",
            )
            # keep=True marks NEW cells (not in the occupied set); the official
            # path ends with remove_duplicates=True for NEW cells.
            remove_duplicates = keep
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                # Local memory fix: gather parent features by selected index
                # instead of materializing the full [N, K, feat] repeat.
                sel_idx = torch.nonzero(candidate_mask).squeeze(-1)
                parent_feat = self._anchor_feat[sel_idx // self.n_offsets]
                new_feat = scatter_max(
                    parent_feat,
                    inverse_indices.unsqueeze(1).expand(-1, parent_feat.size(1)),
                    dim=0,
                )[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets+1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                for name in ("sensitivity_feat", "sensitivity_scaling", "sensitivity_offsets"):
                    tensor = getattr(self, name)
                    extension = torch.zeros(
                        (new_opacities.shape[0], 1), device="cuda"
                    )
                    if tensor.numel() == 0:
                        setattr(self, name, extension)
                    else:
                        setattr(self, name, torch.cat([tensor, extension], dim=0))

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        for name in ("sensitivity_feat", "sensitivity_scaling", "sensitivity_offsets"):
            tensor = getattr(self, name)
            if tensor.numel() > 0:
                setattr(self, name, tensor[~prune_mask])

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1))
        probs = torch.stack([prob, prob_adj], dim=-1)
        probs = torch.softmax(probs, dim=-1)

        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        bit_feat = self.EG_mix_prob_2.forward(_feat,
                                            mean, mean_adj,
                                            scale, scale_adj,
                                            probs[..., 0], probs[..., 1],
                                            Q=Q_feat)

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0
        t_codec = 0

        t_total_0 = get_time()
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]  # N, 50
        _grid_offsets = self._offset[mask_anchor]  # N, 10, 3
        _scaling = self.get_scaling[mask_anchor]  # N, 6
        _mask = self.get_mask[mask_anchor]  # N, 10, 1

        N = _anchor.shape[0]

        t_anchor_0 = get_time()
        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path= os.path.join(pre_path_name, 'xyz_gpcc.npz')
        means_strings = compress_gpcc(_anchor_int)
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
        bits_xyz = os.path.getsize(npz_path) * 8
        t_anchor += get_time() - t_anchor_0

        _anchor = _anchor_int * self.voxel_size
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask = _mask[sorted_indices]

        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_slice = _anchor[N_start:N_end]

            # encode feat
            feat_context = self.calc_interp_feat(anchor_slice)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end]
            feat = STE_multistep.apply(feat, Q_feat, self._anchor_feat.mean())
            torch.cuda.synchronize(); t0 = time.time()

            t_feature_0 = get_time()
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = torch.clamp(scale, min=1e-9)
            bit_feat = 0
            for cc in range(5):
                mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat, mean_scale, to_dec=cc)
                probs = torch.stack([prob[:, cc*10:cc*10+10], prob_adj], dim=-1)
                probs = torch.softmax(probs, dim=-1)

                feat_tmp = feat[:, cc*10:cc*10+10].contiguous().view(-1)
                Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)

                bit_feat += encoder_gaussian_mixed_chunk(
                    feat_tmp,
                    [mean[:, cc*10:cc*10+10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                    [scale[:, cc*10:cc*10+10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_feat_tmp,
                    file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)
            t_feature += get_time() - t_feature_0

            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)

            t_scaling_0 = get_time()
            scaling = _scaling[N_start:N_end].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            mask = _mask[N_start:N_end]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name, chunk_size=10_0000)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            t_offset += get_time() - t_offset_0

            torch.cuda.empty_cache()

        bit_anchor = bits_xyz
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        t_hash_0 = get_time()
        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bit_hash = encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel()*32
        t_hash += get_time() - t_hash_0

        t_mask_0 = get_time()
        bit_masks = encoder(_mask, file_name=masks_b_name)
        t_mask += get_time() - t_mask_0

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        # 32*3*2/bit2MB_scale is for xyz_bound_min and xyz_bound_max
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale + 32*3*2/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        log_info_time = f"\nEncoded time in s: " \
                   f"anchor {round(t_anchor, 4)}, " \
                   f"feat {round(t_feature, 4)}, " \
                   f"scaling {round(t_scaling, 4)}, " \
                   f"offsets {round(t_offset, 4)}, " \
                   f"hash {round(t_hash, 4)}, " \
                   f"masks {round(t_mask, 4)}, " \
                   f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time
        return log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feature = 0
        t_scaling = 0
        t_offset = 0
        t_hash = 0
        t_mask = 0

        t_total_0 = get_time()

        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')

        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))

        xyz_decoded_list = []
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        t_anchor_0 = get_time()
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        data_dict = np.load(npz_path)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * voxel_size
        t_anchor += get_time() - t_anchor_0
        N = anchor_decoded.shape[0]

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        t_mask_0 = get_time()
        masks_decoded = decoder(N*self.n_offsets, masks_b_name)  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)
        t_mask += get_time() - t_mask_0

        t_hash_0 = get_time()
        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder(N_hash, hash_b_name)  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)
        t_hash += get_time() - t_hash_0

        for s in range(steps):

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            anchor_sort = anchor_decoded[N_start:N_end]
            feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)

            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            t_feature_0 = get_time()
            feat_decoded = torch.zeros(size=[N_num, self.feat_dim], device='cuda', dtype=torch.float32)
            mean_scale = torch.cat([mean, scale, prob], dim=-1)
            scale = torch.clamp(scale, min=1e-9)
            for cc in range(5):
                mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(feat_decoded, mean_scale, to_dec=cc)
                probs = torch.stack([prob[:, cc*10:cc*10+10], prob_adj], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                Q_feat_tmp = Q_feat[:, cc*10:cc*10+10].contiguous().view(-1)

                feat_decoded_tmp = decoder_gaussian_mixed_chunk(
                    [mean[:, cc*10:cc*10+10].contiguous().view(-1), mean_adj.contiguous().view(-1)],
                    [scale[:, cc*10:cc*10+10].contiguous().view(-1), scale_adj.contiguous().view(-1)],
                    [probs[..., 0].contiguous().view(-1), probs[..., 1].contiguous().view(-1)],
                    Q_feat_tmp,
                    file_name=feat_b_name.replace('.b', f'_{cc}.b'), chunk_size=50_0000)

                feat_decoded_tmp = feat_decoded_tmp.view(N_num, 10)
                feat_decoded[:, cc*10:cc*10+10] = feat_decoded_tmp
            feat_decoded_list.append(feat_decoded)
            t_feature += get_time() - t_feature_0

            t_scaling_0 = get_time()
            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)
            t_scaling += get_time() - t_scaling_0

            t_offset_0 = get_time()
            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, chunk_size=10_0000)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)
            t_offset += get_time() - t_offset_0

            xyz_decoded_list.append(anchor_sort)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N, 6], device='cuda')
        _mask = torch.zeros(size=[N, self.n_offsets+1, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N, :10] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        self._anchor_feat = nn.Parameter(_anchor_feat)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        log_info_time = f"\nDecoded time in s: " \
                        f"anchor {round(t_anchor, 4)}, " \
                        f"feat {round(t_feature, 4)}, " \
                        f"scaling {round(t_scaling, 4)}, " \
                        f"offsets {round(t_offset, 4)}, " \
                        f"hash {round(t_hash, 4)}, " \
                        f"masks {round(t_mask, 4)}, " \
                        f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time

        return log_info
