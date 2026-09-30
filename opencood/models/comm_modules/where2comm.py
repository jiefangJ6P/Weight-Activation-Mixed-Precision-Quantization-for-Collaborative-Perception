# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        self.smooth = False
        self.thre = args.get('thre', args.get('threshold', 0))
        print("where2comm thre =", self.thre)
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len_or_B, pairwise_t_matrix_or_warp_mask, req_mask=None):
        # Supports both call patterns used in this repo:
        # 1) forward(batch_confidence_maps, record_len, pairwise_t_matrix)
        # 2) forward(batch_confidence_maps, B, batch_warp_maks_list, req_mask)
        _, _, H, W = batch_confidence_maps[0].shape
        return_three_values = True
        if isinstance(record_len_or_B, int):
            B = record_len_or_B
            record_len = None
            batch_warp_maks_list = pairwise_t_matrix_or_warp_mask
            return_three_values = False
        else:
            record_len = record_len_or_B
            pairwise_t_matrix = pairwise_t_matrix_or_warp_mask
            B, _, _, _, _ = pairwise_t_matrix.shape
            batch_warp_maks_list = None

        if req_mask is not None:
            _, H_points_mask, W_points_mask = req_mask[0].shape
            if H_points_mask != H or W_points_mask != W:
                req_mask = F.interpolate(req_mask, size=(H, W), mode='nearest')

        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(
                1)  # dim1=2 represents the confidence of two anchors
            if batch_warp_maks_list is not None:
                ori_communication_maps = ori_communication_maps * batch_warp_maks_list[b]

            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where(communication_maps > self.thre, ones_mask, zeros_mask)

            if req_mask is not None:
                communication_mask = req_mask[b] * communication_mask

            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask_nodiag).to(communication_mask_nodiag.device)
            communication_mask_nodiag[0] = ones_mask[0]

            if communication_mask_nodiag.shape[0] > 1:
                communication_rate = communication_mask_nodiag[1:].sum() / ((communication_mask_nodiag.shape[0] - 1) * H * W)
            else:
                communication_rate = communication_mask_nodiag[0].sum() / (H * W)

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps * communication_mask_nodiag)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.concat(communication_masks, dim=0)
        if return_three_values:
            return batch_communication_maps, communication_masks, communication_rates
        return communication_masks, communication_rates
