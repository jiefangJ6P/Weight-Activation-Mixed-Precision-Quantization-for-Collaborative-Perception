# import torch.nn as nn
# from icecream import ic
# from opencood.models.sub_modules.pillar_vfe import PillarVFE
# from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
# from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
# from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
# from opencood.models.sub_modules.downsample_conv import DownsampleConv
# from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.fuse_modules.fusion_in_one import Where2comm
# from opencood.utils.transformation_utils import normalize_pairwise_tfm
#
# class PointPillarWhere2comm(nn.Module):
#     """
#     Where2comm implementation with point pillar backbone.
#     """
#     def __init__(self, args):
#         super(PointPillarWhere2comm, self).__init__()
#
#         self.pillar_vfe = PillarVFE(args['pillar_vfe'],
#                                     num_point_features=4,
#                                     voxel_size=args['voxel_size'],
#                                     point_cloud_range=args['lidar_range'])
#         self.scatter = PointPillarScatter(args['point_pillar_scatter'])
#         is_resnet = args['base_bev_backbone'].get("resnet", False)
#         if is_resnet:
#             self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
#         else:
#             self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
#         self.voxel_size = args['voxel_size']
#
#         self.fusion_net = Where2comm(args['where2comm'], dim=args['feat_dim'])
#         self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
#
#         self.shrink_flag = False
#         if 'shrink_header' in args:
#             self.shrink_flag = True
#             self.shrink_conv = DownsampleConv(args['shrink_header'])
#             self.out_channel = args['shrink_header']['dim'][-1]
#
#         self.compression = False
#         if "compression" in args:
#             self.compression = True
#             self.naive_compressor = NaiveCompressor(64, args['compression'])
#
#         self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
#                                   kernel_size=1)
#         self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
#                                   kernel_size=1)
#         self.use_dir = False
#         if 'dir_args' in args.keys():
#             self.use_dir = True
#             self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
#                                   kernel_size=1) # BIN_NUM = 2
#
#         if 'backbone_fix' in args.keys() and args['backbone_fix']:
#             self.backbone_fix()
#
#         self.comm_sum = 0.0
#         self.comm_n = 0
#
#     def backbone_fix(self):
#         """
#         Fix the parameters of backbone during finetune on timedelay。
#         """
#         for p in self.pillar_vfe.parameters():
#             p.requires_grad = False
#
#         for p in self.scatter.parameters():
#             p.requires_grad = False
#
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         if self.compression:
#             for p in self.naive_compressor.parameters():
#                 p.requires_grad = False
#         if self.shrink_flag:
#             for p in self.shrink_conv.parameters():
#                 p.requires_grad = False
#
#         for p in self.cls_head.parameters():
#             p.requires_grad = False
#         for p in self.reg_head.parameters():
#             p.requires_grad = False
#
#         if self.use_dir:
#             for p in self.dir_head.parameters():
#                 p.requires_grad = False
#         print("Backbone fixed.")
#
#     def forward(self, data_dict, fusion_flag=True):
#         voxel_features = data_dict['processed_lidar']['voxel_features']
#         voxel_coords = data_dict['processed_lidar']['voxel_coords']
#         voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
#         record_len = data_dict['record_len']
#
#         batch_dict = {'voxel_features': voxel_features,
#                       'voxel_coords': voxel_coords,
#                       'voxel_num_points': voxel_num_points,
#                       'record_len': record_len}
#         # n, 4 -> n, c
#         batch_dict = self.pillar_vfe(batch_dict)
#         # n, c -> N, C, H, W
#         batch_dict = self.scatter(batch_dict)
#
#         batch_dict_no_fusion = self.backbone(batch_dict)
#
#         # calculate pairwise affine transformation matrix
#         _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
#         normalized_affine_matrix = normalize_pairwise_tfm(
#             data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
#
#         spatial_features_2d = batch_dict_no_fusion['spatial_features_2d']
#
#         if self.shrink_flag:
#             spatial_features_2d = self.shrink_conv(spatial_features_2d)
#
#         if self.compression:
#             spatial_features_2d = self.naive_compressor(spatial_features_2d)
#
#         psm_single = self.cls_head(spatial_features_2d)
#
#         comm_sum = 0.0
#         comm_now = 0.0
#
#         _, CC, HH, WW = spatial_features_2d.shape
#         fused_feature, comm_rate = self.fusion_net(spatial_features_2d, psm_single,
#                                                     record_len, normalized_affine_matrix)
#         comm_now += comm_rate * CC * HH * WW
#         comm_sum += CC * HH * WW
#         self.comm_n += 1
#         self.comm_sum += comm_now / comm_sum
#
#         psm = self.cls_head(fused_feature)
#         rm = self.reg_head(fused_feature)
#
#         output_dict = {'cls_preds': psm,
#                        'reg_preds': rm}
#
#         if self.use_dir:
#             output_dict.update({'dir_preds': self.dir_head(fused_feature)})
#
#         # if self.comm_n == 2170 :
#         #     print(f"comm_rate = {self.comm_sum / self.comm_n}")
#
#         return output_dict

# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
import torch


class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()

        # PIllar VFE，将点云数据转换为体素特征，将点云数据转化为稀疏表示以供后续使用
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])  # 将体素特征转换为二维特征图，以便于后续的卷积操作
        if 'resnet' in args['base_bev_backbone']:  # 使用基础的bev backbone或者resnet backbone作为特征提取器
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:  # 是否使用shrink header，用于降低内存消耗
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:  # 是否使用压缩器，用于降低内存消耗
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False  # 是否使用deformable convolution，用于增强模型的感受野
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        # 分类头，用来预测目标的类别
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        # 回归头，用来预测目标的位置
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = True
        self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):  # 在某些特定的训练场景（如时间延迟微调）中冻结部分网络的参数，以避免对这些参数的更新。
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

# #原代码
#     def regroup(self, x, record_len):  # 用于根据记录的长度 (record_len) 将拼接的特征分组，主要是为了解决多车辆感知中不同车辆数据的长度不一致问题。
#         cum_sum_len = torch.cumsum(record_len, dim=0)
#         split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
#         return split_x
# #原代码
#修改后
    def regroup(self, x, record_len):
        """
        ONNX 导出：PyTorch 1.10 的 exporter 不支持 torch.tensor_split
        这里对 batch_size=1（部署最常见）走 fast-path，避免出现 tensor_split
        """
        # record_len: [B]
        B = record_len.shape[0]

        # 最常见部署：B=1，直接返回 list，避免 tensor_split
        if B == 1:
            return [x]

        # B>1：用 torch.split 替代 tensor_split（更容易被 ONNX 识别）
        # 注意：这会把 sections 变成 Python list，导出时它会被“固定”为当前 trace 的值
        sections = record_len.to(torch.long).tolist()
        return list(torch.split(x, sections, dim=0))
#修改后

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']  # 点云数据经过体素化处理后，每个体素的特征。
        voxel_coords = data_dict['processed_lidar']['voxel_coords']  # 每个体素的坐标。
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']  # 每个体素的特征。
        record_len = data_dict['record_len']  # 表示批次中每个车辆的点云数据长度。用于处理多个车辆的点云融合。

        pairwise_t_matrix = data_dict['pairwise_t_matrix']  # 用于表示不同车辆之间的相对位置的转换矩阵。

        # 将提取的点云特征、体素坐标、点数以及记录长度封装到一个 batch_dict 中，供后续操作使用。
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)  # 点云特征提取
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)  # 特征映射到BEV空间

        batch_dict = self.backbone(batch_dict)  # 主干网络提取高层次特征
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']  # 二维特征图

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)  # 分类结果
        rm_single = self.reg_head(spatial_features_2d)  # 回归结果

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:  # 多尺度融合，多车信息融合
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                                                              psm_single,
                                                                              record_len,
                                                                              pairwise_t_matrix,
                                                                              self.backbone,
                                                                              # [self.shrink_conv, self.cls_head, self.reg_head])
                                                                              )
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                                                              psm_single,
                                                                              record_len,
                                                                              pairwise_t_matrix)

        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)  # 融合后的分类结果
        rm = self.reg_head(fused_feature)  # 融合后的回归结果

        # 通过融合特征生成的分类和回归结果存储在 output_dict 中，并将 result_dict 更新到输出字典中。
        output_dict = {'psm': psm,
                       'rm': rm,
                       'cls_preds': psm,
                       'reg_preds': rm
                       }
        output_dict.update(result_dict)

        # regroup：根据 record_len 将批次中的多车辆特征分割开来，每个车辆的数据长度不一致，使用 torch.tensor_split 将它们分离。
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        # 将每个车辆的分类和回归结果分别提取出来，并根据虚拟车辆和实际车辆分组存储。
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        # 将提取的单个车辆的结果（psm_single_v, psm_single_i, rm_single_v, rm_single_i）以及通信率 communication_rates 更新到 output_dict 中，最终返回。
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                            'psm_single_i': psm_single_i,
                            'rm_single_v': rm_single_v,
                            'rm_single_i': rm_single_i,
                            'comm_rate': communication_rates
                            })
        output_dict.update({'dir_preds': self.dir_head(fused_feature)})
        return output_dict
