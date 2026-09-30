import torch
import torch.nn as nn

##################原代码#####################
# class PointPillarScatter(nn.Module):
#     def __init__(self, model_cfg):
#         super().__init__()
#
#         self.model_cfg = model_cfg
#         self.num_bev_features = self.model_cfg['num_features']
#         self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1]
#
#         assert self.nz == 1
#
#     def forward(self, batch_dict):
#         """ 将生成的pillar按照坐标索引还原到原空间中
#         Args:
#             pillar_features:(M, 64)
#             coords:(M, 4) 第一维是batch_index
#
#         Returns:
#             batch_spatial_features:(4, 64, H, W)
#
#             |-------|
#             |       |             |-------------|
#             |       |     ->      |  *          |
#             |       |             |             |
#             | *     |             |-------------|
#             |-------|
#
#             Lidar Point Cloud        Feature Map
#             x-axis up                Along with W
#             y-axis right             Along with H
#
#             Something like clockwise rotation of 90 degree.
#
#         """
#         pillar_features, coords = batch_dict['pillar_features'], batch_dict[
#             'voxel_coords']
#         batch_spatial_features = []
#         batch_size = coords[:, 0].max().int().item() + 1
#
#         for batch_idx in range(batch_size):
#             spatial_feature = torch.zeros(
#                 self.num_bev_features,
#                 self.nz * self.nx * self.ny,
#                 dtype=pillar_features.dtype,
#                 device=pillar_features.device)
#             # batch_index的mask
#             batch_mask = coords[:, 0] == batch_idx
#             # 根据mask提取坐标
#             this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
#             # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
#             indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
#             # 转换数据类型
#             indices = indices.type(torch.long)
#             # 根据mask提取pillar_features
#             pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
#             pillars = pillars.t() # (64,batch_idx_voxel)
#             # 在索引位置填充pillars
#             spatial_feature[:, indices] = pillars
#             # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
#             batch_spatial_features.append(spatial_feature)
#
#         batch_spatial_features = \
#             torch.stack(batch_spatial_features, 0)
#         batch_spatial_features = \
#             batch_spatial_features.view(batch_size, self.num_bev_features *
#                                         self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
#         batch_dict['spatial_features'] = batch_spatial_features
#
#         return batch_dict
##################原代码#####################

##################修改后的代码#####################
#把“按 mask 取子集”改成“不用布尔索引的定长 scatter
#把“先筛出这一批的点，再写入”改成“直接一次性按 (batch, y, x) 线性索引写入”
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']      # (M, C)
        coords = batch_dict['voxel_coords']                  # (M, 4), [b, z, y, x]

        batch_size = int(coords[:, 0].max().item()) + 1
        C = self.num_bev_features
        HW = self.nz * self.nx * self.ny

        # (B, C, HW)
        batch_spatial_features = torch.zeros(
            batch_size,
            C,
            HW,
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )

        # 线性位置: z + y * nx + x
        indices = coords[:, 1] + coords[:, 2] * self.nx + coords[:, 3]
        indices = indices.long()                              # (M,)
        batch_idx = coords[:, 0].long()                       # (M,)

        # 展平成 (B*HW, C)，直接写入
        spatial_flat = batch_spatial_features.permute(0, 2, 1).contiguous()
        spatial_flat = spatial_flat.view(batch_size * HW, C)

        linear_idx = batch_idx * HW + indices                 # (M,)
        spatial_flat[linear_idx] = pillar_features            # (M, C)

        batch_spatial_features = spatial_flat.view(batch_size, HW, C)
        batch_spatial_features = batch_spatial_features.permute(0, 2, 1).contiguous()
        batch_spatial_features = batch_spatial_features.view(
            batch_size, C * self.nz, self.ny, self.nx
        )

        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
##################修改后的代码#####################




class SimplePointPillarScatter(nn.Module):
    def __init__(self, feature_dim, grid_size):
        super().__init__()

        self.num_bev_features = feature_dim
        self.nx, self.ny, self.nz = grid_size  # [704, 200, 1] 

        assert self.nz == 1

    def forward(self, pillar_features, coords):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
            pillars = pillars.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature) 

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]

        return batch_spatial_features

