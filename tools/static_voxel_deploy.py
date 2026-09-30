import types

import torch


MAX_VOXELS = 26640


def pad_to_max_voxels(
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_num_points: torch.Tensor,
    max_voxels: int = MAX_VOXELS,
):
    n = voxel_features.shape[0]

    if voxel_coords.shape[0] != n or voxel_num_points.shape[0] != n:
        raise RuntimeError(
            "输入第0维不一致: "
            f"voxel_features={tuple(voxel_features.shape)}, "
            f"voxel_coords={tuple(voxel_coords.shape)}, "
            f"voxel_num_points={tuple(voxel_num_points.shape)}"
        )

    if n > max_voxels:
        raise RuntimeError(
            f"当前样本 voxel 数 {n} 超过固定上限 MAX_VOXELS={max_voxels}"
        )

    if n == max_voxels:
        return (
            voxel_features.contiguous(),
            voxel_coords.contiguous(),
            voxel_num_points.contiguous(),
        )

    pad_n = max_voxels - n

    vf_pad = torch.zeros(
        (pad_n, voxel_features.shape[1], voxel_features.shape[2]),
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )
    vc_pad = torch.zeros(
        (pad_n, voxel_coords.shape[1]),
        dtype=voxel_coords.dtype,
        device=voxel_coords.device,
    )
    vnp_pad = torch.zeros(
        (pad_n,),
        dtype=voxel_num_points.dtype,
        device=voxel_num_points.device,
    )

    return (
        torch.cat([voxel_features, vf_pad], dim=0).contiguous(),
        torch.cat([voxel_coords, vc_pad], dim=0).contiguous(),
        torch.cat([voxel_num_points, vnp_pad], dim=0).contiguous(),
    )


def _pillar_vfe_forward_static(self, batch_dict):
    voxel_features = batch_dict["voxel_features"]
    voxel_num_points = batch_dict["voxel_num_points"]
    coords = batch_dict["voxel_coords"]

    safe_num_points = torch.where(
        voxel_num_points > 0,
        voxel_num_points,
        torch.ones_like(voxel_num_points),
    )

    points_mean = (
        voxel_features[:, :, :3].sum(dim=1, keepdim=True)
        / safe_num_points.type_as(voxel_features).view(-1, 1, 1)
    )
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (
        coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
    )
    f_center[:, :, 1] = voxel_features[:, :, 1] - (
        coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
    )
    f_center[:, :, 2] = voxel_features[:, :, 2] - (
        coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
    )

    if self.use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]

    if self.with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)

    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = self.get_paddings_indicator(safe_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
    features = features * mask

    for pfn in self.pfn_layers:
        features = pfn(features)

    pillar_features = features.squeeze()
    if pillar_features.dim() == 1:
        pillar_features = pillar_features.unsqueeze(0)

    valid_pillar_mask = (voxel_num_points > 0).unsqueeze(-1).type_as(pillar_features)
    pillar_features = pillar_features * valid_pillar_mask

    batch_dict["pillar_features"] = pillar_features
    return batch_dict


def _scatter_forward_static(self, batch_dict):
    pillar_features = batch_dict["pillar_features"]
    coords = batch_dict["voxel_coords"]
    voxel_num_points = batch_dict.get("voxel_num_points")

    batch_size = int(coords[:, 0].max().item()) + 1
    c = self.num_bev_features
    hw = self.nz * self.nx * self.ny

    batch_spatial_features = torch.zeros(
        batch_size,
        c,
        hw + 1,
        dtype=pillar_features.dtype,
        device=pillar_features.device,
    )

    indices = coords[:, 1] + coords[:, 2] * self.nx + coords[:, 3]
    indices = indices.long()
    batch_idx = coords[:, 0].long()

    if voxel_num_points is None:
        valid_mask = torch.ones_like(indices, dtype=torch.bool)
    else:
        valid_mask = voxel_num_points > 0

    sentinel_index = torch.full_like(indices, hw)
    safe_indices = torch.where(valid_mask, indices, sentinel_index)

    spatial_flat = batch_spatial_features.permute(0, 2, 1).contiguous()
    spatial_flat = spatial_flat.view(batch_size * (hw + 1), c)

    linear_idx = batch_idx * (hw + 1) + safe_indices
    spatial_flat[linear_idx] = pillar_features

    batch_spatial_features = spatial_flat.view(batch_size, hw + 1, c)
    batch_spatial_features = batch_spatial_features[:, :hw, :]
    batch_spatial_features = batch_spatial_features.permute(0, 2, 1).contiguous()
    batch_spatial_features = batch_spatial_features.view(
        batch_size, c * self.nz, self.ny, self.nx
    )

    batch_dict["spatial_features"] = batch_spatial_features
    return batch_dict


def patch_model_for_static_voxels(model):
    if getattr(model, "_static_voxel_deploy_patched", False):
        return model

    if not hasattr(model, "pillar_vfe") or not hasattr(model, "scatter"):
        raise AttributeError("model 缺少 pillar_vfe 或 scatter，无法应用 static voxel patch")

    model.pillar_vfe.forward = types.MethodType(_pillar_vfe_forward_static, model.pillar_vfe)
    model.scatter.forward = types.MethodType(_scatter_forward_static, model.scatter)
    model._static_voxel_deploy_patched = True
    return model
