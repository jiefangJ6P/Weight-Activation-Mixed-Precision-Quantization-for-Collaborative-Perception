import argparse
import logging
import os

import torch

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from static_voxel_deploy import MAX_VOXELS, pad_to_max_voxels, patch_model_for_static_voxels


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")


def parse_args():
    parser = argparse.ArgumentParser(description="Export the static Where2comm model to ONNX")
    parser.add_argument("--config", required=True, help="OpenCOOD YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--sample-index", type=int, default=21)
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [move_to_device(item, device) for item in value]
        return type(value)(items)
    return value


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    model.load_state_dict(checkpoint, strict=True)


def pick_collate_fn(dataset):
    collate_fn = getattr(dataset, "collate_batch_test", None)
    if callable(collate_fn):
        return collate_fn
    raise RuntimeError("The dataset does not provide collate_batch_test")


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_coords, voxel_num_points, record_len, pairwise_t_matrix):
        data_dict = {
            "processed_lidar": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords.to(torch.int64),
                "voxel_num_points": voxel_num_points.to(torch.int64),
            },
            "record_len": record_len.to(torch.int64),
            "pairwise_t_matrix": pairwise_t_matrix,
        }
        output = self.model(data_dict)
        return (
            output["cls_preds"],
            output["reg_preds"],
            output["dir_preds"],
            output["comm_rate"],
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypes = yaml_utils.load_yaml(args.config, None)
    dataset = build_dataset(hypes, visualize=False, train=False)
    model = train_utils.create_model(hypes).to(device).eval()
    load_checkpoint(model, args.checkpoint)
    patch_model_for_static_voxels(model)

    batch = pick_collate_fn(dataset)([dataset[args.sample_index]])
    batch = move_to_device(batch, device)
    ego = batch["ego"]
    processed_lidar = ego["processed_lidar"]
    voxel_features, voxel_coords, voxel_num_points = pad_to_max_voxels(
        processed_lidar["voxel_features"].to(torch.float32),
        processed_lidar["voxel_coords"].to(torch.int32),
        processed_lidar["voxel_num_points"].to(torch.int32),
        max_voxels=args.max_voxels,
    )
    inputs = (
        voxel_features,
        voxel_coords,
        voxel_num_points,
        ego["record_len"].to(torch.int32),
        ego["pairwise_t_matrix"].to(torch.float32),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logger.info("Exporting sample %d with %d padded voxels", args.sample_index, args.max_voxels)
    torch.onnx.export(
        OnnxExportWrapper(model).to(device).eval(),
        inputs,
        args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[
            "voxel_features",
            "voxel_coords",
            "voxel_num_points",
            "record_len",
            "pairwise_t_matrix",
        ],
        output_names=["cls_preds", "reg_preds", "dir_preds", "comm_rate"],
    )
    logger.info("Saved ONNX model to %s", args.output)


if __name__ == "__main__":
    main()
