import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from static_voxel_deploy import MAX_VOXELS, pad_to_max_voxels


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("export_validation_inputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Export fixed-shape validation inputs for TensorRT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, help="Local directory that receives .npy files")
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Optional target-device path recorded in metadata; defaults to output-root",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-record-len", type=int, default=2)
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--validate-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    return parser.parse_args()


def prepare_hypes(args):
    hypes = yaml_utils.load_yaml(args.config, None)
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = getattr(args, key)
        if value is not None:
            hypes[key] = value
    return hypes


def pick_collate_fn(dataset):
    collate_fn = getattr(dataset, "collate_batch_test", None)
    if callable(collate_fn):
        return collate_fn
    raise RuntimeError("The dataset does not provide collate_batch_test")


def as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    return value.astype(dtype, copy=False) if dtype is not None else value


def record_len_value(batch):
    values = as_numpy(batch["ego"]["record_len"]).reshape(-1)
    if values.size != 1:
        raise RuntimeError("This exporter requires batch size 1")
    return int(values[0])


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def export_sample(batch, dataset_index, export_index, args):
    sample_name = f"{export_index:06d}"
    sample_dir = os.path.join(args.output_root, sample_name)
    runtime_root = args.runtime_root or os.path.abspath(args.output_root)
    runtime_dir = os.path.join(runtime_root, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    ego = batch["ego"]
    processed_lidar = ego["processed_lidar"]
    voxel_features, voxel_coords, voxel_num_points = pad_to_max_voxels(
        processed_lidar["voxel_features"].to(torch.float32),
        processed_lidar["voxel_coords"].to(torch.int32),
        processed_lidar["voxel_num_points"].to(torch.int32),
        max_voxels=args.max_voxels,
    )
    arrays = {
        "voxel_features": as_numpy(voxel_features, np.float32),
        "voxel_coords": as_numpy(voxel_coords, np.int32),
        "voxel_num_points": as_numpy(voxel_num_points, np.int32),
        "record_len": as_numpy(ego["record_len"], np.int32),
        "pairwise_t_matrix": as_numpy(ego["pairwise_t_matrix"], np.float32),
    }
    for name, array in arrays.items():
        np.save(os.path.join(sample_dir, f"{name}.npy"), array)

    structure = {
        "record_len": arrays["record_len"].tolist(),
        "cav_id_list": json_safe(ego.get("cav_id_list")),
        "lidar_pose": json_safe(ego.get("lidar_pose")),
        "lidar_pose_clean": json_safe(ego.get("lidar_pose_clean")),
    }
    meta = {
        "sample_idx": dataset_index,
        "export_sample_idx": export_index,
        "sample_name": sample_name,
        "relative_dir": sample_name,
        "sample_dir": runtime_dir,
        "structure_meta": structure,
        "inputs": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return meta


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    dataset = build_dataset(prepare_hypes(args), visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=pick_collate_fn(dataset),
    )

    records = []
    for dataset_index, batch in enumerate(loader):
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
        if batch is None or record_len_value(batch) != args.keep_record_len:
            continue
        records.append(export_sample(batch, dataset_index, len(records), args))
        if len(records) == 1 or len(records) % 50 == 0:
            logger.info("Exported %d samples", len(records))

    index_path = os.path.join(args.output_root, "index.json")
    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    logger.info("Exported %d samples to %s", len(records), args.output_root)


if __name__ == "__main__":
    main()
