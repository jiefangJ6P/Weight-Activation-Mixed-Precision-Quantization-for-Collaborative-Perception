import os
import sys
import logging
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader

from opencood.tools import train_utils
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_TOOLS_DIR = os.path.dirname(THIS_DIR)
if ONNX_TOOLS_DIR not in sys.path:
    sys.path.insert(0, ONNX_TOOLS_DIR)

from static_voxel_deploy import (
    MAX_VOXELS,
    pad_to_max_voxels,
    patch_model_for_static_voxels,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("verify_onnx")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX Runtime outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--num-samples", type=int, default=21)
    parser.add_argument("--export-sample-index", type=int, default=21)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--validate-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--cuda-provider", action="store_true")
    return parser.parse_args()


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        xs = [move_to_device(v, device) for v in x]
        return type(x)(xs)
    return x


def load_ckpt_to_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model_state", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def pick_collate_fn(dataset):
    if hasattr(dataset, "collate_batch_test") and callable(getattr(dataset, "collate_batch_test")):
        logger.info("Using collate_fn = dataset.collate_batch_test")
        return dataset.collate_batch_test
    raise RuntimeError("dataset 没有 collate_batch_test，请检查 dataset 版本")


class OnnxExportWrapper(torch.nn.Module):
    """
    和你导出 ONNX 时保持一致：
    把 OpenCOOD 的 dict 输入包装成 5 个 Tensor 输入
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_coords, voxel_num_points, record_len, pairwise_t_matrix):
        voxel_coords = voxel_coords.to(torch.int64)
        voxel_num_points = voxel_num_points.to(torch.int64)
        record_len = record_len.to(torch.int64)

        data_dict = {
            "processed_lidar": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            },
            "record_len": record_len,
            "pairwise_t_matrix": pairwise_t_matrix,
        }

        out = self.model(data_dict)
        return (
            out["cls_preds"],
            out["reg_preds"],
            out["dir_preds"],
            out["comm_rate"],
        )


def prepare_hypes(args):
    hypes = yaml_utils.load_yaml(args.config, None)
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = getattr(args, key)
        if value is not None:
            hypes[key] = value

    return hypes


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def print_io_info(sess):
    logger.info("===== ONNX Runtime Inputs =====")
    for i, inp in enumerate(sess.get_inputs()):
        logger.info(f"[Input {i}] name={inp.name}, shape={inp.shape}, type={inp.type}")

    logger.info("===== ONNX Runtime Outputs =====")
    for i, out in enumerate(sess.get_outputs()):
        logger.info(f"[Output {i}] name={out.name}, shape={out.shape}, type={out.type}")


def ort_dtype_from_type_str(type_str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if type_str not in mapping:
        raise KeyError(f"Unsupported ORT input type: {type_str}")
    return mapping[type_str]


def compare_tensor(name, torch_out, ort_out):
    torch_np = to_numpy(torch_out).astype(np.float32)
    ort_np = np.asarray(ort_out).astype(np.float32)

    result = {
        "name": name,
        "shape_match": torch_np.shape == ort_np.shape,
        "torch_shape": tuple(torch_np.shape),
        "ort_shape": tuple(ort_np.shape),
    }

    if torch_np.shape != ort_np.shape:
        logger.error(f"{name}: shape mismatch, torch={torch_np.shape}, ort={ort_np.shape}")
        return result

    abs_diff = np.abs(torch_np - ort_np)
    max_abs = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0

    denom = np.maximum(np.abs(torch_np), 1e-6)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max()) if rel_diff.size > 0 else 0.0
    mean_rel = float(rel_diff.mean()) if rel_diff.size > 0 else 0.0

    result.update({
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
    })

    logger.info(
        f"{name}: "
        f"shape={torch_np.shape}, "
        f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}, "
        f"max_rel={max_rel:.8f}, mean_rel={mean_rel:.8f}"
    )
    return result


def prepare_inputs_from_sample(dataset, sample_index, device):
    sample = dataset[sample_index]
    batch = pick_collate_fn(dataset)([sample])
    batch = move_to_device(batch, device)

    ego = batch["ego"]
    pairwise_t_matrix = ego["pairwise_t_matrix"].to(torch.float32)
    pl = ego["processed_lidar"]

    voxel_features = pl["voxel_features"].to(torch.float32)
    voxel_coords = pl["voxel_coords"].to(torch.int32)
    voxel_num_points = pl["voxel_num_points"].to(torch.int32)
    record_len = ego["record_len"].to(torch.int32)
    voxel_features, voxel_coords, voxel_num_points = pad_to_max_voxels(
        voxel_features=voxel_features,
        voxel_coords=voxel_coords,
        voxel_num_points=voxel_num_points,
        max_voxels=MAX_VOXELS,
    )

    return {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
        "record_len": record_len,
        "pairwise_t_matrix": pairwise_t_matrix,
    }


def summarize_sample_structure(inputs):
    record_len_list = to_numpy(inputs["record_len"]).astype(np.int64).tolist()
    valid_agents = int(sum(record_len_list))
    pairwise_shape = tuple(inputs["pairwise_t_matrix"].shape)
    voxel_count = int(inputs["voxel_features"].shape[0])
    return {
        "record_len": record_len_list,
        "valid_agents": valid_agents,
        "pairwise_shape": pairwise_shape,
        "voxel_count": voxel_count,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")

    # 1) 检查 ONNX 文件是否合法
    logger.info("Checking ONNX model...")
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX check passed.")

    # 2) 准备 hypes / dataset / dataloader
    hypes = prepare_hypes(args)

    dataset = build_dataset(hypes, visualize=False, train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=pick_collate_fn(dataset),
    )

    # 3) 加载 PyTorch 模型
    model = train_utils.create_model(hypes).to(device).eval()
    load_ckpt_to_model(model, args.checkpoint)
    patch_model_for_static_voxels(model)
    logger.info("PyTorch model + ckpt loaded")

    wrapper = OnnxExportWrapper(model).to(device).eval()

    # 4) ONNX Runtime Session
    providers = ["CPUExecutionProvider"]
    if args.cuda_provider:
        providers.insert(0, "CUDAExecutionProvider")

    sess = ort.InferenceSession(args.onnx, providers=providers)
    print_io_info(sess)

    # 5) 验证前 NUM_VERIFY_SAMPLES 条数据
    total_samples = min(args.num_samples, len(dataset))
    logger.info(f"Verifying first {total_samples} samples")
    export_inputs = prepare_inputs_from_sample(dataset, args.export_sample_index, device)
    export_structure = summarize_sample_structure(export_inputs)
    logger.info(
        f"Export reference sample index = {args.export_sample_index}, "
        f"record_len = {export_structure['record_len']}, "
        f"valid_agents = {export_structure['valid_agents']}, "
        f"pairwise_shape = {export_structure['pairwise_shape']}, "
        f"voxel_count = {export_structure['voxel_count']}"
    )
    all_results = []

    for sample_index in range(total_samples):
        inputs = prepare_inputs_from_sample(dataset, sample_index, device)
        sample_structure = summarize_sample_structure(inputs)

        logger.info("=" * 80)
        logger.info(f"Verifying sample index = {sample_index}")
        logger.info(
            f"voxel_features shape = {tuple(inputs['voxel_features'].shape)}, "
            f"dtype = {inputs['voxel_features'].dtype}"
        )
        logger.info(
            f"voxel_coords shape = {tuple(inputs['voxel_coords'].shape)}, "
            f"dtype = {inputs['voxel_coords'].dtype}"
        )
        logger.info(
            f"voxel_num_points shape = {tuple(inputs['voxel_num_points'].shape)}, "
            f"dtype = {inputs['voxel_num_points'].dtype}"
        )
        logger.info(
            f"record_len shape = {tuple(inputs['record_len'].shape)}, "
            f"dtype = {inputs['record_len'].dtype}"
        )
        logger.info(f"record_len values = {sample_structure['record_len']}")
        logger.info(f"valid_agents = {sample_structure['valid_agents']}")
        logger.info(
            f"pairwise_t_matrix shape = {tuple(inputs['pairwise_t_matrix'].shape)}, "
            f"dtype = {inputs['pairwise_t_matrix'].dtype}"
        )
        logger.info(
            f"sample vs export structure: "
            f"same_record_len = {sample_structure['record_len'] == export_structure['record_len']}, "
            f"same_valid_agents = {sample_structure['valid_agents'] == export_structure['valid_agents']}, "
            f"same_pairwise_shape = {sample_structure['pairwise_shape'] == export_structure['pairwise_shape']}"
        )

        with torch.no_grad():
            torch_outputs = wrapper(
                inputs["voxel_features"],
                inputs["voxel_coords"],
                inputs["voxel_num_points"],
                inputs["record_len"],
                inputs["pairwise_t_matrix"],
            )

        torch_cls, torch_reg, torch_dir, torch_comm = torch_outputs
        logger.info("PyTorch inference done")

        ort_inputs = {}
        for inp in sess.get_inputs():
            value = to_numpy(inputs[inp.name])
            ort_inputs[inp.name] = np.ascontiguousarray(
                value.astype(ort_dtype_from_type_str(inp.type), copy=False)
            )

        ort_outputs = sess.run(None, ort_inputs)
        logger.info("ONNX Runtime inference done")

        ort_cls, ort_reg, ort_dir, ort_comm = ort_outputs

        logger.info("===== Compare PyTorch vs ONNX Runtime =====")
        sample_result = {
            "sample_index": sample_index,
            "record_len": sample_structure["record_len"],
            "valid_agents": sample_structure["valid_agents"],
            "pairwise_shape": sample_structure["pairwise_shape"],
            "same_record_len_as_export": sample_structure["record_len"] == export_structure["record_len"],
            "same_valid_agents_as_export": sample_structure["valid_agents"] == export_structure["valid_agents"],
            "same_pairwise_shape_as_export": sample_structure["pairwise_shape"] == export_structure["pairwise_shape"],
            "comparisons": [],
        }
        sample_result["comparisons"].append(
            compare_tensor(f"sample_{sample_index}/cls_preds", torch_cls, ort_cls)
        )
        sample_result["comparisons"].append(
            compare_tensor(f"sample_{sample_index}/reg_preds", torch_reg, ort_reg)
        )
        sample_result["comparisons"].append(
            compare_tensor(f"sample_{sample_index}/dir_preds", torch_dir, ort_dir)
        )
        sample_result["comparisons"].append(
            compare_tensor(f"sample_{sample_index}/comm_rate", torch_comm, ort_comm)
        )
        all_results.append(sample_result)

    logger.info("=" * 80)
    logger.info("Verification finished.")
    logger.info("Summary:")
    for result in all_results:
        logger.info(
            f"[sample_{result['sample_index']}] "
            f"record_len={result['record_len']}, "
            f"valid_agents={result['valid_agents']}, "
            f"pairwise_shape={result['pairwise_shape']}, "
            f"same_record_len_as_export={result['same_record_len_as_export']}, "
            f"same_valid_agents_as_export={result['same_valid_agents_as_export']}, "
            f"same_pairwise_shape_as_export={result['same_pairwise_shape_as_export']}"
        )
        for cmp_result in result["comparisons"]:
            if not cmp_result.get("shape_match", False):
                logger.info(
                    f"  - {cmp_result['name']}: shape mismatch, "
                    f"torch={cmp_result['torch_shape']}, ort={cmp_result['ort_shape']}"
                )
            else:
                logger.info(
                    f"  - {cmp_result['name']}: "
                    f"max_abs={cmp_result['max_abs']:.8f}, "
                    f"mean_abs={cmp_result['mean_abs']:.8f}, "
                    f"max_rel={cmp_result['max_rel']:.8f}, "
                    f"mean_rel={cmp_result['mean_rel']:.8f}"
                )


if __name__ == "__main__":
    main()
