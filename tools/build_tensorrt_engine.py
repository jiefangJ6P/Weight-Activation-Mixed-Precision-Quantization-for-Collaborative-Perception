import argparse
import ctypes
import json
import logging
import os
from typing import Dict, List

import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("build_trt_int8_engine")


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine with representative .npy samples")
    parser.add_argument("--onnx_path", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--engine_path", type=str, required=True, help="Output TensorRT engine path")
    parser.add_argument(
        "--build_mode",
        type=str,
        default="fp32_calib",
        choices=["fp32_calib", "qdq_explicit"],
        help="INT8 build mode: FP32 ONNX with calibrator, or explicit Q/DQ ONNX without calibrator",
    )
    parser.add_argument("--input_root", type=str, default=None, help="Exported sample root")
    parser.add_argument("--index_path", type=str, default=None, help="index.json path")
    parser.add_argument("--cache_path", type=str, default=None, help="INT8 calibration cache path")
    parser.add_argument("--max_samples", type=int, default=50, help="Calibration sample count")
    parser.add_argument("--workspace_mib", type=int, default=4096, help="TensorRT workspace size in MiB")
    parser.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger")
    parser.add_argument("--no_fp16_fallback", action="store_true", help="Disable FP16 fallback during INT8 build")
    return parser.parse_args()


def init_trt_logger(verbose: bool):
    try:
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(f"Failed to load libnvinfer_plugin.so: {exc}")

    level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger_trt = trt.Logger(level)
    trt.init_libnvinfer_plugins(logger_trt, "")
    return logger_trt


def load_index_records(index_path: str, input_root: str, max_samples: int) -> List[str]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    sample_dirs = []
    for record in records:
        relative_dir = record.get("relative_dir") or record.get("sample_name")
        if not relative_dir:
            continue
        sample_dir = os.path.join(input_root, relative_dir)
        if os.path.isdir(sample_dir):
            sample_dirs.append(sample_dir)
        if max_samples is not None and len(sample_dirs) >= max_samples:
            break

    if not sample_dirs:
        raise RuntimeError("No calibration sample directories found")
    return sample_dirs


def load_sample(sample_dir: str) -> Dict[str, np.ndarray]:
    names = [
        "voxel_features",
        "voxel_coords",
        "voxel_num_points",
        "record_len",
        "pairwise_t_matrix",
    ]
    feed = {}
    for name in names:
        path = os.path.join(sample_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        feed[name] = np.ascontiguousarray(np.load(path))
    return feed


class SampleStream:
    def __init__(self, sample_dirs: List[str]):
        self.sample_dirs = sample_dirs
        self.index = 0

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.sample_dirs):
            return None
        sample_dir = self.sample_dirs[self.index]
        self.index += 1
        return load_sample(sample_dir)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, sample_stream: SampleStream, input_specs: Dict[str, Dict], cache_path: str):
        super().__init__()
        self.sample_stream = sample_stream
        self.input_specs = input_specs
        self.cache_path = cache_path
        self.device_buffers = {}
        self.bindings = []

        for name, spec in input_specs.items():
            nbytes = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
            self.device_buffers[name] = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        sample = self.sample_stream.next()
        if sample is None:
            return None

        self.bindings = []
        for name in names:
            if name not in sample:
                raise KeyError(f"Calibration sample missing input: {name}")
            arr = sample[name]
            spec = self.input_specs[name]
            if tuple(arr.shape) != tuple(spec["shape"]):
                raise RuntimeError(f"Shape mismatch for {name}: sample={arr.shape}, expected={spec['shape']}")
            if arr.dtype != spec["dtype"]:
                arr = np.ascontiguousarray(arr.astype(spec["dtype"]))
            cuda.memcpy_htod(self.device_buffers[name], arr)
            self.bindings.append(int(self.device_buffers[name]))
        return self.bindings

    def read_calibration_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Using existing calibration cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        logger.info(f"Calibration cache saved to: {self.cache_path}")


def build_input_specs(network) -> Dict[str, Dict]:
    specs = {}
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        specs[name] = {
            "dtype": np.dtype(trt.nptype(tensor.dtype)),
            "shape": tuple(tensor.shape),
            "is_shape_tensor": bool(getattr(tensor, "is_shape_tensor", False)),
        }
    return specs


def configure_profile(builder, network, config, use_calibration_profile: bool = True):
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        if name == "voxel_features":
            profile.set_shape(name, (26640, 32, 4), (26640, 32, 4), (26640, 32, 4))
        elif name == "voxel_coords":
            profile.set_shape(name, (26640, 4), (26640, 4), (26640, 4))
        elif name == "voxel_num_points":
            profile.set_shape(name, (26640,), (26640,), (26640,))
        elif name == "record_len":
            profile.set_shape_input(name, (2,), (2,), (2,))
        elif name == "pairwise_t_matrix":
            profile.set_shape(name, (1, 5, 5, 4, 4), (1, 5, 5, 4, 4), (1, 5, 5, 4, 4))
        else:
            raise KeyError(f"Unexpected network input: {name}")

    config.add_optimization_profile(profile)
    if use_calibration_profile and hasattr(config, "set_calibration_profile"):
        config.set_calibration_profile(profile)
    return profile


def set_workspace_limit(config, workspace_mib: int):
    workspace_bytes = int(workspace_mib) << 20
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        logger.info(f"TensorRT workspace set with set_memory_pool_limit: {workspace_mib} MiB")
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes
        logger.info(f"TensorRT workspace set with max_workspace_size: {workspace_mib} MiB")
    else:
        logger.warning(
            "This TensorRT IBuilderConfig has neither set_memory_pool_limit nor "
            "max_workspace_size; using TensorRT default workspace size."
        )


def main():
    args = parse_args()
    logger_trt = init_trt_logger(args.verbose)

    sample_dirs = None
    if args.build_mode == "fp32_calib":
        if not args.input_root or not args.index_path or not args.cache_path:
            raise ValueError("fp32_calib requires --input_root, --index_path, and --cache_path")
        sample_dirs = load_index_records(args.index_path, args.input_root, args.max_samples)
        logger.info(f"Calibration samples: {len(sample_dirs)}")
    else:
        logger.info("Build mode: qdq_explicit")
        logger.info("Skipping TensorRT calibrator. Expecting ONNX with explicit Q/DQ nodes.")

    with trt.Builder(logger_trt) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger_trt) as parser:

        with open(args.onnx_path, "rb") as f:
            ok = parser.parse(f.read())
        if not ok:
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

        config = builder.create_builder_config()
        set_workspace_limit(config, args.workspace_mib)
        config.set_flag(trt.BuilderFlag.INT8)
        if not args.no_fp16_fallback:
            config.set_flag(trt.BuilderFlag.FP16)

        configure_profile(
            builder,
            network,
            config,
            use_calibration_profile=(args.build_mode == "fp32_calib"),
        )
        input_specs = build_input_specs(network)

        logger.info("Network inputs:")
        for name, spec in input_specs.items():
            logger.info(
                f"  {name}: shape={spec['shape']}, dtype={spec['dtype']}, "
                f"is_shape_tensor={spec['is_shape_tensor']}"
            )

        if args.build_mode == "fp32_calib":
            stream = SampleStream(sample_dirs)
            calibrator = EntropyCalibrator(stream, input_specs, args.cache_path)
            config.int8_calibrator = calibrator
            logger.info("Using TensorRT entropy calibrator with representative samples.")
        else:
            logger.info("Using explicit quantization from ONNX graph; no calibrator attached.")

        logger.info("Building INT8 engine. This may take several minutes.")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build serialized INT8 engine")

    os.makedirs(os.path.dirname(os.path.abspath(args.engine_path)), exist_ok=True)
    with open(args.engine_path, "wb") as f:
        f.write(serialized)
    logger.info(f"INT8 engine saved to: {args.engine_path}")


if __name__ == "__main__":
    main()
