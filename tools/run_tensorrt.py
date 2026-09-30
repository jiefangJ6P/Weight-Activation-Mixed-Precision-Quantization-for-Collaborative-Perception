import os
import json
import time
import ctypes
import logging
import argparse
import shutil
import subprocess
import threading

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("batch_run_trt_where2comm")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch TensorRT inference for where2comm on NX")
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="TensorRT engine path"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory of exported input samples"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="index.json path"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only run the first N samples from index.json"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they already exist"
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU index used by nvidia-smi for power sampling"
    )
    parser.add_argument(
        "--power_interval_sec",
        type=float,
        default=0.1,
        help="Power sampling interval in seconds"
    )
    parser.add_argument(
        "--disable_power_monitor",
        action="store_true",
        help="Disable nvidia-smi power sampling"
    )
    return parser.parse_args()


def init_trt_plugins():
    try:
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise RuntimeError(f"Failed to load libnvinfer_plugin.so: {e}")

    logger_trt = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger_trt, "")
    return logger_trt


def trt_nptype(dtype):
    return np.dtype(trt.nptype(dtype))


class NvidiaSmiPowerMonitor:
    def __init__(self, gpu_index=0, interval_sec=0.1):
        self.gpu_index = gpu_index
        self.interval_sec = max(float(interval_sec), 0.01)
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self._end_time = None
        self._available = shutil.which("nvidia-smi") is not None

    def start(self):
        if not self._available:
            logger.warning("nvidia-smi not found; power monitoring is disabled.")
            return False

        self.samples = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._end_time = None
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if self._thread is None:
            return self.summary()

        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_sec * 5.0))
        self._end_time = time.perf_counter()
        return self.summary()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            timestamp = time.perf_counter()
            power_w = self._read_power_w()
            if power_w is not None:
                self.samples.append((timestamp, power_w))
            self._stop_event.wait(self.interval_sec)

    def _read_power_w(self):
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
            )
        except Exception:
            return None

        first_line = out.strip().splitlines()[0].strip() if out.strip() else ""
        if not first_line:
            return None
        try:
            return float(first_line)
        except ValueError:
            return None

    def summary(self):
        duration = None
        if self._start_time is not None:
            end_time = self._end_time if self._end_time is not None else time.perf_counter()
            duration = max(0.0, end_time - self._start_time)

        if not self.samples:
            return {
                "enabled": self._available,
                "gpu_index": self.gpu_index,
                "sample_count": 0,
                "sampling_interval_sec": self.interval_sec,
                "duration_sec": duration,
                "avg_power_w": None,
                "min_power_w": None,
                "max_power_w": None,
                "energy_j": None,
            }

        powers = [p for _, p in self.samples]
        avg_power = float(sum(powers) / len(powers))
        energy_j = avg_power * duration if duration is not None else None
        return {
            "enabled": True,
            "gpu_index": self.gpu_index,
            "sample_count": len(powers),
            "sampling_interval_sec": self.interval_sec,
            "duration_sec": duration,
            "avg_power_w": avg_power,
            "min_power_w": float(min(powers)),
            "max_power_w": float(max(powers)),
            "energy_j": float(energy_j) if energy_j is not None else None,
        }


class TensorRTWhere2CommBatchRunner:
    def __init__(self, engine_path, logger_trt):
        self.engine_path = engine_path
        self.logger_trt = logger_trt

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger_trt) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()

        self.input_names = []
        self.output_names = []
        self.use_tensor_api = hasattr(self.engine, "num_io_tensors")
        self.tensor_names = []
        self.binding_name_to_index = {}

        if self.use_tensor_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self.tensor_names.append(name)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.tensor_names.append(name)
                self.binding_name_to_index[name] = i
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

        logger.info("TensorRT engine loaded successfully")
        logger.info(f"Inputs: {self.input_names}")
        logger.info(f"Outputs: {self.output_names}")
        for name in self._iter_tensor_names():
            logger.info(
                f"Tensor meta | name={name} | mode={'INPUT' if self._is_input(name) else 'OUTPUT'} "
                f"| shape={self._get_shape(name)} | dtype={self._get_dtype(name)} "
                f"| location={self._get_location(name)} | shape_io={self._is_shape_inference_io(name)}"
            )

    def _iter_tensor_names(self):
        return self.tensor_names

    def _binding_index(self, name_or_index):
        if isinstance(name_or_index, str):
            return self.binding_name_to_index[name_or_index]
        return name_or_index

    def _is_input(self, name_or_index):
        if self.use_tensor_api:
            return self.engine.get_tensor_mode(name_or_index) == trt.TensorIOMode.INPUT
        return self.engine.binding_is_input(self._binding_index(name_or_index))

    def _get_shape(self, name_or_index):
        if self.use_tensor_api:
            return tuple(self.engine.get_tensor_shape(name_or_index))
        return tuple(self.engine.get_binding_shape(self._binding_index(name_or_index)))

    def _get_dtype(self, name_or_index):
        if self.use_tensor_api:
            return self.engine.get_tensor_dtype(name_or_index)
        return self.engine.get_binding_dtype(self._binding_index(name_or_index))
#Segmentation fault (core dumped)
    def _get_location(self, name_or_index):
        if self.use_tensor_api:
            return self.engine.get_tensor_location(name_or_index)
        return self.engine.get_location(self._binding_index(name_or_index))

    def _is_shape_inference_io(self, name):
        if self.use_tensor_api and hasattr(self.engine, "is_shape_inference_io"):
            return bool(self.engine.is_shape_inference_io(name))
        return False

    def _set_input_shape(self, name_or_index, shape):
        if self.use_tensor_api:
            self.context.set_input_shape(name_or_index, shape)
            return True
        return self.context.set_binding_shape(self._binding_index(name_or_index), shape)

    def _get_context_shape(self, name_or_index):
        if self.use_tensor_api:
            return tuple(self.context.get_tensor_shape(name_or_index))
        return tuple(self.context.get_binding_shape(self._binding_index(name_or_index)))

    def _all_input_shapes_specified(self):
        if self.use_tensor_api:
            return True
        return self.context.all_binding_shapes_specified

    def _load_sample_inputs(self, sample_dir):
        required = [
            "voxel_features.npy",
            "voxel_coords.npy",
            "voxel_num_points.npy",
            "record_len.npy",
            "pairwise_t_matrix.npy",
        ]

        for name in required:
            path = os.path.join(sample_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing input file: {path}")

        voxel_features = np.load(os.path.join(sample_dir, "voxel_features.npy")).astype(np.float32, copy=False)
        voxel_coords = np.load(os.path.join(sample_dir, "voxel_coords.npy"))
        voxel_num_points = np.load(os.path.join(sample_dir, "voxel_num_points.npy"))
        record_len = np.load(os.path.join(sample_dir, "record_len.npy"))
        pairwise_t_matrix = np.load(os.path.join(sample_dir, "pairwise_t_matrix.npy")).astype(np.float32, copy=False)

        return {
            "voxel_features": np.ascontiguousarray(voxel_features),
            "voxel_coords": np.ascontiguousarray(voxel_coords),
            "voxel_num_points": np.ascontiguousarray(voxel_num_points),
            "record_len": np.ascontiguousarray(record_len),
            "pairwise_t_matrix": np.ascontiguousarray(pairwise_t_matrix),
        }

    def infer_one_sample(self, sample_dir):
        host_inputs = self._load_sample_inputs(sample_dir)

        for name in self._iter_tensor_names():
            if self._is_input(name):
                if name not in host_inputs:
                    raise KeyError(f"Input data not found for tensor: {name}")
                shape = tuple(host_inputs[name].shape)
                engine_shape = self._get_shape(name)

                # Static engine: input shape must match engine binding shape exactly.
                if all(int(x) >= 0 for x in engine_shape):
                    if engine_shape != shape:
                        raise RuntimeError(
                            f"Static engine shape mismatch for {name}: "
                            f"engine={engine_shape}, input={shape}"
                        )
                else:
                    ok = self._set_input_shape(name, shape)
                    if not ok:
                        raise RuntimeError(f"set_binding_shape failed for {name}, shape={shape}")

        if not self._all_input_shapes_specified():
            raise RuntimeError("Not all binding shapes are specified.")

        bindings = {}
        device_allocations = []
        host_outputs = {}

        try:
            for name in self._iter_tensor_names():
                dtype = trt_nptype(self._get_dtype(name))
                location = self._get_location(name)

                if self._is_input(name):
                    arr = host_inputs[name]
                    if arr.dtype != dtype:
                        logger.warning(f"Casting input {name} from {arr.dtype} to {dtype}")
                        arr = arr.astype(dtype)
                        host_inputs[name] = arr

                    if location == trt.TensorLocation.HOST:
                        bindings[name] = arr
                    else:
                        dmem = cuda.mem_alloc(arr.nbytes)
                        bindings[name] = dmem
                        device_allocations.append(dmem)
                else:
                    out_shape = self._get_context_shape(name)
                    if any(int(x) < 0 for x in out_shape):
                        raise RuntimeError(f"Output shape not resolved for {name}: {out_shape}")

                    host_arr = np.empty(out_shape, dtype=dtype)
                    if location == trt.TensorLocation.HOST:
                        bindings[name] = host_arr
                    else:
                        dmem = cuda.mem_alloc(host_arr.nbytes)
                        bindings[name] = dmem
                        device_allocations.append(dmem)
                    host_outputs[name] = host_arr

            infer_start = cuda.Event()
            infer_end = cuda.Event()

            if self.use_tensor_api:
                for name in self._iter_tensor_names():
                    buf = bindings[name]
                    addr = buf.ctypes.data if isinstance(buf, np.ndarray) else int(buf)
                    self.context.set_tensor_address(name, addr)

                for name in self.input_names:
                    if self._get_location(name) == trt.TensorLocation.DEVICE:
                        cuda.memcpy_htod_async(bindings[name], host_inputs[name], self.stream)

                infer_start.record(self.stream)
                ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
                if not ok:
                    raise RuntimeError("execute_async_v3 failed")
                infer_end.record(self.stream)

                for name in self.output_names:
                    if self._get_location(name) == trt.TensorLocation.DEVICE:
                        cuda.memcpy_dtoh_async(host_outputs[name], bindings[name], self.stream)
            else:
                ordered_bindings = [int(bindings[name]) for name in self._iter_tensor_names()]

                for name in self.input_names:
                    cuda.memcpy_htod_async(bindings[name], host_inputs[name], self.stream)

                infer_start.record(self.stream)
                ok = self.context.execute_async_v2(bindings=ordered_bindings, stream_handle=self.stream.handle)
                if not ok:
                    raise RuntimeError("execute_async_v2 failed")
                infer_end.record(self.stream)

                for name in self.output_names:
                    cuda.memcpy_dtoh_async(host_outputs[name], bindings[name], self.stream)

            self.stream.synchronize()
            infer_time_sec = infer_start.time_till(infer_end) / 1000.0

        finally:
            for dmem in device_allocations:
                try:
                    dmem.free()
                except Exception:
                    pass

        return host_outputs, infer_time_sec

    def save_outputs(self, sample_dir, outputs):
        save_names = ["cls_preds", "reg_preds", "dir_preds", "comm_rate"]

        output_meta = {
            "output_meta": {}
        }

        for name in save_names:
            if name not in outputs:
                raise KeyError(f"TensorRT output missing: {name}")

            arr = outputs[name]
            out_path = os.path.join(sample_dir, f"{name}.npy")
            np.save(out_path, arr)

            output_meta["output_meta"][name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()) if arr.size > 0 else None,
                "max": float(arr.max()) if arr.size > 0 else None,
                "mean": float(arr.mean()) if arr.size > 0 else None,
            }

        with open(os.path.join(sample_dir, "trt_output_meta.json"), "w", encoding="utf-8") as f:
            json.dump(output_meta, f, ensure_ascii=False, indent=2)


def should_skip(sample_dir, overwrite=False):
    required_outputs = [
        "cls_preds.npy",
        "reg_preds.npy",
        "dir_preds.npy",
        "comm_rate.npy",
        "trt_output_meta.json",
    ]
    if overwrite:
        return False
    return all(os.path.exists(os.path.join(sample_dir, x)) for x in required_outputs)


def resolve_sample_dir(record, input_root):
    if "sample_dir" in record and isinstance(record["sample_dir"], str):
        return record["sample_dir"]

    if "relative_dir" in record and isinstance(record["relative_dir"], str):
        return os.path.join(input_root, record["relative_dir"])

    if "sample_name" in record and isinstance(record["sample_name"], str):
        return os.path.join(input_root, record["sample_name"])

    raise KeyError(f"Cannot resolve sample directory from record: {record}")


def main():
    args = parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index.json not found: {args.index_path}")

    logger_trt = init_trt_plugins()
    runner = TensorRTWhere2CommBatchRunner(args.engine_path, logger_trt)

    with open(args.index_path, "r", encoding="utf-8") as f:
        index_records = json.load(f)

    if args.max_samples is not None:
        index_records = index_records[:args.max_samples]

    logger.info(f"Total samples to run: {len(index_records)}")

    total_wall_time = 0.0
    total_trt_infer_time = 0.0
    success = 0
    failed = 0
    skipped = 0

    failed_records = []

    power_monitor = None
    if not args.disable_power_monitor:
        power_monitor = NvidiaSmiPowerMonitor(
            gpu_index=args.gpu_index,
            interval_sec=args.power_interval_sec,
        )
        if power_monitor.start():
            logger.info(
                f"Power monitor started with nvidia-smi: "
                f"gpu_index={args.gpu_index}, interval={args.power_interval_sec:.3f}s"
            )

    try:
        for i, record in enumerate(index_records):
            sample_idx = record.get("sample_idx", i)
            sample_name = record.get("sample_name", f"{sample_idx:06d}")
            sample_dir = resolve_sample_dir(record, args.input_root)

            if not os.path.exists(sample_dir):
                logger.error(f"[{i}] sample_dir not found: {sample_dir}")
                failed += 1
                failed_records.append({
                    "sample_idx": sample_idx,
                    "sample_name": sample_name,
                    "sample_dir": sample_dir,
                    "error": "sample_dir not found"
                })
                continue

            if should_skip(sample_dir, overwrite=args.overwrite):
                logger.info(f"[{i}] Skip existing outputs: {sample_name}")
                skipped += 1
                continue

            try:
                t0 = time.perf_counter()
                outputs, trt_infer_time = runner.infer_one_sample(sample_dir)
                runner.save_outputs(sample_dir, outputs)
                wall_time = time.perf_counter() - t0

                total_wall_time += wall_time
                total_trt_infer_time += trt_infer_time
                success += 1

                if i % 10 == 0:
                    logger.info(
                        f"[{i}/{len(index_records)}] Done: {sample_name} | "
                        f"trt_infer={trt_infer_time:.4f}s | "
                        f"wall={wall_time:.4f}s | "
                        f"avg_trt_infer={(total_trt_infer_time / max(success, 1)):.4f}s | "
                        f"avg_wall={(total_wall_time / max(success, 1)):.4f}s"
                    )

            except Exception as e:
                logger.exception(f"[{i}] Failed on sample {sample_name}: {e}")
                failed += 1
                failed_records.append({
                    "sample_idx": sample_idx,
                    "sample_name": sample_name,
                    "sample_dir": sample_dir,
                    "error": str(e)
                })
    finally:
        power_summary = power_monitor.stop() if power_monitor is not None else {
            "enabled": False,
            "gpu_index": args.gpu_index,
            "sample_count": 0,
            "sampling_interval_sec": args.power_interval_sec,
            "duration_sec": None,
            "avg_power_w": None,
            "min_power_w": None,
            "max_power_w": None,
            "energy_j": None,
        }

    if power_summary.get("energy_j") is not None and success > 0:
        power_summary["energy_per_sample_j"] = power_summary["energy_j"] / success
    else:
        power_summary["energy_per_sample_j"] = None

    summary = {
        "engine_path": args.engine_path,
        "input_root": args.input_root,
        "index_path": args.index_path,
        "max_samples": args.max_samples,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "avg_trt_inference_time_sec": (total_trt_infer_time / success) if success > 0 else None,
        "avg_end_to_end_time_sec": (total_wall_time / success) if success > 0 else None,
        "power": power_summary,
        "failed_records": failed_records,
    }

    summary_path = os.path.join(args.input_root, "trt_batch_infer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"Batch inference finished")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Average TensorRT inference time: {(total_trt_infer_time / success) if success > 0 else 0:.4f}s")
    logger.info(f"Average end-to-end time: {(total_wall_time / success) if success > 0 else 0:.4f}s")
    if power_summary.get("avg_power_w") is not None:
        logger.info(f"Average GPU power: {power_summary['avg_power_w']:.4f} W")
        logger.info(f"Min/Max GPU power: {power_summary['min_power_w']:.4f} / {power_summary['max_power_w']:.4f} W")
        if power_summary.get("energy_j") is not None:
            logger.info(f"Total GPU energy: {power_summary['energy_j']:.4f} J")
        else:
            logger.info("Total GPU energy: unavailable")
        if power_summary.get("energy_per_sample_j") is not None:
            logger.info(f"Energy per successful sample: {power_summary['energy_per_sample_j']:.4f} J")
        else:
            logger.info("Energy per successful sample: unavailable")
        logger.info(f"Power samples: {power_summary['sample_count']}")
    else:
        logger.info("Average GPU power: unavailable")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
