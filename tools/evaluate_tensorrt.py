import os
import json
import argparse
import logging

import numpy as np
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("eval_trt_where2comm")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TensorRT where2comm outputs with OpenCOOD post_process")
    parser.add_argument("--config_path", type=str, required=True, help="OpenCOOD config.yaml path")
    parser.add_argument("--input_root", type=str, required=True, help="TensorRT output root")
    parser.add_argument("--index_path", type=str, required=True, help="index.json path")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save eval yaml/json")
    parser.add_argument("--start_index", type=int, default=0, help="Start position in index.json, inclusive")
    parser.add_argument("--end_index", type=int, default=None, help="End position in index.json, inclusive")
    parser.add_argument("--max_samples", type=int, default=None, help="Only evaluate first N samples from index.json")
    parser.add_argument("--data_dir", type=str, default=None, help="Override hypes['data_dir']")
    parser.add_argument("--root_dir", type=str, default=None, help="Override hypes['root_dir']")
    parser.add_argument("--validate_dir", type=str, default=None, help="Override hypes['validate_dir']")
    parser.add_argument("--test_dir", type=str, default=None, help="Override hypes['test_dir']")
    return parser.parse_args()


def pick_collate_fn(dataset):
    if hasattr(dataset, "collate_batch_test") and callable(getattr(dataset, "collate_batch_test")):
        return dataset.collate_batch_test
    raise RuntimeError("dataset 没有 collate_batch_test，请检查 dataset 版本")


def prepare_hypes(args):
    hypes = yaml_utils.load_yaml(args.config_path, None)

    if args.data_dir is not None:
        hypes["data_dir"] = args.data_dir
    if args.root_dir is not None:
        hypes["root_dir"] = args.root_dir
    if args.validate_dir is not None:
        hypes["validate_dir"] = args.validate_dir
    if args.test_dir is not None:
        hypes["test_dir"] = args.test_dir

    return hypes


def to_torch(arr, dtype):
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    return tensor.to(dtype=dtype)


def resolve_sample_dir(record, input_root):
    if "relative_dir" in record and isinstance(record["relative_dir"], str):
        return os.path.join(input_root, record["relative_dir"])
    if "sample_name" in record and isinstance(record["sample_name"], str):
        return os.path.join(input_root, record["sample_name"])
    if "sample_dir" in record and isinstance(record["sample_dir"], str):
        return record["sample_dir"]
    raise KeyError(f"Cannot resolve sample directory from record: {record}")


def load_trt_outputs(sample_dir):
    required = ["cls_preds.npy", "reg_preds.npy", "dir_preds.npy", "comm_rate.npy"]
    for name in required:
        path = os.path.join(sample_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing TensorRT output file: {path}")

    cls_preds = np.load(os.path.join(sample_dir, "cls_preds.npy")).astype(np.float32, copy=False)
    reg_preds = np.load(os.path.join(sample_dir, "reg_preds.npy")).astype(np.float32, copy=False)
    dir_preds = np.load(os.path.join(sample_dir, "dir_preds.npy")).astype(np.float32, copy=False)
    comm_rate = np.load(os.path.join(sample_dir, "comm_rate.npy")).astype(np.float32, copy=False)

    output_dict = {
        "ego": {
            "cls_preds": to_torch(cls_preds, torch.float32),
            "reg_preds": to_torch(reg_preds, torch.float32),
            "dir_preds": to_torch(dir_preds, torch.float32),
            "comm_rate": to_torch(comm_rate, torch.float32),
        }
    }
    return output_dict


def build_batch_from_dataset(dataset, sample_idx):
    sample = dataset[sample_idx]
    batch = pick_collate_fn(dataset)([sample])
    return batch


def main():
    args = parse_args()

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"index.json not found: {args.index_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Preparing hypes...")
    hypes = prepare_hypes(args)

    logger.info("Building dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)

    with open(args.index_path, "r", encoding="utf-8") as f:
        index_records = json.load(f)

    start_index = max(args.start_index, 0)
    end_index = args.end_index

    if end_index is not None:
        index_records = index_records[start_index:end_index + 1]
    else:
        index_records = index_records[start_index:]

    if args.max_samples is not None:
        index_records = index_records[:args.max_samples]

    logger.info(f"Total samples to evaluate: {len(index_records)}")

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    comm_rates = []
    failed_records = []

    for i, record in enumerate(index_records):
        sample_idx = int(record.get("sample_idx", i))
        sample_name = record.get("sample_name", f"{sample_idx:06d}")
        sample_dir = resolve_sample_dir(record, args.input_root)

        try:
            batch = build_batch_from_dataset(dataset, sample_idx)
            output_dict = load_trt_outputs(sample_dir)

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch, output_dict)

            for iou_thresh in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    iou_thresh,
                )

            comm_rate_tensor = output_dict["ego"]["comm_rate"]
            comm_rate_value = float(comm_rate_tensor.reshape(-1)[0].item()) if comm_rate_tensor.numel() > 0 else 0.0
            comm_rates.append(comm_rate_value)

            if i % 10 == 0:
                logger.info(
                    f"[{i}/{len(index_records)}] Done: {sample_name} | "
                    f"pred={0 if pred_box_tensor is None else pred_box_tensor.shape[0]} | "
                    f"gt={gt_box_tensor.shape[0]} | comm_rate={comm_rate_value:.6f}"
                )

        except Exception as e:
            logger.exception(f"[{i}] Failed on sample {sample_name}: {e}")
            failed_records.append(
                {
                    "sample_idx": sample_idx,
                    "sample_name": sample_name,
                    "sample_dir": sample_dir,
                    "error": str(e),
                }
            )

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, args.save_dir, "trt_where2comm")
    avg_comm_rate = float(sum(comm_rates) / len(comm_rates)) if comm_rates else 0.0

    summary = {
        "num_samples": len(index_records),
        "num_success": len(index_records) - len(failed_records),
        "num_failed": len(failed_records),
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "comm_rate": avg_comm_rate,
        "failed_records": failed_records,
    }

    summary_path = os.path.join(args.save_dir, "trt_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"AP@0.3: {ap30:.4f}")
    logger.info(f"AP@0.5: {ap50:.4f}")
    logger.info(f"AP@0.7: {ap70:.4f}")
    logger.info(f"Comm Rate: {avg_comm_rate:.6f}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
