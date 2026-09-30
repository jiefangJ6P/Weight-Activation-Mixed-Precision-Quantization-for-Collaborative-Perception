import argparse
import importlib.util
import inspect
import logging
import os

import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_pseudo_quantized")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pseudo-quantized checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="fused_quantized_model.pth")
    parser.add_argument("--scales", required=True, help="activation_scales.pth")
    parser.add_argument("--quantizer-module", required=True, help="Pseudo-quantization script used to create the files")
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/pseudo_evaluation")
    parser.add_argument("--record-len", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_quantizer_module(path):
    spec = importlib.util.spec_from_file_location("pseudo_quantizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import quantizer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_quantizer(module, activation_bits):
    parameters = inspect.signature(module.ActivationQuantizer).parameters
    kwargs = {"method": "percentile"}
    if "a_bits" in parameters:
        kwargs["a_bits"] = activation_bits
    return module.ActivationQuantizer(**kwargs)


def inference(batch, model, dataset, fusion_method):
    if fusion_method == "intermediate":
        return inference_utils.inference_intermediate_fusion(batch, model, dataset)
    if fusion_method == "late":
        return inference_utils.inference_late_fusion(batch, model, dataset)
    if fusion_method == "early":
        return inference_utils.inference_early_fusion(batch, model, dataset)
    return inference_utils.inference_no_fusion(batch, model, dataset)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypes = yaml_utils.load_yaml(args.config)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
    )

    quantizer_module = load_quantizer_module(args.quantizer_module)
    model = train_utils.create_model(hypes).cpu().eval()
    quantizer_module.fuse_modules_custom(model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model.to(device)

    activation_params = torch.load(args.scales, map_location="cpu")
    quantizer = make_quantizer(quantizer_module, args.activation_bits)
    if isinstance(activation_params, dict) and "scales" in activation_params:
        quantizer.layer_scales = activation_params["scales"]
        quantizer.layer_zero_points = activation_params.get("zero_points", {})
    else:
        quantizer.layer_scales = activation_params
    quantizer.is_calibrating = False
    hooks = quantizer_module.register_activation_hooks(model, quantizer)

    result_stat = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    fusion_method = hypes.get("fusion", {}).get("core_method", "intermediate")
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if args.record_len is not None:
                current = int(batch["ego"]["record_len"].reshape(-1)[0])
                if current != args.record_len:
                    continue
            if args.max_samples is not None and processed >= args.max_samples:
                break
            batch = train_utils.to_device(batch, device)
            result = inference(batch, model, dataset, fusion_method)
            for threshold in result_stat:
                eval_utils.caluclate_tp_fp(
                    result["pred_box_tensor"],
                    result["pred_score"],
                    result["gt_box_tensor"],
                    result_stat,
                    threshold,
                )
            processed += 1
            if processed == 1 or processed % 50 == 0:
                logger.info("Processed %d samples", processed)

    for hook in hooks:
        hook.remove()
    os.makedirs(args.output_dir, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, args.output_dir, fusion_method)
    logger.info("AP@0.3=%.4f AP@0.5=%.4f AP@0.7=%.4f", ap30, ap50, ap70)


if __name__ == "__main__":
    main()
