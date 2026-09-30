#权重激活都非对称量化
import os
import torch
import torch.nn as nn
import torch.quantization as tq
import logging
import argparse
from torch.utils.data import DataLoader

from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 配置区域
# 路径
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "pointpillar_where2comm_dair.yaml")
SAVE_DIR = os.path.join(REPO_ROOT, "outputs", "pseudo_quantization", "w4a4")



# 1. 融合逻辑 (Linear+BN, Conv+BN)

def fuse_modules_custom(module: nn.Module):
    #递归融合 Conv+BN(+ReLU) 和 Linear+BN
    children = list(module.named_children())

    for name, child in children:
        fuse_modules_custom(child)

        if isinstance(child, nn.Sequential):
            submods = list(child.named_children())
            i = 0
            while i < len(submods):
                name_i, m_i = submods[i]

                # Case A: Conv2d + BN
                if isinstance(m_i, nn.Conv2d) and i + 1 < len(submods) and \
                        isinstance(submods[i + 1][1], nn.BatchNorm2d):

                    fuse_list = [name_i, submods[i + 1][0]]
                    if i + 2 < len(submods) and isinstance(submods[i + 2][1], (nn.ReLU, nn.ReLU6)):
                        fuse_list.append(submods[i + 2][0])
                        # logger.info(f"[Fuse] {name}: Conv2d + BN + ReLU -> {fuse_list}")
                        tq.fuse_modules(child, fuse_list, inplace=True)
                        i += 3
                    else:
                        # logger.info(f"[Fuse] {name}: Conv2d + BN -> {fuse_list}")
                        tq.fuse_modules(child, fuse_list, inplace=True)
                        i += 2
                    continue

                # Case B: Linear + BN1d
                elif isinstance(m_i, nn.Linear) and i + 1 < len(submods) and \
                        isinstance(submods[i + 1][1], nn.BatchNorm1d):

                    fuse_list = [name_i, submods[i + 1][0]]
                    # logger.info(f"[Fuse] {name}: Linear + BN1d -> {fuse_list}")
                    tq.fuse_modules(child, fuse_list, inplace=True)
                    i += 2
                    continue
                i += 1


def check_bn_layers(model: nn.Module):
    has_bn = False
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            logger.warning(f"[Warning] Found unfused BN layer: {name}")
            has_bn = True
    if not has_bn:
        logger.info("[Check] No active BN layers found. Fusion successful.")



# 2. 权重与 Bias 量化 (Per-Channel / Per-Tensor)

# def fake_quantize_weights_and_bias(model: nn.Module):
#     """对权重和Bias进行伪量化"""
#     logger.info("===> Applying Fake Quantization to Weights and Biases...")
#
#     for name, param in model.named_parameters():
#         # param.dim() >= 2 -> Weight (Per-Channel)
#         if param.dim() >= 2:
#             num_channels = param.shape[0]  # Assume output channel is dim 0
#             param_flat = param.reshape(num_channels, -1)
#
#             # Scale calculation
#             min_vals = param_flat.min(dim=1)[0]
#             max_vals = param_flat.max(dim=1)[0]
#             abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
#             scales = abs_max / 127.0
#             scales = torch.clamp(scales, min=1e-8)
#
#             # Fake Quantize
#             quantized = torch.round(param_flat / scales.unsqueeze(1))
#             quantized = torch.clamp(quantized, -128, 127)
#             dequantized = quantized * scales.unsqueeze(1)
#
#             param.data.copy_(dequantized.reshape(param.shape))
#
#         # param.dim() < 2 -> Bias (Per-Tensor)
#         else:
#             # min_val = param.min()
#             # max_val = param.max()
#             # abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
#             # scale = abs_max / 127.0
#             # scale = torch.clamp(scale, min=1e-8)
#             #
#             # quantized = torch.round(param / scale)
#             # quantized = torch.clamp(quantized, -128, 127)
#             # dequantized = quantized * scale
#             #
#             # param.data.copy_(dequantized)
#             continue

def _asym_qrange(bits: int):
    assert bits in (4, 8)
    qmax = (1 << (bits - 1)) - 1   # 4->7, 8->127
    qmin = -(1 << (bits - 1))      # 4->-8, 8->-128
    return float(qmin), float(qmax)

def fake_quantize_weights_and_bias(model: nn.Module, w_bits: int = 8):
    qmin, qmax = _asym_qrange(w_bits)
    qrange = qmax - qmin

    for module_name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                          nn.Linear)):
            if m.weight is None:
                continue

            w = m.weight.data

            # per-channel 维度：ConvTranspose(out_channels) 在 dim=1，其它默认 dim=0
            if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                ch_axis = 1
            else:
                ch_axis = 0

            w_perm = w.transpose(0, ch_axis).contiguous() if ch_axis != 0 else w
            w_flat = w_perm.reshape(w_perm.shape[0], -1)

            x_min = w_flat.min(dim=1).values
            x_max = w_flat.max(dim=1).values
            scales = ((x_max - x_min) / qrange).clamp(min=1e-8)
            zero_points = torch.round(qmin - x_min / scales).clamp(qmin, qmax)

            q = torch.round(w_flat / scales.unsqueeze(1) + zero_points.unsqueeze(1)).clamp(qmin, qmax)
            w_qdq_perm = ((q - zero_points.unsqueeze(1)) * scales.unsqueeze(1)).reshape_as(w_perm)

            w_qdq = w_qdq_perm.transpose(0, ch_axis).contiguous() if ch_axis != 0 else w_qdq_perm
            m.weight.data.copy_(w_qdq)



# 3. 激活量化
# 注册 forward hook
# 跑若干 calibration batch
# 收集激活样本（降采样）
# 拼接成经验分布
# 用 quantile 估计阈值
# 固定 scale，用于推理伪量化

class ActivationQuantizer:
    def __init__(self, method='percentile', percentile=100, ema_decay=0.9, a_bits: int = 8):
        self.method = method
        self.percentile = percentile
        self.ema_decay = ema_decay
        self.a_bits = a_bits
        self.layer_stats = {}
        self.layer_scales = {}
        self.layer_zero_points = {}
        self.is_calibrating = False
        self.sensitive_patterns = [
            'cls_head', 'reg_head', 'dir_head', 'pillar_scatter',
            'fusion_net', 'naive_compressor', 'shrink_conv',
        ]

    def _qrange(self):
        assert self.a_bits in (4, 8)
        qmax = (1 << (self.a_bits - 1)) - 1  # 4->7, 8->127
        qmin = -(1 << (self.a_bits - 1))     # 4->-8, 8->-128
        return float(qmin), float(qmax)


    def is_sensitive(self, name):
        return any(p in name for p in self.sensitive_patterns)

    def start_calibration(self):
        self.is_calibrating = True
        self.layer_stats = {}
        self.layer_zero_points = {}
        logger.info(f"Start Calibration ({self.method})...")

    def update_stats(self, name, tensor):
        # 简单降采样防止内存溢出
        if name not in self.layer_stats: self.layer_stats[name] = []
        if len(self.layer_stats[name]) < 500:
            flat = tensor.detach().flatten()
            # 随机采样 2000 个点
            if flat.numel() > 2000:
                indices = torch.randperm(flat.numel())[:2000]
                flat = flat[indices]
            self.layer_stats[name].append(flat.cpu())

    def finish_calibration(self):
        self.is_calibrating = False
        logger.info("Computing calibration scales...")

        qmin, qmax = self._qrange()
        qrange = qmax - qmin

        for name, stats in self.layer_stats.items():
            if not stats:
                continue
            all_vals = torch.cat(stats)

            if self.method == 'percentile':
                q_high = self.percentile / 100.0
                q_low = 1.0 - q_high
                x_min = torch.quantile(all_vals, q_low).item()
                x_max = torch.quantile(all_vals, q_high).item()
            else:
                x_min = all_vals.min().item()
                x_max = all_vals.max().item()

            if x_max - x_min < 1e-8:
                scale = 1e-8
                zero_point = 0.0
            else:
                scale = max((x_max - x_min) / (qrange), 1e-8)
                zero_point = round(qmin - x_min / scale)
                zero_point = float(max(qmin, min(qmax, zero_point)))

            self.layer_scales[name] = scale
            self.layer_zero_points[name] = zero_point

        logger.info(f"Calibration done. Scales computed for {len(self.layer_scales)} layers.")

    def quantize(self, name, tensor):
        if self.is_sensitive(name):
            return tensor

        if self.is_calibrating:
            self.update_stats(name, tensor)
            return tensor

        if name in self.layer_scales:
            scale = self.layer_scales[name]
            qmin, qmax = self._qrange()
            zero_point = self.layer_zero_points.get(name, 0.0)
            q = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
            return (q - zero_point) * scale
        else:
            return tensor


def register_activation_hooks(model, quantizer):
    hooks = []
    # 目标层：Conv, Linear, ConvTranspose
    target_types = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)

    def get_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                return quantizer.quantize(name, out)
            return out

        return hook

    for name, m in model.named_modules():
        if isinstance(m, target_types):
            h = m.register_forward_hook(get_hook(name))
            hooks.append(h)
    return hooks


# ================================================
# 4. 主程序
# ================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Input FP32 checkpoint')
    parser.add_argument('--config', default=CONFIG_PATH, help='OpenCOOD YAML configuration')
    parser.add_argument('--output_dir', default=SAVE_DIR, help='Output directory')
    parser.add_argument('--calibrate', action='store_true', default=True, help='Perform activation calibration')
    parser.add_argument('--calib_batches', type=int, default=40, help='Number of batches for calibration')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 构建模型
    config = yaml_utils.load_yaml(args.config)
    logger.info("Building Model...")
    model = train_utils.create_model(config)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 2. 加载权重
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # 3. 执行融合
    logger.info("Fusing modules...")

    model.cpu()
    fuse_modules_custom(model)
    check_bn_layers(model)
    model.to(device)  # 融合完放回设备

    # 4. 权重量化 (Fake Quant)
    # 直接修改 model 的权重参数
    # fake_quantize_weights_and_bias(model)
    fake_quantize_weights_and_bias(model, w_bits=4)
    #fake_quantize_weights_and_bias(model, w_bits=8)

    # 5. 激活量化 + 校准
    quantizer = ActivationQuantizer(method='percentile', percentile=99.9, a_bits=4)
    hooks = register_activation_hooks(model, quantizer)

    if args.calibrate:
        logger.info("===> Building calibration dataloader...")
        # 1. 构建数据集
        calib_dataset = build_dataset(config, visualize=False, train=False)
        collate = getattr(calib_dataset, 'collate_batch_test', None)
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate
        )

        # 2. 开启校准模式
        quantizer.start_calibration()

        # 3. 运行校准循环
        fusion_method = config.get('fusion', {}).get('core_method', 'intermediate')
        logger.info(f"Running calibration with method: {fusion_method} on {device}...")

        count = 0
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if batch is None: continue
                if count >= args.calib_batches: break

                # 将数据移动到设备 (GPU/CPU)
                batch_device = train_utils.to_device(batch, device)

                # 自动调用 model(data)，触发注册的 Hooks
                try:
                    if fusion_method == 'intermediate':
                        inference_utils.inference_intermediate_fusion(batch_device, model, calib_dataset)
                    elif fusion_method == 'late':
                        inference_utils.inference_late_fusion(batch_device, model, calib_dataset)
                    elif fusion_method == 'early':
                        inference_utils.inference_early_fusion(batch_device, model, calib_dataset)
                    else:

                        inference_utils.inference_no_fusion(batch_device, model, calib_dataset)
                except Exception as e:
                    logger.warning(f"Batch {i} failed: {e}. Skipping...")
                    continue

                count += 1
                if count % 10 == 0:
                    logger.info(f"Calibrated {count}/{args.calib_batches} batches...")

        # 4. 结束校准，计算 Scale
        quantizer.finish_calibration()

        # 5. 保存校准得到的 Scale
        torch.save(
            {"scales": quantizer.layer_scales, "zero_points": quantizer.layer_zero_points},
            os.path.join(args.output_dir, "activation_scales.pth")
        )

    # 6. 移除 Hook (保存模型不需要 Hook)
    for h in hooks:
        h.remove()

    # 7. 保存最终模型

    save_path = os.path.join(args.output_dir, "fused_quantized_model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    logger.info(f"Activation scales saved to {os.path.join(args.output_dir, 'activation_scales.pth')}")


if __name__ == "__main__":
    main()
