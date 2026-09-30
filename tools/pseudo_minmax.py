# 权重激活都对称量化，激活校准使用 min/max
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
SAVE_DIR = os.path.join(REPO_ROOT, "outputs", "pseudo_quantization", "w8a8_minmax")



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

def fake_quantize_weights_and_bias(model: nn.Module):
    qmin, qmax = -128, 127

    for module_name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                          nn.Linear)):
            if m.weight is None:
                continue

            w = m.weight.data

            # 选择 per-channel 维度：ConvTranspose2d/3d 通常希望按 out_channels（dim=1）
            if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # w shape: (in_channels, out_channels/groups, ...)
                ch_axis = 1
            else:
                # ConvNd / Linear: 通常按输出通道/输出特征（dim=0）
                ch_axis = 0

            # 把 per-channel 维度移到最前面，再 flatten
            w_perm = w.transpose(0, ch_axis).contiguous() if ch_axis != 0 else w
            w_flat = w_perm.reshape(w_perm.shape[0], -1)

            # per-channel absmax（对称）
            x_min = w_flat.min(dim=1).values
            x_max = w_flat.max(dim=1).values
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            scales = (abs_max / float(qmax)).clamp(min=1e-8)

            q = torch.round(w_flat / scales.unsqueeze(1)).clamp(qmin, qmax)
            w_qdq_perm = (q * scales.unsqueeze(1)).reshape_as(w_perm)

            # 还原维度
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
    def __init__(self, method='percentile', percentile=99.99, ema_decay=0.9):
        self.method = method
        self.percentile = percentile
        self.ema_decay = ema_decay
        self.layer_stats = {}
        self.layer_scales = {}
        self.layer_zero_points = {}
        self.is_calibrating = False

        # 新增：minmax 用
        self.layer_mins = {}   # name -> global min
        self.layer_maxs = {}   # name -> global max


        # 敏感层列表，跳过量化
        self.sensitive_patterns = [
            'cls_head', 'reg_head', 'dir_head', 'pillar_scatter',
            'fusion_net', 'naive_compressor', 'shrink_conv',
        ]

    def is_sensitive(self, name):
        return any(p in name for p in self.sensitive_patterns)

    def start_calibration(self):
        self.is_calibrating = True
        self.layer_stats = {}
        self.layer_zero_points = {}
        self.layer_mins = {}
        self.layer_maxs = {}
        logger.info(f"Start Calibration ({self.method})...")

    def update_stats(self, name, tensor):
        # ===== Min/Max 模式：跨 batch 聚合全局 min/max =====
        if self.method == 'minmax':
            batch_min = tensor.detach().amin().item()
            batch_max = tensor.detach().amax().item()
            prev_min = self.layer_mins.get(name, batch_min)
            prev_max = self.layer_maxs.get(name, batch_max)
            self.layer_mins[name] = min(prev_min, batch_min)
            self.layer_maxs[name] = max(prev_max, batch_max)
            return

        # ===== 保留你原来的 percentile 采样逻辑 =====
        if name not in self.layer_stats:
            self.layer_stats[name] = []
        if len(self.layer_stats[name]) < 500:
            flat = tensor.detach().flatten()
            if flat.numel() > 2000:
                indices = torch.randperm(flat.numel())[:2000]
                flat = flat[indices]
            self.layer_stats[name].append(flat.cpu())

    def finish_calibration(self):
        self.is_calibrating = False
        logger.info("Computing calibration scales...")

        qmin, qmax = -128, 127

        if self.method == 'minmax':
            for name in self.layer_mins.keys():
                x_min = self.layer_mins[name]
                x_max = self.layer_maxs[name]

                abs_max = max(abs(x_min), abs(x_max))
                scale = max(abs_max / float(qmax), 1e-8)

                self.layer_scales[name] = scale
                self.layer_zero_points[name] = 0.0
            logger.info(f"Calibration done. Scales computed for {len(self.layer_scales)} layers.")
            return

        # ===== 下面保留 percentile / fallback max 的统计入口，但使用对称量化 scale =====
        for name, stats in self.layer_stats.items():
            if not stats:
                continue
            all_vals = torch.cat(stats)
            if self.method == 'percentile':
                abs_max = torch.quantile(all_vals.abs(), self.percentile / 100.0).item()
            else:
                x_min = all_vals.min().item()
                x_max = all_vals.max().item()
                abs_max = max(abs(x_min), abs(x_max))

            scale = max(abs_max / float(qmax), 1e-8)

            self.layer_scales[name] = scale
            self.layer_zero_points[name] = 0.0

        logger.info(f"Calibration done. Scales computed for {len(self.layer_scales)} layers.")

    def quantize(self, name, tensor):
        if self.is_sensitive(name): return tensor

        if self.is_calibrating:
            self.update_stats(name, tensor)
            return tensor

        if name in self.layer_scales:
            scale = self.layer_scales[name]
            q = torch.round(tensor / scale).clamp(-128, 127)
            return q * scale
        else:
            return tensor


def register_activation_hooks(model, quantizer):
    hooks = []
    # 目标层：Conv, Linear, ConvTranspose (ReLU通常跟在Conv后被融合，或者单独存在)
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

    # 这里的设备选择要小心。
    # 通常 opencood 的 inference_utils 需要 GPU 来做 NMS 或 pillar generation
    # 如果你的服务器有 GPU，建议在这里放到 cuda
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
    # 注意：融合通常在 CPU 上做比较安全，但 PyTorch 新版本也支持 GPU 融合
    # 如果报错，可以先 model.cpu() -> fuse -> model.to(device)
    model.cpu()
    fuse_modules_custom(model)
    check_bn_layers(model)
    model.to(device)  # 融合完放回设备

    # 4. 权重量化 (Fake Quant)
    # 这步会直接修改 model 的权重参数
    fake_quantize_weights_and_bias(model)

    # 5. 激活量化 + 校准 (使用你提供的代码逻辑)
    # quantizer = ActivationQuantizer(method='percentile', percentile=99.9)
    quantizer = ActivationQuantizer(method='minmax')
    # 注册 Hook：此时 Hook 里的 self.is_calibrating = False，暂时不起作用
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
            num_workers=0,  # 校准时设为0调试更安全
            collate_fn=collate
        )

        # 2. 开启校准模式
        quantizer.start_calibration()

        # 3. 运行校准循环 (集成你的逻辑)
        fusion_method = config.get('fusion', {}).get('core_method', 'intermediate')
        logger.info(f"Running calibration with method: {fusion_method} on {device}...")

        count = 0
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if batch is None: continue
                if count >= args.calib_batches: break

                # 将数据移动到设备 (GPU/CPU)
                batch_device = train_utils.to_device(batch, device)

                # 【关键】调用 inference_utils 触发完整前向传播
                # 这会自动调用 model(data)，从而触发我们注册的 Hooks
                try:
                    if fusion_method == 'intermediate':
                        inference_utils.inference_intermediate_fusion(batch_device, model, calib_dataset)
                    elif fusion_method == 'late':
                        inference_utils.inference_late_fusion(batch_device, model, calib_dataset)
                    elif fusion_method == 'early':
                        inference_utils.inference_early_fusion(batch_device, model, calib_dataset)
                    else:
                        # 默认直接过模型 (针对单车场景)
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
        torch.save(quantizer.layer_scales, os.path.join(args.output_dir, "activation_scales.pth"))

    # 6. 移除 Hook (保存模型不需要 Hook)
    for h in hooks:
        h.remove()

    # 7. 保存最终模型
    # 此时模型里：
    # 1. 结构已经融合 (Conv+BN -> Conv)
    # 2. 权重已经 Fake Quantized (值变成了离散点)
    # 3. 激活 Scale 已经保存到单独文件，推理时加载
    save_path = os.path.join(args.output_dir, "fused_quantized_model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    logger.info(f"Activation scales saved to {os.path.join(args.output_dir, 'activation_scales.pth')}")


if __name__ == "__main__":
    main()
