# 中文使用说明

本仓库是论文《Weight-Activation Mixed-Precision Quantization for Collaborative Perception》的开源代码整理版，包含模型训练、验证、伪量化、ONNX 导出、显式 Q/DQ 插入和 TensorRT/NVIDIA Jetson Xavier NX 部署流程。

## 开源边界

- 包含：公开数据集的读取与预处理、模型定义、训练/推理、论文量化实验和部署脚本。
- 不包含：任何原始数据、私有数据、数据集制作/标注/同步/导出脚本、实验日志、TensorRT engine 和临时调试文件。
- `opencood/data_utils` 是模型运行所需的公开数据集加载代码，不属于数据集制作工具。

## 环境

论文服务器使用的 `mapenv` 环境核心版本如下：

- Ubuntu 22.04
- Python 3.7.16
- PyTorch 1.10.1+cu111
- torchvision 0.11.2+cu111
- CUDA Toolkit 11.2（PyTorch 内置 CUDA 11.1）
- spconv 1.2.1
- ONNX/ONNX Runtime 1.14.1

安装：

```bash
conda env create -f environment.yml
conda activate mapenv
pip install -e .
```

## 数据与配置

项目使用的三套公开数据集配置均已提供：

- `configs/pointpillar_where2comm_dair.yaml`：DAIR-V2X-C
- `configs/pointpillar_where2comm_opv2v.yaml`：OPV2V
- `configs/pointpillar_where2comm_v2xsim.yaml`：V2X-Sim 2.0

请自行下载相应数据集。仓库不会下载、生成或修改数据集。使用前修改配置中的数据路径。DAIR-V2X-C 配置需要修改：

- `data_dir`
- `root_dir`
- `validate_dir`
- `test_dir`

OPV2V 配置对应 `train`、`validate`、`test` 三个目录；V2X-Sim 配置对应三个 `v2xsim_infos_*.pkl` 文件，并默认从 `data/V2X-Sim-2.0` 解析点云。

需要说明：当前提供的论文 PDF 在部署与量化实验章节只报告 DAIR-V2X-C；OPV2V 和 V2X-Sim 配置用于项目中的三数据集训练与基础模型实验。

## 运行顺序

1. 按主 README 的 Quick Start 训练模型，或准备你自己的 OpenCOOD checkpoint；本仓库不附带模型权重。
2. 使用 `tools/pseudo_*.py` 复现表 IV–VII 的伪量化消融。
3. 使用 `tools/export_onnx.py` 导出静态 FP32 ONNX。
4. 使用 `tools/verify_onnx.py` 验证 PyTorch 与 ONNX Runtime 一致性。
5. 使用 `tools/export_validation_inputs.py` 导出固定形状验证输入。
6. 使用 `tools/insert_qdq.py` 插入显式 Q/DQ；最终配置量化回归头和方向头，保留分类头、融合模块和压缩模块为高精度。
7. 在 Jetson/TensorRT 环境构建 engine，执行 `tools/run_tensorrt.py`，最后使用 `tools/evaluate_tensorrt.py` 计算 AP。

所有完整命令和参数解释见 `docs/REPRODUCTION.md`。

## 注意

- TensorRT engine 与设备架构和 TensorRT 版本绑定，因此仓库只提供生成脚本，不提供 engine。
- 论文使用两智能体样本（`record_len=2`）和固定最大 voxel 数 26640。
- Xavier NX 的 TensorRT/PyCUDA 通常使用 JetPack 自带环境，不能直接照搬服务器端 conda 环境。
