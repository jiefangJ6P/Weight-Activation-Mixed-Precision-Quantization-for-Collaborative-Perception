# Weight-Activation Mixed-Precision Quantization for Collaborative Perception

Official research code for **Weight-Activation Mixed-Precision Quantization for Collaborative Perception**.

This repository contains the training, evaluation, pseudo-quantization, ONNX export, explicit Q/DQ insertion, and TensorRT deployment code used in the paper. It is built on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [CoAlign](https://github.com/yifanlu0227/CoAlign), and the spatial-confidence collaboration model from [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm).

> Dataset-construction utilities and all private/raw datasets are intentionally excluded. The included `opencood/data_utils` code only loads and preprocesses public datasets at runtime.

## Highlights

- Static collaborative graph reformulation for fixed-shape deployment.
- Pseudo-quantization experiments for bias, mapping, calibration, and bit-width selection.
- Explicit ONNX Q/DQ insertion with module- and head-wise precision control.
- PyTorch, ONNX Runtime, and TensorRT consistency/evaluation tools.
- Reproduced `mapenv` software environment and documented checkpoint interfaces for training and deployment.

## Repository Layout

```text
.
├── configs/                     # DAIR-V2X-C, OPV2V, and V2X-Sim configs
├── docs/                        # Detailed reproduction and Chinese guide
├── opencood/                    # Training/inference framework
├── tools/                       # Quantization and deployment pipeline
├── environment.yml             # Conda environment matching mapenv
├── requirements.txt            # Python-only dependency list
└── setup.py
```

## Installation

The reference server environment is Ubuntu 22.04, Python 3.7.16, PyTorch 1.10.1+cu111, CUDA 11.1/11.2, and an RTX 4090. TensorRT deployment was evaluated separately on an NVIDIA Jetson Xavier NX (8 GB).

```bash
conda env create -f environment.yml
conda activate mapenv
pip install -e .
```

If `spconv==1.2.1` or the CUDA extensions do not build on your system, follow the source-build notes in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Data

The training code supports the three datasets used in the project: **DAIR-V2X-C**, **OPV2V**, and **V2X-Sim 2.0**. Download them from their official sources and arrange them as expected by OpenCOOD/CoAlign. This release does not redistribute any dataset or dataset-generation code.

The exact logged experiment configurations are provided with sanitized paths:

- `configs/pointpillar_where2comm_dair.yaml`
- `configs/pointpillar_where2comm_opv2v.yaml`
- `configs/pointpillar_where2comm_v2xsim.yaml`

Update each configuration's dataset paths before training or evaluation. The quantization and TensorRT results reported in the supplied paper PDF use DAIR-V2X-C; the other two configurations reproduce the project's three-dataset training setup.

## Quick Start

Train a DAIR-V2X-C model:

```bash
python opencood/tools/train.py \
  --hypes_yaml configs/pointpillar_where2comm_dair.yaml \
  --fusion_method intermediate
```

To evaluate a trained model, pass its OpenCOOD output directory to `opencood/tools/inference.py --model_dir /path/to/model_dir`. Model weights are not distributed in this repository.

Run the final W8A8 symmetric percentile pseudo-quantization experiment:

```bash
python tools/pseudo_quantize.py \
  --checkpoint /path/to/net_epoch_bestval.pth \
  --calib_batches 40
```

Export the static FP32 ONNX model:

```bash
python tools/export_onnx.py \
  --config configs/pointpillar_where2comm_dair.yaml \
  --checkpoint /path/to/net_epoch_bestval.pth \
  --output outputs/where2comm_fp32.onnx
```

The complete Q/DQ and TensorRT workflow is documented in [docs/REPRODUCTION.md](docs/REPRODUCTION.md). A Chinese guide is available in [docs/README_zh-CN.md](docs/README_zh-CN.md).

## Paper Results

| Engine | AP@0.3 | AP@0.5 | AP@0.7 | Latency (ms) |
|---|---:|---:|---:|---:|
| TensorRT FP32 | 0.8085 | 0.7756 | 0.6158 | 637.7 |
| TensorRT FP16 | 0.8082 | 0.7753 | 0.6155 | 389.1 |
| Calibration INT8 | 0.5450 | 0.5244 | 0.3646 | 179.9 |
| Mixed-precision Q/DQ INT8 | 0.7501 | 0.7217 | 0.5655 | 272.1 |

Latency was measured on the Jetson Xavier NX with batch size 1 and two agents. TensorRT engines are hardware- and version-specific, so generated engines are not distributed.

## License

This code retains the upstream Academic Software License. It is intended for academic and nonprofit research use; see [LICENSE](LICENSE) for the full terms. Third-party components remain subject to their original licenses.

## Acknowledgements

This project builds on OpenCOOD, CoAlign, and Where2comm. We thank their authors for releasing their code.
