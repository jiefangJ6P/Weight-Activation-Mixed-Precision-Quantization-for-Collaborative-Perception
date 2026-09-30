# Reproduction Guide

Run all server-side commands from the repository root.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate mapenv
pip install -e .
```

The historical environment used GCC/G++ 7 and `spconv==1.2.1`. If the PyPI package is unavailable for your platform, build spconv 1.2.1 against the active CUDA toolkit. Then build the bundled CUDA extensions:

```bash
python opencood/pcdet_utils/setup.py build_ext --inplace
python opencood/utils/setup.py build_ext --inplace
```

The environment file targets reproduction of the original Python 3.7 stack. Newer CUDA/PyTorch combinations may require code changes and are not claimed to be numerically identical.

## 2. Dataset

The repository includes the exact logged configurations for all three project datasets:

| Dataset | Configuration | Expected local layout |
|---|---|---|
| DAIR-V2X-C | `configs/pointpillar_where2comm_dair.yaml` | `data/DAIR-V2X-C/cooperative-vehicle-infrastructure` |
| OPV2V | `configs/pointpillar_where2comm_opv2v.yaml` | `data/OPV2V/{train,validate,test}` |
| V2X-Sim 2.0 | `configs/pointpillar_where2comm_v2xsim.yaml` | `data/V2X-Sim-2.0` plus `data/v2xsim2_info` |

Download each dataset from its official source. For DAIR-V2X-C, use the complemented annotations expected by OpenCOOD/CoAlign. Edit these keys in `configs/pointpillar_where2comm_dair.yaml`:

```yaml
data_dir: /path/to/cooperative-vehicle-infrastructure
root_dir: /path/to/cooperative-vehicle-infrastructure/train.json
validate_dir: /path/to/cooperative-vehicle-infrastructure/val.json
test_dir: /path/to/cooperative-vehicle-infrastructure/val.json
```

No dataset-generation step is part of this repository.

Train the other two dataset models with the corresponding logged configuration:

```bash
python opencood/tools/train.py \
  --hypes_yaml configs/pointpillar_where2comm_opv2v.yaml \
  --fusion_method intermediate

python opencood/tools/train.py \
  --hypes_yaml configs/pointpillar_where2comm_v2xsim.yaml \
  --fusion_method intermediate
```

The supplied paper PDF reports its quantization/deployment tables on DAIR-V2X-C. OPV2V and V2X-Sim are retained here because they are part of the project's three-dataset training setup.

## 3. Training and PyTorch Evaluation

Train from scratch:

```bash
python opencood/tools/train.py \
  --hypes_yaml configs/pointpillar_where2comm_dair.yaml \
  --fusion_method intermediate
```

Evaluate a trained checkpoint. `model_dir` must contain the experiment `config.yaml` and an OpenCOOD checkpoint such as `net_epoch_bestval_at11.pth`:

```bash
python opencood/tools/inference.py \
  --model_dir /path/to/model_dir \
  --fusion_method intermediate \
  --epoch 11 \
  --filter_record_len 2
```

## 4. Pseudo-Quantization Ablations

All scripts use the DAIR configuration by default, require `--checkpoint`, and write under `outputs/pseudo_quantization/`.

```bash
# Final global rule: W8A8, symmetric weights/activations, no bias quantization,
# percentile activation calibration.
python tools/pseudo_quantize.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40

# Activation calibration ablations.
python tools/pseudo_minmax.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40
python tools/pseudo_ema.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40

# Bit-width ablations.
python tools/pseudo_w4a8.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40
python tools/pseudo_w8a4.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40
python tools/pseudo_w4a4.py --checkpoint /path/to/net_epoch_bestval.pth --calib_batches 40
```

The scripts save fake-quantized weights and activation scales. They evaluate the quantization rule in PyTorch; they do not create integer TensorRT engines.

Evaluate any generated pseudo-quantized model with its matching quantizer module. For example:

```bash
python tools/evaluate_pseudo_quantized.py \
  --config configs/pointpillar_where2comm_dair.yaml \
  --checkpoint outputs/pseudo_quantization/w8a8_percentile/fused_quantized_model.pth \
  --scales outputs/pseudo_quantization/w8a8_percentile/activation_scales.pth \
  --quantizer-module tools/pseudo_quantize.py \
  --record-len 2
```

## 5. Static ONNX Export and Verification

The deployment graph uses five inputs and four outputs:

| Name | Shape/dtype used in the paper |
|---|---|
| `voxel_features` | `26640 x 32 x 4`, float32 |
| `voxel_coords` | `26640 x 4`, integer |
| `voxel_num_points` | `26640`, integer |
| `record_len` | `2`, integer |
| `pairwise_t_matrix` | `1 x 5 x 5 x 4 x 4`, float32 |

```bash
python tools/export_onnx.py \
  --config configs/pointpillar_where2comm_dair.yaml \
  --checkpoint /path/to/net_epoch_bestval.pth \
  --output outputs/where2comm_fp32.onnx \
  --sample-index 21

python tools/verify_onnx.py \
  --config configs/pointpillar_where2comm_dair.yaml \
  --checkpoint /path/to/net_epoch_bestval.pth \
  --onnx outputs/where2comm_fp32.onnx \
  --num-samples 21 \
  --export-sample-index 21
```

## 6. Export Fixed-Shape Validation Inputs

The paper evaluates only the 1618 two-agent samples from the 1789-sample validation split.

```bash
python tools/export_validation_inputs.py \
  --config configs/pointpillar_where2comm_dair.yaml \
  --output-root outputs/trt_inputs \
  --runtime-root /workspace/mpq-collaborative-perception/outputs/trt_inputs \
  --keep-record-len 2
```

Use `--max-samples 50` to produce a small calibration set.

## 7. Explicit Q/DQ Graphs

Final paper configuration (quantized backbone plus regression/direction heads; fusion, compression, shrink layer, and classification head remain high precision):

```bash
python tools/insert_qdq.py \
  --input-onnx outputs/where2comm_fp32.onnx \
  --output-onnx outputs/where2comm_qdq_rd.onnx \
  --report-path outputs/where2comm_qdq_rd.json \
  --input-root outputs/trt_inputs \
  --index-path outputs/trt_inputs/index.json \
  --max-samples 20 \
  --act-percentile 99.9 \
  --weight-bits 8 \
  --act-bits 8 \
  --quantize-reg-head \
  --quantize-dir-head
```

Head-wise ablations are generated by combining `--quantize-cls-head`, `--quantize-reg-head`, and `--quantize-dir-head`. `--quantize-shrink` and `--quantize-compressor` are intentionally disabled for the final engine.

## 8. TensorRT Engines

FP32/FP16 engines can be built with the TensorRT `trtexec` shipped on the target:

```bash
trtexec \
  --onnx=outputs/where2comm_fp32.onnx \
  --saveEngine=outputs/where2comm_fp16.engine \
  --fp16 \
  --minShapes=voxel_features:26640x32x4,voxel_coords:26640x4,voxel_num_points:26640,record_len:2,pairwise_t_matrix:1x5x5x4x4 \
  --optShapes=voxel_features:26640x32x4,voxel_coords:26640x4,voxel_num_points:26640,record_len:2,pairwise_t_matrix:1x5x5x4x4 \
  --maxShapes=voxel_features:26640x32x4,voxel_coords:26640x4,voxel_num_points:26640,record_len:2,pairwise_t_matrix:1x5x5x4x4 \
  --memPoolSize=workspace:2048MiB
```

Build the explicit Q/DQ engine:

```bash
python tools/build_tensorrt_engine.py \
  --build_mode qdq_explicit \
  --onnx_path outputs/where2comm_qdq_rd.onnx \
  --engine_path outputs/where2comm_qdq_rd.engine
```

Build a calibration-based INT8 baseline:

```bash
python tools/build_tensorrt_engine.py \
  --build_mode fp32_calib \
  --onnx_path outputs/where2comm_fp32.onnx \
  --engine_path outputs/where2comm_calibration_int8.engine \
  --input_root outputs/trt_inputs \
  --index_path outputs/trt_inputs/index.json \
  --cache_path outputs/where2comm_int8.cache \
  --max_samples 50
```

## 9. TensorRT Inference and AP

Run on the target device after transferring the engine and exported inputs:

```bash
python tools/run_tensorrt.py \
  --engine_path outputs/where2comm_qdq_rd.engine \
  --input_root outputs/trt_inputs \
  --index_path outputs/trt_inputs/index.json \
  --overwrite
```

Evaluate the saved outputs:

```bash
python tools/evaluate_tensorrt.py \
  --config_path configs/pointpillar_where2comm_dair.yaml \
  --input_root outputs/trt_inputs \
  --index_path outputs/trt_inputs/index.json \
  --save_dir outputs/evaluation
```

For timing, discard warm-up iterations and keep batch size, post-processing, clocks, power mode, and input subset identical across engines. The reported paper latency was measured on a Jetson Xavier NX 8 GB.
