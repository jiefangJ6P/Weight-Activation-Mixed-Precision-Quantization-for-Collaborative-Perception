import argparse
import json
import os
import tempfile
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


DEFAULT_EXCLUDE_PREFIXES = [
    "model.shrink_conv",
    "model.cls_head",
    "model.reg_head",
    "model.dir_head",
    "model.naive_compressor",
    "model.fusion_net",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Insert explicit activation+weight Q/DQ pairs for backbone-like Conv/ConvTranspose nodes."
    )
    parser.add_argument("--input-onnx", type=str, required=True)
    parser.add_argument("--output-onnx", type=str, required=True)
    parser.add_argument("--report-path", type=str, required=True)
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--act-sample-size", type=int, default=2048)
    parser.add_argument("--act-percentile", type=float, default=99.9)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--quantize-shrink", action="store_true")
    parser.add_argument("--quantize-compressor", action="store_true")
    parser.add_argument("--quantize-cls-head", action="store_true")
    parser.add_argument("--quantize-reg-head", action="store_true")
    parser.add_argument("--quantize-dir-head", action="store_true")
    parser.add_argument("--exclude-prefixes", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_exclude_prefixes(args):
    if args.exclude_prefixes is not None:
        return list(args.exclude_prefixes)

    exclude_prefixes = list(DEFAULT_EXCLUDE_PREFIXES)
    enabled = {
        "model.shrink_conv": args.quantize_shrink,
        "model.naive_compressor": args.quantize_compressor,
        "model.cls_head": args.quantize_cls_head,
        "model.reg_head": args.quantize_reg_head,
        "model.dir_head": args.quantize_dir_head,
    }
    return [prefix for prefix in exclude_prefixes if not enabled.get(prefix, False)]


def get_initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def get_node_weight_name(node):
    if node.op_type not in ("Conv", "ConvTranspose"):
        return None
    if len(node.input) < 2:
        return None
    return node.input[1]


def should_quantize_node(node, initializer_names, exclude_prefixes):
    weight_name = get_node_weight_name(node)
    if not weight_name or weight_name not in initializer_names:
        return False
    if any(weight_name.startswith(prefix) for prefix in exclude_prefixes):
        return False
    return True


def get_weight_axis(node):
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        return 1
    raise KeyError(f"Unsupported node type for weight quantization: {node.op_type}")


def symmetric_per_channel_qparams(weight, axis, qmax):
    moved = np.moveaxis(weight, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    abs_max = np.max(np.abs(flat), axis=1)
    scale = np.maximum(abs_max / float(qmax), 1e-8).astype(np.float32)
    zero_point = np.zeros_like(scale, dtype=np.int8)
    reshape_dims = [1] * weight.ndim
    reshape_dims[axis] = weight.shape[axis]
    scale_reshaped = scale.reshape(reshape_dims)
    quantized = np.round(weight / scale_reshaped)
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
    return quantized, scale, zero_point


def symmetric_per_tensor_qparams(abs_values, qmax):
    if abs_values.size == 0:
        max_abs = 1e-8
    else:
        max_abs = float(np.percentile(abs_values, 99.9))
        if max_abs <= 0:
            max_abs = float(np.max(abs_values))
        max_abs = max(max_abs, 1e-8)
    scale = np.array(max_abs / float(qmax), dtype=np.float32)
    zero_point = np.array(0, dtype=np.int8)
    return scale, zero_point, max_abs


def load_index_records(index_path, input_root, max_samples):
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


def load_sample(sample_dir):
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
        arr = np.load(path)
        if name in ("voxel_features", "pairwise_t_matrix"):
            arr = arr.astype(np.float32, copy=False)
        feed[name] = np.ascontiguousarray(arr)
    return feed


def get_quantized_nodes(model, exclude_prefixes):
    initializer_names = set(init.name for init in model.graph.initializer)
    touched_nodes = []
    activation_tensor_names = []
    graph_input_names = set(inp.name for inp in model.graph.input)

    for node_index, node in enumerate(model.graph.node):
        if not should_quantize_node(node, initializer_names, exclude_prefixes):
            continue
        touched_nodes.append({
            "node_index": node_index,
            "node_name": node.name or "<noname>",
            "op_type": node.op_type,
            "weight_name": node.input[1],
            "weight_axis": get_weight_axis(node),
            "act_input_name": node.input[0],
            "output_names": list(node.output),
        })
        act_name = node.input[0]
        if act_name not in graph_input_names and act_name not in activation_tensor_names:
            activation_tensor_names.append(act_name)
    return touched_nodes, activation_tensor_names


def build_calibration_model(model, activation_tensor_names):
    calib_model = onnx.ModelProto()
    calib_model.CopyFrom(model)
    existing_outputs = {out.name for out in calib_model.graph.output}

    for name in activation_tensor_names:
        if name in existing_outputs:
            continue
        calib_model.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )
    return calib_model


def collect_activation_stats(model, activation_tensor_names, sample_dirs, act_sample_size, act_percentile):
    if not activation_tensor_names:
        return {}

    calib_model = build_calibration_model(model, activation_tensor_names)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        temp_onnx_path = tmp.name
    try:
        onnx.save(calib_model, temp_onnx_path)
        session = ort.InferenceSession(temp_onnx_path, providers=["CPUExecutionProvider"])
        activation_values = {name: [] for name in activation_tensor_names}

        for sample_dir in sample_dirs:
            outputs = session.run(activation_tensor_names, load_sample(sample_dir))
            for name, out in zip(activation_tensor_names, outputs):
                arr = np.asarray(out, dtype=np.float32)
                if arr.size == 0:
                    continue
                flat = np.abs(arr.reshape(-1))
                if flat.size > act_sample_size:
                    idx = np.linspace(0, flat.size - 1, num=act_sample_size, dtype=np.int64)
                    flat = flat[idx]
                activation_values[name].append(flat)

        stats = {}
        qmax = 127
        for name, chunks in activation_values.items():
            merged = np.concatenate(chunks, axis=0) if chunks else np.asarray([], dtype=np.float32)
            max_abs = float(np.percentile(merged, act_percentile)) if merged.size else 1e-8
            if max_abs <= 0 and merged.size:
                max_abs = float(np.max(merged))
            max_abs = max(max_abs, 1e-8)
            scale = np.array(max_abs / float(qmax), dtype=np.float32)
            zero_point = np.array(0, dtype=np.int8)
            stats[name] = {
                "scale": scale,
                "zero_point": zero_point,
                "max_abs": max_abs,
                "num_observations": int(merged.size),
            }
        return stats
    finally:
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)


def make_weight_qdq_nodes(weight_name, quant_name, scale_name, zero_name, dq_output_name, axis):
    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[weight_name, scale_name, zero_name],
        outputs=[quant_name],
        name=f"{weight_name}_QuantizeLinear",
        axis=axis,
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[quant_name, scale_name, zero_name],
        outputs=[dq_output_name],
        name=f"{weight_name}_DequantizeLinear",
        axis=axis,
    )
    return [q_node, dq_node]


def make_activation_qdq_nodes(tensor_name, quant_name, scale_name, zero_name, dq_output_name):
    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[tensor_name, scale_name, zero_name],
        outputs=[quant_name],
        name=f"{tensor_name}_act_QuantizeLinear",
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[quant_name, scale_name, zero_name],
        outputs=[dq_output_name],
        name=f"{tensor_name}_act_DequantizeLinear",
    )
    return [q_node, dq_node]


def main():
    args = parse_args()
    exclude_prefixes = resolve_exclude_prefixes(args)
    model = onnx.load(args.input_onnx)
    graph = model.graph
    initializer_map = get_initializer_map(graph)
    initializer_names = set(initializer_map)

    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_onnx)), exist_ok=True)

    touched_nodes, activation_tensor_names = get_quantized_nodes(model, exclude_prefixes)
    sample_dirs = load_index_records(args.index_path, args.input_root, args.max_samples)
    activation_stats = collect_activation_stats(
        model,
        activation_tensor_names,
        sample_dirs,
        act_sample_size=args.act_sample_size,
        act_percentile=args.act_percentile,
    )

    processed_weights = {}
    processed_activations = {}
    new_initializers = []
    new_nodes = []

    for node_index, node in enumerate(graph.node):
        updated_node = helper.make_node(
            node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            **{attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        )

        act_input_name = updated_node.input[0] if updated_node.input else None
        if act_input_name in activation_stats:
            if act_input_name not in processed_activations:
                scale_name = f"{act_input_name}_act_qdq_scale"
                zero_name = f"{act_input_name}_act_qdq_zero_point"
                quant_name = f"{act_input_name}_act_qdq_quant"
                dq_output_name = f"{act_input_name}_act_qdq_dequant"
                stats = activation_stats[act_input_name]

                new_initializers.extend([
                    numpy_helper.from_array(stats["scale"], name=scale_name),
                    numpy_helper.from_array(stats["zero_point"], name=zero_name),
                ])
                new_nodes.extend(
                    make_activation_qdq_nodes(
                        act_input_name,
                        quant_name,
                        scale_name,
                        zero_name,
                        dq_output_name,
                    )
                )
                processed_activations[act_input_name] = {
                    "dq_output_name": dq_output_name,
                    "scale_name": scale_name,
                    "zero_name": zero_name,
                    "max_abs": stats["max_abs"],
                    "num_observations": stats["num_observations"],
                }
            updated_node.input[0] = processed_activations[act_input_name]["dq_output_name"]

        if should_quantize_node(updated_node, initializer_names, exclude_prefixes):
            weight_name = updated_node.input[1]
            axis = get_weight_axis(updated_node)

            if weight_name not in processed_weights:
                weight_tensor = numpy_helper.to_array(initializer_map[weight_name]).astype(np.float32)
                qmax = 2 ** (args.weight_bits - 1) - 1
                q_weight, scale, zero_point = symmetric_per_channel_qparams(
                    weight_tensor,
                    axis=axis,
                    qmax=qmax,
                )
                quant_name = f"{weight_name}_weight_qdq_quant"
                scale_name = f"{weight_name}_weight_qdq_scale"
                zero_name = f"{weight_name}_weight_qdq_zero_point"
                dq_output_name = f"{weight_name}_weight_qdq_dequant"

                new_initializers.extend([
                    numpy_helper.from_array(scale.astype(np.float32), name=scale_name),
                    numpy_helper.from_array(zero_point.astype(np.int8), name=zero_name),
                ])
                new_nodes.extend(
                    make_weight_qdq_nodes(weight_name, quant_name, scale_name, zero_name, dq_output_name, axis)
                )
                processed_weights[weight_name] = {
                    "dq_output_name": dq_output_name,
                    "axis": axis,
                    "scale_name": scale_name,
                    "zero_name": zero_name,
                    "num_channels": int(scale.shape[0]),
                    "weight_shape": list(weight_tensor.shape),
                    "quantized_dtype": str(q_weight.dtype),
                }

            updated_node.input[1] = processed_weights[weight_name]["dq_output_name"]

        new_nodes.append(updated_node)

    replaced_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer) + new_initializers,
        value_info=list(graph.value_info),
    )
    replaced_graph.sparse_initializer.extend(graph.sparse_initializer)
    replaced_graph.quantization_annotation.extend(graph.quantization_annotation)

    new_model = helper.make_model(
        replaced_graph,
        producer_name="step3_make_backbone_act_weight_qdq_onnx",
    )
    new_model.ir_version = model.ir_version
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)

    report = {
        "input_onnx": args.input_onnx,
        "output_onnx": args.output_onnx,
        "report_path": args.report_path,
        "input_root": args.input_root,
        "index_path": args.index_path,
        "max_samples": args.max_samples,
        "act_sample_size": args.act_sample_size,
        "act_percentile": args.act_percentile,
        "quantize_targets": {
            "backbone": True,
            "shrink_conv": args.quantize_shrink,
            "naive_compressor": args.quantize_compressor,
            "cls_head": args.quantize_cls_head,
            "reg_head": args.quantize_reg_head,
            "dir_head": args.quantize_dir_head,
        },
        "exclude_prefixes": exclude_prefixes,
        "dry_run": args.dry_run,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "touched_node_count": len(touched_nodes),
        "touched_nodes": touched_nodes,
        "processed_weight_count": len(processed_weights),
        "processed_weights": processed_weights,
        "processed_activation_count": len(processed_activations),
        "processed_activations": processed_activations,
        "op_histogram": dict(Counter(node.op_type for node in new_model.graph.node)),
    }

    if not args.dry_run:
        onnx.save(new_model, args.output_onnx)
        onnx.checker.check_model(new_model)

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
