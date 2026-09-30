# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import random
import os
import time
import math
import subprocess
import threading
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--epoch', default=-1, type=int, help="epoch used")
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    parser.add_argument('--filter_record_len', type=int, default=None,
                        help='only infer samples whose effective record_len matches this value')
    parser.add_argument('--filter_max_samples', type=int, default=None,
                        help='maximum number of filtered samples to infer')
    parser.add_argument('--power_sample_interval', type=float, default=0.1,
                        help='seconds between nvidia-smi power samples during inference loop')
    parser.add_argument('--gpu_power_id', type=str, default=None,
                        help='physical GPU id/UUID for nvidia-smi power sampling; defaults to first CUDA_VISIBLE_DEVICES entry')
    opt = parser.parse_args()
    return opt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_effective_record_len(dataset, idx, comm_range):
    """
    Compute the effective record_len after communication-range filtering.
    """
    base_data_dict = dataset.retrieve_base_data(idx)

    ego_lidar_pose = None
    for _, cav_content in base_data_dict.items():
        if cav_content['ego']:
            ego_lidar_pose = cav_content['params']['lidar_pose']
            break

    assert ego_lidar_pose is not None

    cav_num = 0
    for _, cav_content in base_data_dict.items():
        lidar_pose = cav_content['params']['lidar_pose']
        distance = math.sqrt((lidar_pose[0] - ego_lidar_pose[0]) ** 2 +
                             (lidar_pose[1] - ego_lidar_pose[1]) ** 2)
        if distance <= comm_range:
            cav_num += 1

    return cav_num


def build_inference_dataset(dataset, record_len_filter=None, max_samples=None):
    if record_len_filter is None and max_samples is None:
        return dataset

    selected_indices = []
    target_total = max_samples if max_samples is not None else len(dataset)
    comm_range = dataset.params['comm_range']

    for idx in range(len(dataset)):
        effective_record_len = get_effective_record_len(dataset, idx, comm_range)

        if record_len_filter is not None and effective_record_len != record_len_filter:
            continue

        selected_indices.append(idx)
        if len(selected_indices) >= target_total:
            break

    print(f"Selected {len(selected_indices)} / {len(dataset)} samples for inference.")
    if record_len_filter is not None:
        print(f"Filter condition: effective record_len == {record_len_filter}")
    if max_samples is not None:
        print(f"Maximum filtered samples: {max_samples}")

    return Subset(dataset, selected_indices)


def override_communication_threshold(hypes, comm_thre):
    """
    Override communication threshold for supported Where2comm config layouts.
    """
    if comm_thre is None:
        return

    model_args = hypes.get('model', {}).get('args', {})
    communication_args = None
    config_path = None

    if 'fusion_args' in model_args and 'communication' in model_args['fusion_args']:
        communication_args = model_args['fusion_args']['communication']
        config_path = 'model.args.fusion_args.communication'
    elif 'where2comm' in model_args and 'communication' in model_args['where2comm']:
        communication_args = model_args['where2comm']['communication']
        config_path = 'model.args.where2comm.communication'

    if communication_args is None:
        raise KeyError("Cannot find communication config under model.args.fusion_args or model.args.where2comm")

    if 'thre' in communication_args:
        communication_args['thre'] = comm_thre
    if 'threshold' in communication_args:
        communication_args['threshold'] = comm_thre
    if 'thre' not in communication_args and 'threshold' not in communication_args:
        communication_args['thre'] = comm_thre

    print(f"Override communication threshold: {config_path} -> {comm_thre}")


def get_power_gpu_id(opt_gpu_power_id):
    if opt_gpu_power_id is not None:
        return opt_gpu_power_id

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices:
        return visible_devices.split(',')[0].strip()

    return '0'


class NvidiaSmiPowerSampler:
    """
    Samples whole-GPU power draw while inference is running.

    NVIDIA SMI does not expose reliable per-process power draw, so this records
    the selected device's instantaneous board power during the measured window.
    """

    def __init__(self, gpu_id, interval=0.1):
        self.gpu_id = str(gpu_id)
        self.interval = max(float(interval), 0.01)
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self.available = False

    def _query_power(self):
        cmd = [
            'nvidia-smi',
            f'--id={self.gpu_id}',
            '--query-gpu=power.draw',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip().splitlines()[0].strip()
        return float(value)

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                power = self._query_power()
            except (FileNotFoundError, ValueError, IndexError):
                power = None
            if power is not None:
                self.available = True
                self.samples.append((time.time(), power))
            self._stop_event.wait(self.interval)

    def start(self):
        try:
            power = self._query_power()
        except (FileNotFoundError, ValueError, IndexError):
            print("GPU power sampling disabled: nvidia-smi power query is unavailable.")
            return

        if power is None:
            print(f"GPU power sampling disabled: cannot query GPU {self.gpu_id}.")
            return

        self.available = True
        self.samples.append((time.time(), power))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        print(f"GPU power sampling enabled on physical GPU {self.gpu_id}.")

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()

    def summary(self):
        if not self.samples:
            return None

        powers = [sample[1] for sample in self.samples]
        return {
            'count': len(powers),
            'avg': sum(powers) / len(powers),
            'min': min(powers),
            'max': max(powers)
        }


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single', 'intermediatelate'] 

    hypes = yaml_utils.load_yaml(None, opt)
        
    
    override_communication_threshold(hypes, opt.comm_thre)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model, epoch=opt.epoch)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting random seed
    set_random_seed(303)
    
    if hypes['fusion']['core_method'] != opt.fusion_method and opt.fusion_method == "intermediatelate":
        print(f"Change the fusion method in dataset config to {opt.fusion_method}")
        hypes['fusion']['core_method'] = opt.fusion_method
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    inference_dataset = build_inference_dataset(
        opencood_dataset,
        record_len_filter=opt.filter_record_len,
        max_samples=opt.filter_max_samples
    )
    data_loader = DataLoader(inference_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note
    total_comm_rates = []
    inference_times = []
    power_sampler = NvidiaSmiPowerSampler(
        get_power_gpu_id(opt.gpu_power_id),
        interval=opt.power_sample_interval
    )
    power_sampler.start()

    try:
        for i, batch_data in tqdm(enumerate(data_loader)):

            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()

                if opt.fusion_method in ('late', 'intermediatelate'):
                    infer_result = inference_utils.inference_late_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'early':
                    infer_result = inference_utils.inference_early_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                    comm_rate= infer_result.get('comm_rate', 0)
                    # comm_rates = infer_result.get('comm_rate', 0)
                    #有修改
                    total_comm_rates.append(comm_rate)
                    print(f"comm_rate: {comm_rate}")

                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no_w_uncertainty':
                    infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_time = time.perf_counter() - infer_start
                inference_times.append(infer_time)
                print(f"inference_time: {infer_time:.6f}s")

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)
                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                    infer_result.update({"cav_box_np": cav_box_np, \
                                        "lidar_agent_record": lidar_agent_record})

                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    """
                    If you want 3D visualization, uncomment lines below
                    """
                    # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                    # simple_vis.visualize(infer_result,
                    #                     batch_data['ego'][
                    #                         'origin_lidar'][0],
                    #                     hypes['postprocess']['gt_range'],
                    #                     vis_save_path,
                    #                     method='3d',
                    #                     left_hand=left_hand)
                    
                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
            torch.cuda.empty_cache()
    finally:
        power_sampler.stop()

    if len(inference_times) > 0:
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time per sample: {avg_inference_time:.6f}s "
              f"({len(inference_times)} samples)")
    else:
        avg_inference_time = 0
        print("Average inference time per sample: no valid samples.")

    power_summary = power_sampler.summary()
    if power_summary is not None:
        avg_gpu_power = power_summary['avg']
        print("GPU power draw during inference: "
              f"avg={power_summary['avg']:.2f}W, "
              f"min={power_summary['min']:.2f}W, "
              f"max={power_summary['max']:.2f}W, "
              f"samples={power_summary['count']}")
    else:
        avg_gpu_power = 0
        print("GPU power draw during inference: unavailable.")

    print(total_comm_rates)
    if len(total_comm_rates) > 0:
        avg = sum(total_comm_rates) / len(total_comm_rates)
        comm_rate = avg.item() if hasattr(avg, "item") else avg
        #.item() 是 PyTorch Tensor 的方法，用来把一个 0‑维张量转换成 Python 标量。但是如果结果已经是 float，再调用 .item() 就会报错
        #comm_rates = (sum(total_comm_rates)/len(total_comm_rates)).item()
    else:
        comm_rate = 0
    
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    
    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = ('Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | '
               'comm_rate: {:.06f} | avg_inference_time: {:.06f}s | avg_gpu_power: {:.02f}W\n').format(
                    resume_epoch, ap30, ap50, ap70, comm_rate, avg_inference_time, avg_gpu_power)
        if opt.comm_thre is not None:
            msg = ('Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | '
                   'comm_rate: {:.06f} | comm_thre: {:.04f} | entropy: {:.06f} | '
                   'avg_inference_time: {:.06f}s | avg_gpu_power: {:.02f}W\n').format(
                        resume_epoch, ap30, ap50, ap70, comm_rate, opt.comm_thre, 0,
                        avg_inference_time, avg_gpu_power)
        f.write(msg)
        print(msg)

if __name__ == '__main__':
    main()
