import logging
import os
import re
from pathlib import Path
from datetime import datetime

import hydra
import einops
import imageio
from tqdm import *
import numpy as np
import torch
from transformers import AutoTokenizer

from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.agent.dataset import Bench2DriveDataset
from src.agent.drivepi0.input_utils import (
    build_processor_from_config,
    prepare_drivepi_inputs,
)
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)

val_list = [
    'StaticCutIn_Town05_Route226_Weather18',
    'MergerIntoSlowTrafficV2_Town12_Route857_Weather25',
    'YieldToEmergencyVehicle_Town04_Route166_Weather10',
    'ConstructionObstacle_Town10HD_Route74_Weather22',
    'VehicleTurningRoutePedestrian_Town15_Route445_Weather11',
    'VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21',
    'SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2',
    'LaneChange_Town06_Route307_Weather21',
    'ConstructionObstacleTwoWays_Town12_Route1093_Weather1',
    'HazardAtSideLaneTwoWays_Town12_Route1151_Weather7',
    'OppositeVehicleTakingPriority_Town04_Route214_Weather6',
    'NonSignalizedJunctionRightTurn_Town03_Route126_Weather18',
    'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9',
    'ParkedObstacle_Town06_Route282_Weather22',
    'ControlLoss_Town10HD_Route378_Weather14',
    'ControlLoss_Town04_Route170_Weather14',
    'OppositeVehicleRunningRedLight_Town04_Route180_Weather23',
    'InterurbanAdvancedActorFlow_Town06_Route324_Weather2',
    'HighwayCutIn_Town12_Route1029_Weather15',
    'MergerIntoSlowTraffic_Town06_Route317_Weather5',
    'NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3',
    'AccidentTwoWays_Town12_Route1115_Weather23',
    'ParkingCrossingPedestrian_Town13_Route545_Weather25',
    'VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8',
    'ParkingExit_Town12_Route922_Weather12',
    'VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23',
    'HardBreakRoute_Town01_Route32_Weather6',
    'DynamicObjectCrossing_Town01_Route3_Weather3',
    'ConstructionObstacle_Town12_Route78_Weather0',
    'EnterActorFlow_Town03_Route132_Weather2',
    'HazardAtSideLane_Town10HD_Route373_Weather9',
    'InvadingTurn_Town02_Route95_Weather9',
    'TJunction_Town05_Route260_Weather0',
    'VehicleTurningRoute_Town15_Route504_Weather10',
    'DynamicObjectCrossing_Town02_Route11_Weather11',
    'TJunction_Town06_Route306_Weather20',
    'ParkedObstacleTwoWays_Town13_Route1333_Weather26',
    'SignalizedJunctionRightTurn_Town03_Route118_Weather14',
    'NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13',
    'VehicleOpensDoorTwoWays_Town12_Route1203_Weather7',
    'CrossingBicycleFlow_Town12_Route977_Weather15',
    'SignalizedJunctionLeftTurn_Town04_Route173_Weather26',
    'HighwayExit_Town06_Route312_Weather0',
    'Accident_Town05_Route218_Weather10',
    'ParkedObstacle_Town10HD_Route372_Weather8',
    'InterurbanActorFlow_Town12_Route1291_Weather1',
    'ParkingCutIn_Town13_Route1343_Weather1',
    'VehicleTurningRoutePedestrian_Town15_Route481_Weather19',
    'PedestrianCrossing_Town13_Route747_Weather19',
    'BlockedIntersection_Town03_Route135_Weather5',
]

class DrivePiZeroInferenceAgent:
    def __init__(self, cfg):
        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.model = DrivePiZeroInference(cfg, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        
        self.tokenizer, self.processor = build_processor_from_config(cfg)
        print('Agent done')
    
    
    def run(self, data_dict):
        images_front = data_dict["image_front"]
        images_front = images_front.unsqueeze(0)
        images_front_history = data_dict["image_front_history"]
        images_front_history = images_front_history.unsqueeze(0)
        # images_back = data_dict["image_back"]
        # images_back = images_back.unsqueeze(0)
            
        state = data_dict["state"].unsqueeze(0)
        texts = [data_dict["language_instruction"]]
            
        images_front = einops.rearrange(images_front, "B H W C -> B C H W")  # remove cond_steps dimension
        images_front = images_front.unsqueeze(1)
        images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")  # remove cond_steps dimension
        images_front_history = images_front_history.unsqueeze(1)
        # images_back = einops.rearrange(images_back, "B H W C -> B C H W")  # remove cond_steps dimension
        # images_back = images_back.unsqueeze(1)
            
        # images = torch.cat((images_front, images_front_history, images_back), dim=1)
        images = torch.cat((images_front, images_front_history), dim=1)
        inputs = prepare_drivepi_inputs(
            model=self.model,
            processor=self.processor,
            texts=texts,
            images=images,
            proprios=state,
            dtype=self.dtype,
            device=self.device,
            split_mask=True,
        )
        trajectory = self.model.infer_action(**inputs)
        return trajectory
    
    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")
    
    def run_batch_inference(self, val_data_dir: str, output_dir: str, statistic_path: str = "statistics.json"):
        """
        按场景批量推理，每个场景保存一个 npy 文件
        
        Args:
            val_data_dir: val 数据目录路径（包含所有 pkl 文件）
            output_dir: 输出目录路径
            statistic_path: 统计文件路径，用于归一化
        """
        from src.data.utils.normalization import Normalize
        from src.data.utils.image import image_normalization
        
        val_data_path = Path(val_data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        normalize = Normalize.get_instance(statistic_path)
        
        # 遍历每个场景
        for scene_name in tqdm(val_list, desc="Processing scenes"):
            # 找到该场景的所有 step 文件
            scene_files = []
            for pkl_file in val_data_path.glob(f"{scene_name}_step*.pkl"):
                # 提取 step 数字
                match = re.search(r'_step(\d+)\.pkl$', pkl_file.name)
                if match:
                    step_num = int(match.group(1))
                    scene_files.append((step_num, pkl_file))
            
            if len(scene_files) == 0:
                log.warning(f"No files found for scene: {scene_name}")
                continue
            
            # 按 step 数字排序
            scene_files.sort(key=lambda x: x[0])
            
            # 存储该场景的所有轨迹
            scene_pred_trajectories = []
            scene_gt_trajectories = []
            
            # 处理该场景的每个 step 文件
            for step_num, pkl_file in tqdm(scene_files, desc=f"Scene: {scene_name}", leave=False):
                try:
                    # 加载 pkl 文件
                    with open(pkl_file, 'rb') as f:
                        import pickle
                        all_data = pickle.load(f)
                    
                    # 准备数据
                    state = normalize.prepare_state(all_data)
                    data_dict = {}
                    
                    # 处理图像
                    image_front_path = all_data['his_cam_front'].numpy().decode('utf-8')
                    data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                    # image_back_path = all_data['his_cam_back'].numpy().decode('utf-8')
                    # data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                    image_front_time_path = all_data['his_cam_front_time'].numpy().decode('utf-8')
                    data_dict['image_front_history'] = image_normalization(image_front_time_path, (224, 224))
                    
                    data_dict["language_instruction"] = "predict trajectory"
                    data_dict['state'] = state
                    
                    # 推理
                    pred_trajectory = self.run(data_dict=data_dict)  # 应该是 [horizon, 2]
                    
                    # 确保 pred_trajectory 形状正确
                    if len(pred_trajectory.shape) == 1:
                        # 如果是 [horizon*2]，需要 reshape
                        horizon = pred_trajectory.shape[0] // 2
                        pred_trajectory = pred_trajectory.reshape(horizon, 2)
                    elif len(pred_trajectory.shape) == 3:
                        # 如果是 [1, horizon, 2]，先 squeeze
                        pred_trajectory = pred_trajectory.squeeze(0)
                    elif len(pred_trajectory.shape) == 2 and pred_trajectory.shape[0] != 10:
                        # 如果形状是 [2, 2] 或其他不对的形状，记录警告
                        log.warning(f"Unexpected pred_trajectory shape: {pred_trajectory.shape}, expected [10, 2]")
                    # 现在应该是 [horizon, 2]
                    
                    # 获取真实轨迹（与 eval.py 保持一致：从 DataLoader 获取的方式）
                    gt_trajectory = normalize.prepare_traj(all_data)  # [horizon, 2] 或 [1, horizon, 2]
                    
                    # 确保轨迹形状正确：如果是 [1, horizon, 2]，先 squeeze；如果是 [horizon, 2]，保持不变
                    if len(gt_trajectory.shape) == 3:
                        gt_trajectory = gt_trajectory.squeeze(0)  # [horizon, 2]
                    
                    # 反归一化预测轨迹到真实坐标（按照 eval.py 的方式）
                    # pred_trajectory 形状: [horizon, 2]，先加 batch 维变成 [1, horizon, 2]
                    pred = pred_trajectory.unsqueeze(0)  # [1, horizon, 2]
                    # infer_traj 内部使用 fur_x_01 / fur_x_99, fur_y_01 / fur_y_99
                    # 对 x / y 分量做 standard_normalize 的逆变换
                    # (norm + 1) * (p_99 - p_01) / 2 + p_01
                    x_pred, y_pred = normalize.infer_traj(pred)
                    # 拼回 [horizon, 2]（torch.stack 会自动处理 numpy array）
                    # 确保 x_pred 和 y_pred 是 tensor
                    if isinstance(x_pred, np.ndarray):
                        x_pred = torch.from_numpy(x_pred)
                    if isinstance(y_pred, np.ndarray):
                        y_pred = torch.from_numpy(y_pred)
                    pred_traj_real = torch.stack([x_pred, y_pred], dim=1)  # [horizon, 2]
                    
                    # 反归一化真实轨迹到真实坐标（按照 eval.py 的方式）
                    # gt_trajectory 形状: [horizon, 2]，先加 batch 维变成 [1, horizon, 2]
                    label = gt_trajectory.unsqueeze(0)  # [1, horizon, 2]
                    x_label, y_label = normalize.infer_traj(label)
                    # 确保 x_label 和 y_label 是 tensor
                    if isinstance(x_label, np.ndarray):
                        x_label = torch.from_numpy(x_label)
                    if isinstance(y_label, np.ndarray):
                        y_label = torch.from_numpy(y_label)
                    # 拼回 [horizon, 2]（torch.stack 会自动处理 numpy array）
                    gt_traj_real = torch.stack([x_label, y_label], dim=1)  # [horizon, 2]
                    
                    # 确保最终形状正确：应该是 [horizon, 2]
                    if len(pred_traj_real.shape) != 2 or pred_traj_real.shape[1] != 2:
                        log.warning(f"Unexpected pred_traj_real shape: {pred_traj_real.shape}, expected [horizon, 2]. "
                                  f"pred_trajectory shape: {pred_trajectory.shape}, "
                                  f"x_pred shape: {x_pred.shape if hasattr(x_pred, 'shape') else type(x_pred)}, "
                                  f"y_pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else type(y_pred)}")
                    if len(gt_traj_real.shape) != 2 or gt_traj_real.shape[1] != 2:
                        log.warning(f"Unexpected gt_traj_real shape: {gt_traj_real.shape}, expected [horizon, 2]")
                    
                    # 转换为 numpy 并存储
                    scene_pred_trajectories.append(pred_traj_real.cpu().numpy())
                    scene_gt_trajectories.append(gt_traj_real.cpu().numpy())
                    
                except Exception as e:
                    log.error(f"Error processing {pkl_file}: {e}")
                    continue
            
            if len(scene_pred_trajectories) == 0:
                log.warning(f"No valid trajectories for scene: {scene_name}")
                continue
            
            # 合并该场景的所有轨迹 [N, horizon, 2]
            scene_pred_array = np.stack(scene_pred_trajectories, axis=0)
            scene_gt_array = np.stack(scene_gt_trajectories, axis=0)
            
            # 保存为 npy 文件
            pred_save_path = output_path / f"{scene_name}_pred_trajectories.npy"
            gt_save_path = output_path / f"{scene_name}_gt_trajectories.npy"
            
            np.save(pred_save_path, scene_pred_array)
            np.save(gt_save_path, scene_gt_array)
            
            log.info(f"Saved {len(scene_pred_trajectories)} trajectories for scene: {scene_name}")
            log.info(f"  - Predicted: {pred_save_path}")
            log.info(f"  - GT: {gt_save_path}")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse
    import pickle
    from src.data.utils.normalization import Normalize
    from src.data.utils.image import image_normalization
    
    parser = argparse.ArgumentParser(
        description="DrivePi-0 Inference Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
        # Single file inference:
        python src/agent/drivepi0/inference.py --mode single --test_data_path "path/to/file.pkl"
        
        # Batch inference by scene:
        python src/agent/drivepi0/inference.py --mode batch --val_data_dir "b2d_dynamic_camera/val"
        """
    )
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch",
                        help="Inference mode: single (one file) or batch (by scene)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to single pkl file (for single mode)")
    parser.add_argument("--val_data_dir", type=str, default="b2d_dynamic_camera/val",
                        help="Path to val data directory (for batch mode)")
    parser.add_argument("--config_path", type=str, default="config/eval/DrivePi0/base.yaml",
                        help="Path to config file")
    parser.add_argument("--statistic_path", type=str, default="statistics.json",
                        help="Path to statistics file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: logs/inference/drive-pi0/TIMESTAMP)")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)
    statistic_path = args.statistic_path
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("logs/inference/drive-pi0") / timestamp
    else:
        output_dir = Path(args.output_dir)
    
    inference_agent = DrivePiZeroInferenceAgent(config)
    
    if args.mode == "single":
        # 单文件推理模式（保持向后兼容）
        if args.test_data_path is None:
            parser.error("--test_data_path is required for single mode")
        
        pkl_data_path = args.test_data_path
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            # prepare state
            normalize = Normalize.get_instance(statistic_path)
            state = normalize.prepare_state(all_data)
            # prepare 
            data_dict = {}
            image_front_path = all_data['his_cam_front'].numpy().decode('utf-8')
            data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
            # image_back_path = all_data['his_cam_back'].numpy().decode('utf-8')
            # data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
            image_front_time_path = all_data['his_cam_front_time'].numpy().decode('utf-8')
            data_dict['image_front_history'] = image_normalization(image_front_time_path, (224, 224))
        
            # prepare text
            data_dict["language_instruction"] = "predict trajectory"

            data_dict['state'] = state
            trajectory = inference_agent.run(data_dict=data_dict)
            x_pred, y_pred = normalize.infer_traj(trajectory)
            print('x:', x_pred)
            print('y:', y_pred)
    
    elif args.mode == "batch":
        # 批量推理模式（按场景）
        log.info(f"Starting batch inference...")
        log.info(f"Val data directory: {args.val_data_dir}")
        log.info(f"Output directory: {output_dir}")
        
        inference_agent.run_batch_inference(
            val_data_dir=args.val_data_dir,
            output_dir=str(output_dir),
            statistic_path=statistic_path
        )
        
        log.info(f"Batch inference completed. Results saved to: {output_dir}")
