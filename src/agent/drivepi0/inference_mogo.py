"""
    用于推理 MOGO 数据集的 DrivePi-0 模型:
    python src/agent/drivepi0/inference_mogo.py --mode batch \ 
        --val_data_dir "/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637/drivepi" \ 
        --config_path config/inference/DrivePi0/base_mogo.yaml \ 
        --statistic_path statistics.json \ 
        --output_dir "logs/inference/drive-pi0/"
"""
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import hydra
import einops
import imageio
from tqdm import *
import numpy as np
import torch
from transformers import AutoTokenizer

from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)  # 保留用于 log_allocated_gpu_memory 等工具函数

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
        print(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        # 保存归一化模式（优先从 data 中读取）
        self.norm_mode = cfg.get("norm_mode") or cfg.data.get("norm_mode", "percentile")
        self.nav_onehot = cfg.data.get("nav_onehot", False)
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
        model_inputs = self.processor(text=texts, images=images)

        # build causal mask and position ids for trajectory
        causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": state_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": state.to(self.dtype),
        }
                
        image_text_proprio_mask, action_mask = (
            self.model.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
        print(f"Loaded model from {path}")
    
    def run_batch_inference(self, val_data_dir: str, output_dir: str, statistic_path: str = "statistics.json", max_samples: Optional[int] = None):
        """
        批量推理，按时间戳目录（类似场景）处理数字命名的 pkl 文件
        每个时间戳目录生成独立的 npy 文件（与 inference.py 的逻辑一致）
        
        Args:
            val_data_dir: val 数据目录路径
                - 如果指向根目录（如 2O-bags/），会遍历所有时间戳目录
                - 如果指向具体的时间戳目录下的 drivepi 目录，只处理该目录
            output_dir: 输出目录路径
            statistic_path: 统计文件路径，用于归一化
            max_samples: 最多处理的样本数量（用于调试，None 表示处理所有样本）
        """
        from src.data.utils.normalization import Normalize
        from src.data.utils.image import image_normalization
        
        val_data_path = Path(val_data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 记录使用的统计文件路径
        print(f"Using statistics file: {statistic_path}")
        # 使用配置中的 norm_mode 创建归一化器
        norm_mode = getattr(self, "norm_mode", "percentile")
        print(f"Using norm_mode: {norm_mode}")
        normalize = Normalize.get_instance(statistic_path, norm_mode=norm_mode)
        
        # 读取并打印 statistics.json 相关字段
        import json
        with open(statistic_path, 'r') as f:
            stats_data = json.load(f)
        print(f"\n{'='*60}")
        print(f"归一化参数 (statistics.json):")
        print(f"{'='*60}")
        print(f"轨迹归一化参数:")
        print(f"  fur_x: [{stats_data['fur_x'][0]:.6f}, {stats_data['fur_x'][1]:.6f}] (1%, 99%百分位数)")
        print(f"  fur_y: [{stats_data['fur_y'][0]:.6f}, {stats_data['fur_y'][1]:.6f}] (1%, 99%百分位数)")
        print(f"目标点归一化参数:")
        print(f"  command_far_x: [{stats_data['command_far_x'][0]:.6f}, {stats_data['command_far_x'][1]:.6f}] (1%, 99%百分位数)")
        print(f"  command_far_y: [{stats_data['command_far_y'][0]:.6f}, {stats_data['command_far_y'][1]:.6f}] (1%, 99%百分位数)")
        print(f"状态归一化参数:")
        print(f"  speed: [{stats_data['speed'][0]:.6f}, {stats_data['speed'][1]:.6f}] (1%, 99%百分位数)")
        print(f"  acceleration: [{stats_data['acceleration'][0]:.6f}, {stats_data['acceleration'][1]:.6f}] (1%, 99%百分位数)")
        print(f"  angular_velocity: [{stats_data['angular_velocity'][0]:.6f}, {stats_data['angular_velocity'][1]:.6f}] (1%, 99%百分位数)")
        print(f"  theta: [{stats_data['theta'][0]:.6f}, {stats_data['theta'][1]:.6f}] (1%, 99%百分位数)")
        print(f"{'='*60}\n")
        
        # 处理图像路径提取的辅助函数
        def extract_image_path(img_ref):
            """从不同格式的图像引用中提取路径字符串"""
            # 处理 dict 类型（StringWrapper 序列化为 dict 的情况）
            if isinstance(img_ref, dict) and '_string_wrapper_marker' in img_ref:
                img_ref = img_ref.get('_bytes_value', b'')
            # 处理有 .numpy() 方法的对象
            if hasattr(img_ref, "numpy"):
                img_ref = img_ref.numpy()
            # 处理 bytes 或 str
            if isinstance(img_ref, bytes):
                return img_ref.decode('utf-8')
            elif isinstance(img_ref, str):
                return img_ref
            else:
                raise ValueError(f"Unsupported type for image reference: {type(img_ref)}")
        
        # 处理单个 pkl 目录的函数
        def process_pkl_directory(pkl_dir: Path, timestamp_name: str):
            """
            处理单个 pkl 目录下的所有文件
            
            Args:
                pkl_dir: 包含 pkl 文件的目录路径
                timestamp_name: 时间戳名称（用于生成输出文件名）
            
            Returns:
                (pred_trajectories, gt_trajectories) 列表，如果处理失败返回 None
            """
            # 读取数字命名的 pkl 文件：00001.pkl, 00002.pkl, ...
            pkl_files = []
            for pkl_file in pkl_dir.glob("*.pkl"):
                # 匹配数字命名的文件（如 00001.pkl）
                match = re.match(r"(\d+)\.pkl$", pkl_file.name)
                if match:
                    file_num = int(match.group(1))
                    pkl_files.append((file_num, pkl_file))
            
            # 按文件编号排序
            pkl_files = sorted(pkl_files, key=lambda x: x[0])
            
            if not pkl_files:
                print(f"[WARNING] No valid numeric '*.pkl' files found in {pkl_dir}")
                return None
            
            # 存储该时间戳的所有轨迹
            scene_pred_trajectories = []
            scene_gt_trajectories = []
            
            # 处理每个文件
            processed_count = 0
            for file_num, pkl_file in tqdm(pkl_files, desc=f"Timestamp: {timestamp_name}", leave=False):
                # 限制处理的样本数量（用于调试）
                if max_samples is not None and processed_count >= max_samples:
                    print(f"达到最大样本数限制 ({max_samples})，停止处理")
                    break
                try:
                    # 加载 pkl 文件
                    with open(pkl_file, 'rb') as f:
                        import pickle
                        try:
                            all_data = pickle.load(f, encoding='latin-1')
                        except TypeError:
                            # 如果是 Python 3 生成的 pickle，不需要 encoding 参数
                            all_data = pickle.load(f)
                    
                    # 准备数据
                    state = normalize.prepare_state(all_data, nav_onehot=self.nav_onehot)
                    
                    data_dict = {}
                    image_front_path = extract_image_path(all_data['his_cam_front'])
                    data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                    image_front_time_path = extract_image_path(all_data['his_cam_front_time'])
                    data_dict['image_front_history'] = image_normalization(image_front_time_path, (224, 224))
                    data_dict["language_instruction"] = "predict trajectory"
                    data_dict['state'] = state
                    
                    # 推理
                    pred_trajectory = self.run(data_dict=data_dict)
                    
                    # 确保 pred_trajectory 形状正确
                    if len(pred_trajectory.shape) == 1:
                        horizon = pred_trajectory.shape[0] // 2
                        pred_trajectory = pred_trajectory.reshape(horizon, 2)
                    elif len(pred_trajectory.shape) == 3:
                        pred_trajectory = pred_trajectory.squeeze(0)
                    elif len(pred_trajectory.shape) == 2 and pred_trajectory.shape[0] != 10:
                        print(f"[WARNING] Unexpected pred_trajectory shape: {pred_trajectory.shape}, expected [10, 2]")
                    
                    # 反归一化预测轨迹
                    pred = pred_trajectory.unsqueeze(0)
                    x_pred, y_pred = normalize.infer_traj(pred)
                    if isinstance(x_pred, np.ndarray):
                        x_pred = torch.from_numpy(x_pred)
                    if isinstance(y_pred, np.ndarray):
                        y_pred = torch.from_numpy(y_pred)
                    pred_traj_real = torch.stack([x_pred, y_pred], dim=1).squeeze(0)
                    if isinstance(pred_traj_real, torch.Tensor):
                        pred_traj_real = pred_traj_real.cpu().numpy()
                    
                    # 获取真实轨迹
                    gt_trajectory = normalize.prepare_traj(all_data)
                    if len(gt_trajectory.shape) == 3:
                        gt_trajectory = gt_trajectory.squeeze(0)
                    label = gt_trajectory.unsqueeze(0)
                    x_label, y_label = normalize.infer_traj(label)
                    if isinstance(x_label, np.ndarray):
                        x_label = torch.from_numpy(x_label)
                    if isinstance(y_label, np.ndarray):
                        y_label = torch.from_numpy(y_label)
                    gt_traj_real = torch.stack([x_label, y_label], dim=1).squeeze(0)
                    if isinstance(gt_traj_real, torch.Tensor):
                        gt_traj_real = gt_traj_real.cpu().numpy()
                    
                    # 存储（已经是 numpy 数组）
                    scene_pred_trajectories.append(pred_traj_real)
                    scene_gt_trajectories.append(gt_traj_real)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"[ERROR] Error processing {pkl_file.name}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            if len(scene_pred_trajectories) == 0:
                print(f"[WARNING] No valid trajectories for timestamp: {timestamp_name}")
                return None
            
            return (scene_pred_trajectories, scene_gt_trajectories)
        
        # ------------------------------------------------------------------
        # 检测目录结构：是根目录还是具体的 drivepi 目录
        # ------------------------------------------------------------------
        timestamp_dirs = []
        
        # 检查是否是根目录（包含多个时间戳目录）
        # 例如：/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/
        if val_data_path.name != "drivepi":
            # 可能是根目录，查找所有时间戳目录下的 drivepi 子目录
            for item in val_data_path.iterdir():
                if item.is_dir():
                    # 检查是否有 drivepi 子目录
                    drivepi_dir = item / "drivepi"
                    if drivepi_dir.exists() and drivepi_dir.is_dir():
                        timestamp_dirs.append((item.name, drivepi_dir))
        else:
            # 已经是 drivepi 目录，直接处理
            # 获取父目录名作为时间戳名称
            timestamp_name = val_data_path.parent.name
            timestamp_dirs.append((timestamp_name, val_data_path))
        
        if not timestamp_dirs:
            print(f"[WARNING] No valid timestamp directories found in {val_data_path}")
            return
        
        print(f"Found {len(timestamp_dirs)} timestamp directories to process")
        
        # 遍历每个时间戳目录（类似 inference.py 遍历场景）
        for timestamp_name, pkl_dir in tqdm(timestamp_dirs, desc="Processing timestamps"):
            result = process_pkl_directory(pkl_dir, timestamp_name)
            
            if result is None:
                print(f"[WARNING] No valid trajectories for timestamp: {timestamp_name}")
                continue
            
            scene_pred_trajectories, scene_gt_trajectories = result
            
            # 合并该时间戳的所有轨迹 [N, horizon, 2]
            scene_pred_array = np.stack(scene_pred_trajectories, axis=0)
            scene_gt_array = np.stack(scene_gt_trajectories, axis=0)
            
            # 保存为 npy 文件（与 inference.py 的命名方式一致）
            pred_save_path = output_path / f"{timestamp_name}_pred_trajectories.npy"
            gt_save_path = output_path / f"{timestamp_name}_gt_trajectories.npy"
            
            np.save(pred_save_path, scene_pred_array)
            np.save(gt_save_path, scene_gt_array)
            
            print(f"Saved {len(scene_pred_trajectories)} trajectories for timestamp: {timestamp_name}")
            print(f"  - Predicted: {pred_save_path} (shape: {scene_pred_array.shape})")
            print(f"  - GT: {gt_save_path} (shape: {scene_gt_array.shape})")


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
    parser.add_argument("--val_data_dir", type=str, default="/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637/drivepi",
                        help="Path to val data directory containing numeric pkl files (e.g., 00001.pkl, 00002.pkl)")
    parser.add_argument("--config_path", type=str, default="config/eval/DrivePi0/base.yaml",
                        help="Path to config file")
    parser.add_argument("--statistic_path", type=str, default="statistics.json",
                        help="Path to statistics file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: logs/inference/drive-pi0/TIMESTAMP)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for debugging, None means process all)")
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
        
        # 兼容不同的数据格式
        def extract_image_path(img_ref):
            """从不同格式的图像引用中提取路径字符串"""
            # 处理 dict 类型（StringWrapper 序列化为 dict 的情况）
            if isinstance(img_ref, dict) and '_string_wrapper_marker' in img_ref:
                img_ref = img_ref.get('_bytes_value', b'')
            # 处理有 .numpy() 方法的对象
            if hasattr(img_ref, "numpy"):
                img_ref = img_ref.numpy()
            # 处理 bytes 或 str
            if isinstance(img_ref, bytes):
                return img_ref.decode('utf-8')
            elif isinstance(img_ref, str):
                return img_ref
            else:
                raise ValueError(f"Unsupported type for image reference: {type(img_ref)}")
        
        pkl_data_path = args.test_data_path
        with open(pkl_data_path, 'rb') as f:
            # 兼容 Python 2.7 生成的 pickle 文件
            try:
                all_data = pickle.load(f, encoding='latin-1')
            except TypeError:
                # 如果是 Python 3 生成的 pickle，不需要 encoding 参数
                all_data = pickle.load(f)
            normalize = Normalize.get_instance(statistic_path, norm_mode=inference_agent.norm_mode)
            state = normalize.prepare_state(all_data, nav_onehot=inference_agent.nav_onehot)
            
            data_dict = {}
            image_front_path = extract_image_path(all_data['his_cam_front'])
            data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
            image_front_time_path = extract_image_path(all_data['his_cam_front_time'])
            data_dict['image_front_history'] = image_normalization(image_front_time_path, (224, 224))
            data_dict["language_instruction"] = "predict trajectory"
            data_dict['state'] = state
            
            trajectory = inference_agent.run(data_dict=data_dict)
            x_pred, y_pred = normalize.infer_traj(trajectory)
            pred_traj_ego = np.stack([x_pred, y_pred], axis=1)
            
            print('x:', pred_traj_ego[:, 0])
            print('y:', pred_traj_ego[:, 1])
    
    elif args.mode == "batch":
        # 批量推理模式（按场景）
        print(f"Starting batch inference...")
        print(f"Val data directory: {args.val_data_dir}")
        print(f"Statistics file: {statistic_path}")
        print(f"Output directory: {output_dir}")
        
        inference_agent.run_batch_inference(
            val_data_dir=args.val_data_dir,
            output_dir=str(output_dir),
            statistic_path=statistic_path,
            max_samples=args.max_samples
        )
        
        print(f"Batch inference completed. Results saved to: {output_dir}")