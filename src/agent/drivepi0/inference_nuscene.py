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
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)

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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        # 保存 norm_mode（从顶层或 data 中读取）
        self.norm_mode = cfg.data.get("norm_mode", "percentile")
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
        log.info(f"Loaded model from {path}")
    
    def run_batch_inference(self, val_data_dir: str, output_dir: str, statistic_path: str = "statistics.json", scene_name_filter: str = None):
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
        
        # 使用 self.norm_mode 创建归一化器
        norm_mode = getattr(self, 'norm_mode', 'percentile')
        normalize = Normalize.get_instance(statistic_path, norm_mode=norm_mode)
        
        # ------------------------------------------------------------------
        # 自动从 val_data_dir 中推断场景前缀：
        # 形如 "scene-0003_step0.pkl" -> 场景名为 "scene-0003"
        # ------------------------------------------------------------------
        scene_to_files = {}
        for pkl_file in val_data_path.glob("*.pkl"):
            match = re.match(r"(.+)_step(\d+)\.pkl$", pkl_file.name)
            if not match:
                continue
            scene_name = match.group(1)
            step_num = int(match.group(2))
            scene_to_files.setdefault(scene_name, []).append((step_num, pkl_file))
        
        if not scene_to_files:
            log.warning(f"No valid '*_step*.pkl' files found in {val_data_path}")
            return
        
        # 根据可选的 scene_name_filter 决定要处理的场景
        all_scene_names = sorted(scene_to_files.keys())
        if scene_name_filter is not None:
            if scene_name_filter not in scene_to_files:
                log.warning(f"Scene '{scene_name_filter}' not found in {val_data_path}. "
                            f"Available scenes: {all_scene_names}")
                return
            target_scenes = [scene_name_filter]
            log.info(f"Filtering to single scene: {scene_name_filter}")
        else:
            target_scenes = all_scene_names
        
        # 遍历按名字排序后的场景（或单个指定场景）
        for scene_name in tqdm(target_scenes, desc="Processing scenes"):
            scene_files = sorted(scene_to_files[scene_name], key=lambda x: x[0])
            
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
                    
                    # 获取真实轨迹：直接从原始数据读取，不经过归一化→反归一化
                    # 避免因 norm_mode（如 percentile_clip）导致的信息丢失
                    fur_x_gt = all_data["fur_x"]
                    fur_y_gt = all_data["fur_y"]
                    
                    # 转换为 numpy 数组（如果还不是）
                    if isinstance(fur_x_gt, np.ndarray):
                        fur_x_gt_arr = fur_x_gt
                        fur_y_gt_arr = fur_y_gt
                    else:
                        fur_x_gt_arr = np.array(fur_x_gt)
                        fur_y_gt_arr = np.array(fur_y_gt)
                    
                    # 处理形状：如果是 [T, horizon]，取最后一个时间步
                    if fur_x_gt_arr.ndim == 2:
                        fur_x_gt_last = fur_x_gt_arr[-1]  # [horizon]
                        fur_y_gt_last = fur_y_gt_arr[-1]  # [horizon]
                    else:
                        fur_x_gt_last = fur_x_gt_arr  # [horizon]
                        fur_y_gt_last = fur_y_gt_arr  # [horizon]
                    
                    # 拼接成 [horizon, 2] 形状
                    gt_traj_real = np.stack([fur_x_gt_last, fur_y_gt_last], axis=-1)  # [horizon, 2]
                    
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
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Only run inference on a single scene name (e.g. 'scene-0003')")
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
            normalize = Normalize.get_instance(statistic_path, norm_mode=inference_agent.norm_mode)
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
        if args.scene_name is not None:
            log.info(f"Only infer scene: {args.scene_name}")
        
        inference_agent.run_batch_inference(
            val_data_dir=args.val_data_dir,
            output_dir=str(output_dir),
            statistic_path=statistic_path,
            scene_name_filter=args.scene_name
        )
        
        log.info(f"Batch inference completed. Results saved to: {output_dir}")