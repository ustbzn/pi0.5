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
    
    def run_batch_inference(self, val_data_dir: str, output_dir: str, statistic_path: str = "statistics.json"):
        """
        批量推理，处理数字命名的 pkl 文件（如 00001.pkl, 00002.pkl 等）
        
        Args:
            val_data_dir: val 数据目录路径（包含所有 pkl 文件）
            output_dir: 输出目录路径
            statistic_path: 统计文件路径，用于归一化
        """
        from src.data.utils.normalization import Normalize
        from src.data.utils.image import image_normalization
        import copyreg
        
        # 注册 StringWrapper 的还原函数，确保可以加载转换后的 pkl 文件
        def _reconstruct_string_wrapper(bytes_value):
            """还原函数，用于 pickle 加载"""
            class CompatibleWrapper:
                def __init__(self, bv):
                    self._bytes_value = bv
                def numpy(self):
                    return self._bytes_value
            return CompatibleWrapper(bytes_value)
        
        # 注册还原函数到 copyreg（用于 pickle 加载时自动还原）
        # 由于 StringWrapper 类型可能不在当前命名空间，我们通过函数名注册
        # 这样 pickle 在遇到 StringWrapper 时会使用这个函数
        try:
            # 使用一个临时类型来注册，实际使用时 pickle 会调用还原函数
            import types
            temp_type = type('StringWrapper', (object,), {})
            copyreg.pickle(temp_type, lambda obj: (_reconstruct_string_wrapper, (obj._bytes_value,)))
        except:
            pass
        
        val_data_path = Path(val_data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        normalize = Normalize.get_instance(statistic_path)
        
        # ------------------------------------------------------------------
        # 读取数字命名的 pkl 文件：00001.pkl, 00002.pkl, ... 00011.pkl
        # ------------------------------------------------------------------
        pkl_files = []
        for pkl_file in val_data_path.glob("*.pkl"):
            # 匹配数字命名的文件（如 00001.pkl）
            match = re.match(r"(\d+)\.pkl$", pkl_file.name)
            if match:
                file_num = int(match.group(1))
                pkl_files.append((file_num, pkl_file))
        
        # 按文件编号排序
        pkl_files = sorted(pkl_files, key=lambda x: x[0])
        
        if not pkl_files:
            log.warning(f"No valid numeric '*.pkl' files found in {val_data_path}")
            return
        
        log.info(f"Found {len(pkl_files)} pkl files to process")
        
        # 存储所有轨迹
        all_pred_trajectories = []
        all_gt_trajectories = []
        
        # 处理每个文件
        for file_num, pkl_file in tqdm(pkl_files, desc="Processing files"):
            try:
                # 加载 pkl 文件（兼容 Python 2.7 生成的 pickle 文件）
                with open(pkl_file, 'rb') as f:
                    import pickle
                    try:
                        all_data = pickle.load(f, encoding='latin-1')
                    except TypeError:
                        # 如果是 Python 3 生成的 pickle，不需要 encoding 参数
                        all_data = pickle.load(f)
                
                # 准备数据
                state = normalize.prepare_state(all_data)
                data_dict = {}
                
                # 处理图像（兼容不同的数据格式）
                def extract_image_path(img_ref):
                    """从不同格式的图像引用中提取路径字符串"""
                    if hasattr(img_ref, "numpy"):
                        img_ref = img_ref.numpy()
                    if isinstance(img_ref, bytes):
                        return img_ref.decode('utf-8')
                    elif isinstance(img_ref, str):
                        return img_ref
                    else:
                        raise ValueError(f"Unsupported type for image reference: {type(img_ref)}")
                
                image_front_path = extract_image_path(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                # image_back_path = extract_image_path(all_data['his_cam_back'])
                # data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                image_front_time_path = extract_image_path(all_data['his_cam_front_time'])
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
                all_pred_trajectories.append(pred_traj_real.cpu().numpy())
                all_gt_trajectories.append(gt_traj_real.cpu().numpy())
                
            except Exception as e:
                log.error(f"Error processing {pkl_file.name}: {e}")
                import traceback
                log.error(traceback.format_exc())
                continue
        
        if len(all_pred_trajectories) == 0:
            log.warning(f"No valid trajectories processed")
            return
        
        # 合并所有轨迹 [N, horizon, 2]
        all_pred_array = np.stack(all_pred_trajectories, axis=0)
        all_gt_array = np.stack(all_gt_trajectories, axis=0)
        
        # 使用输入目录名作为标识，如果没有则使用默认名称
        dataset_name = val_data_path.name if val_data_path.name else "mogo_data"
        
        # 保存为 npy 文件
        pred_save_path = output_path / f"{dataset_name}_pred_trajectories.npy"
        gt_save_path = output_path / f"{dataset_name}_gt_trajectories.npy"
        
        np.save(pred_save_path, all_pred_array)
        np.save(gt_save_path, all_gt_array)
        
        log.info(f"Saved {len(all_pred_trajectories)} trajectories")
        log.info(f"  - Predicted: {pred_save_path} (shape: {all_pred_array.shape})")
        log.info(f"  - GT: {gt_save_path} (shape: {all_gt_array.shape})")


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
            if hasattr(img_ref, "numpy"):
                img_ref = img_ref.numpy()
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
            # prepare state
            normalize = Normalize.get_instance(statistic_path)
            state = normalize.prepare_state(all_data)
            # prepare 
            data_dict = {}
            image_front_path = extract_image_path(all_data['his_cam_front'])
            data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
            # image_back_path = extract_image_path(all_data['his_cam_back'])
            # data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
            image_front_time_path = extract_image_path(all_data['his_cam_front_time'])
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