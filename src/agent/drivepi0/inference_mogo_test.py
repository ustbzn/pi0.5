"""
    用于推理 MOGO 数据集的 DrivePi-0 模型:
    python src/agent/drivepi0/inference_mogo_test.py --mode batch \ 
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

from PIL import Image
import numpy as np
from transformers import Qwen3VLProcessor

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
        self.model = DrivePiZeroInference(cfg, use_ddp=False, load_pretrained=False)
        self.model.tie_action_proprio_weights()
        if cfg.get("lora", False):
            self.model.inject_lora(cfg)
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

        self.state_only_for_action = cfg.get("state_only_for_action", False)
        if self.state_only_for_action:
            self.tokenizer = None
            self.processor = None
            self.max_seq_len = None
            self.tokenizer_padding = None
            log.info("Strict state-only enabled for inference; skipping Qwen3VLProcessor initialization.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.pretrained_model_path, padding_side="right"
            )
            self.processor = Qwen3VLProcessor.from_pretrained(
                cfg.pretrained_model_path,
                trust_remote_code=True,
            )
            # 保存 padding 参数，确保推理时 input_ids 长度与 mask 一致
            self.max_seq_len = cfg.max_seq_len
            self.tokenizer_padding = cfg.get("tokenizer_padding", "max_length")
        # 保存归一化模式（优先从 data 中读取）
        self.norm_mode = cfg.get("norm_mode") or cfg.data.get("norm_mode", "percentile")
        self.nav_onehot = cfg.data.get("nav_onehot", False)
        self.use_his_trajectory = cfg.data.get("use_his_trajectory", False)
        self.state_frame_index = cfg.data.get("state_frame_index", None)

        # HDMap 开关
        self.use_hdmap_image = cfg.data.get("use_hdmap_image", False)
        self.use_hdmap_proprio = cfg.data.get("use_hdmap_proprio", False)
        self.use_hdmap_cnn = cfg.data.get("use_hdmap_cnn", False)
        # 图像尺寸
        self.cam_image_size = tuple(cfg.data.get("cam_image_size", [224, 224]))
        self.hdmap_image_size = tuple(cfg.data.get("hdmap_image_size", [352, 192]))
        self.hdmap_bev_downsample_size = tuple(cfg.data.get("hdmap_bev_downsample_size", [48, 90]))
        self.hdmap_downsample_mode = cfg.data.get("hdmap_downsample_mode", None)
        self.hdmap_channel_scale = cfg.data.get("hdmap_channel_scale", None)
        print('Agent done')
    
    
    def run(self, data_dict):
        state = data_dict["state"].unsqueeze(0)
        batch_size = state.shape[0]

        if self.state_only_for_action:
            (
                causal_mask,
                vlm_position_ids,
                state_position_ids,
                action_position_ids,
            ) = self.model.build_state_only_causal_mask_and_position_ids(
                batch_size=batch_size,
                device=state.device,
                dtype=self.dtype,
            )

            image_text_proprio_mask, action_mask = (
                self.model.split_full_mask_into_submasks(causal_mask)
            )

            inputs = {
                "input_ids": torch.empty((batch_size, 0), dtype=torch.long),
                "pixel_values": torch.empty((0,), dtype=self.dtype),
                "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": state_position_ids,
                "action_position_ids": action_position_ids,
                "proprios": state.to(self.dtype),
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            trajectory = self.model.infer_action(**inputs)
            return trajectory

        # 4 cameras (front, right, left, back) x 2 time steps (current + history) = 8 images
        cam_names = ['front', 'right', 'left', 'back']
        cam_labels = [
            'current front view', 'current right view',
            'current left view', 'current rear view',
            'previous front view', 'previous right view',
            'previous left view', 'previous rear view',
        ]

        image_list = []
        for cam in cam_names:
            img = data_dict[f"image_{cam}"].unsqueeze(0)  # [1, H, W, C]
            img = einops.rearrange(img, "B H W C -> B C H W")
            image_list.append(img.unsqueeze(1))  # [1, 1, C, H, W]
        for cam in cam_names:
            img = data_dict[f"image_{cam}_time"].unsqueeze(0)  # [1, H, W, C]
            img = einops.rearrange(img, "B H W C -> B C H W")
            image_list.append(img.unsqueeze(1))  # [1, 1, C, H, W]

        # 方案 A: HDMap BEV 作为额外图像
        if self.use_hdmap_image and "image_hdmap_bev" in data_dict:
            hdmap_img = data_dict["image_hdmap_bev"].unsqueeze(0)  # [1, H, W, C]
            hdmap_img = einops.rearrange(hdmap_img, "B H W C -> B C H W")
            image_list.append(hdmap_img.unsqueeze(1))
            cam_labels.append('HDMap BEV view')

        batch_size = 1
        num_images_per_sample = len(image_list)
        state = data_dict["state"].unsqueeze(0)
        texts = [data_dict["language_instruction"]]

        # 转换为 PIL Image 列表（支持不同尺寸的图像）
        pil_images = []
        for img_tensor in image_list:
            img = Image.fromarray(
                img_tensor[0, 0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            pil_images.append(img)

        # 构建 Qwen3VL 聊天格式（与训练一致，每张图像带语义标签）
        chat_texts = []
        for i, text in enumerate(texts):
            content = []
            for label in cam_labels:
                content.append({"type": "text", "text": f"{label}:"})
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            chat_texts.append(chat_text)

        model_inputs = self.processor(
            text=chat_texts,
            images=pil_images,
            return_tensors="pt",
            padding=self.tokenizer_padding,
            max_length=self.max_seq_len,
            truncation=True,
        )

        # build causal mask and position ids for trajectory
        causal_mask, _, state_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )
        # P0-A: 使用官方 3D M-RoPE position_ids 替代 1D arange
        vlm_position_ids = self.model.build_qwen3_vl_position_ids(
            model_inputs["input_ids"],
            model_inputs["image_grid_thw"],
            model_inputs["attention_mask"],
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "image_grid_thw": model_inputs["image_grid_thw"],
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": state_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": state.to(self.dtype),
        }

        # 方案 C: 传递原始 BEV 数据
        if self.use_hdmap_cnn and "hdmap_bev_raw" in data_dict:
            inputs["hdmap_bev"] = data_dict["hdmap_bev_raw"].unsqueeze(0).to(self.device, dtype=self.dtype)
        if self.use_hdmap_cnn and "hdmap_valid_mask" in data_dict:
            inputs["hdmap_valid_mask"] = data_dict["hdmap_valid_mask"].reshape(1).to(self.device, dtype=self.dtype)

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
        """从训练 checkpoint 加载完整模型权重"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        ckpt = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }

        model_state = self.model.state_dict()
        filtered_ckpt = {}
        skipped = []
        for k, v in ckpt.items():
            if k not in model_state:
                skipped.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped.append(f"[shape mismatch] {k}")
                continue
            filtered_ckpt[k] = v

        if skipped:
            # proprio 的 key 因 tie_weights 被合并到 action，属于正常跳过
            real_skipped = [s for s in skipped if "mixtures.proprio" not in s]
            if real_skipped:
                print(f"[WARNING] Skipped {len(real_skipped)} unexpected keys: {real_skipped[:5]}")

        missing, _ = self.model.load_state_dict(filtered_ckpt, strict=False)
        if missing:
            print(f"[WARNING] Missing {len(missing)} keys in checkpoint")
            for k in missing[:10]:
                print(f"  {k}")

        print(f"Loaded {len(filtered_ckpt)} / {len(model_state)} keys from {path}")
    
    def run_batch_inference(self, val_data_dir: str, output_dir: str, statistic_path: str = "statistics.json", max_samples: Optional[int] = None, rank: int = 0, world_size: int = 1):
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
        from src.data.utils.hdmap_features import extract_hdmap_features, load_hdmap_image, load_hdmap_bev_downsampled, HDMAP_PROPRIO_DIM, HDMAP_BEV_CHANNELS
        
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
            load_data = json.load(f)
        if "percentile" in load_data:
            pct = load_data["percentile"]
            mm = load_data.get("minmax", {})
        else:
            pct = load_data
            mm = {}
        
        if norm_mode == "minmax":
            stats_data = mm
        else:
            stats_data = pct
            
        print(f"\n{'='*60}")
        print(f"归一化参数 (statistics.json) [norm_mode={norm_mode}]:")
        print(f"{'='*60}")
        for key, val in stats_data.items():
            try:
                print(f"  {key}: [{val[0]:.6f}, {val[1]:.6f}]")
            except Exception:
                print(f"  {key}: {val}")
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
        def process_pkl_directory(pkl_dir: Path, timestamp_name: str, timestamp_filter: Optional[str] = None):
            """
            处理单个 pkl 目录下的所有文件
            
            Args:
                pkl_dir: 包含 pkl 文件的目录路径
                timestamp_name: 时间戳名称（用于生成输出文件名）
                timestamp_filter: 如果指定，只处理匹配该时间戳的文件（用于扁平化目录结构）
            
            Returns:
                (pred_trajectories, gt_trajectories) 列表，如果处理失败返回 None
            """
            # 读取 pkl 文件，支持多种命名格式：
            # 1. 数字命名：00001.pkl, 00002.pkl, ...
            # 2. 时间戳_stepN格式：20251210141637_step1.pkl, 20251210141637_step2.pkl, ...
            pkl_files = []
            for pkl_file in pkl_dir.glob("*.pkl"):
                # 如果指定了时间戳过滤，先检查是否匹配
                if timestamp_filter is not None:
                    if not pkl_file.name.startswith(timestamp_filter + "_step"):
                        continue
                
                # 尝试匹配数字命名的文件（如 00001.pkl）
                match_num = re.match(r"(\d+)\.pkl$", pkl_file.name)
                if match_num:
                    file_num = int(match_num.group(1))
                    pkl_files.append((file_num, pkl_file))
                else:
                    # 尝试匹配时间戳_stepN格式（如 20251210141637_step1.pkl）
                    match_timestamp = re.match(r"(\d+)_step(\d+)\.pkl$", pkl_file.name)
                    if match_timestamp:
                        step_num = int(match_timestamp.group(2))
                        pkl_files.append((step_num, pkl_file))
            
            # 按文件编号排序
            pkl_files = sorted(pkl_files, key=lambda x: x[0])
            
            if not pkl_files:
                if timestamp_filter:
                    print(f"[WARNING] No valid '*.pkl' files found in {pkl_dir} matching timestamp {timestamp_filter}")
                else:
                    print(f"[WARNING] No valid '*.pkl' files found in {pkl_dir}")
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
                    # 与 dataset.py 保持一致：若无 navigation_id，从 navigation_command 推断并填充
                    if 'navigation_id' not in all_data:
                        from src.data.dataset import build_navigation_id_sequence
                        target_len = np.asarray(all_data['his_theta']).shape[0]
                        all_data = dict(all_data)
                        all_data['navigation_id'] = build_navigation_id_sequence(all_data, target_len)

                    # 方案 B: 提取 hdmap 特征
                    if self.use_hdmap_proprio:
                        hdmap_bev_path = all_data.get('hdmap_bev')
                        if hdmap_bev_path is not None:
                            all_data['hdmap_features'] = extract_hdmap_features(hdmap_bev_path)
                        else:
                            all_data['hdmap_features'] = np.zeros(HDMAP_PROPRIO_DIM, dtype=np.float32)

                    state = normalize.prepare_state(
                        all_data,
                        nav_onehot=self.nav_onehot,
                        state_frame_index=self.state_frame_index,
                        use_his_trajectory=self.use_his_trajectory,
                        use_hdmap_proprio=self.use_hdmap_proprio,
                    )

                    data_dict = {}
                    # 4 cameras x 2 time steps = 8 images（与训练一致）
                    # 新数据格式：当前帧 cam_front / cam_left / cam_right / cam_back
                    #             上一帧  cam_front_prev_1 / cam_left_prev_1 / cam_right_prev_1 / cam_back_prev_1
                    cam_key_map = {
                        'front': ('cam_front',      'cam_front_prev_1'),
                        'right': ('cam_right',      'cam_right_prev_1'),
                        'left':  ('cam_left',       'cam_left_prev_1'),
                        'back':  ('cam_back',       'cam_back_prev_1'),
                    }
                    for cam_name, (cur_key, prev_key) in cam_key_map.items():
                        cam_path = extract_image_path(all_data[cur_key])
                        data_dict[f'image_{cam_name}'] = image_normalization(cam_path, self.cam_image_size)
                        cam_time_path = extract_image_path(all_data[prev_key])
                        data_dict[f'image_{cam_name}_time'] = image_normalization(cam_time_path, self.cam_image_size)

                    # 方案 A: 加载 HDMap BEV 图像
                    if self.use_hdmap_image:
                        hdmap_bev_path = all_data.get('hdmap_bev')
                        if hdmap_bev_path is not None:
                            hdmap_img = load_hdmap_image(hdmap_bev_path, size=self.hdmap_image_size)
                            if hdmap_img is not None:
                                data_dict['image_hdmap_bev'] = torch.as_tensor(hdmap_img, dtype=torch.uint8)
                            else:
                                h, w = self.hdmap_image_size
                                data_dict['image_hdmap_bev'] = torch.zeros((h, w, 3), dtype=torch.uint8)
                        else:
                            h, w = self.hdmap_image_size
                            data_dict['image_hdmap_bev'] = torch.zeros((h, w, 3), dtype=torch.uint8)

                    # 方案 C: 加载下采样 BEV 数据
                    if self.use_hdmap_cnn:
                        hdmap_bev_path = all_data.get('hdmap_bev')
                        if hdmap_bev_path is not None:
                            bev_ds, valid_mask = load_hdmap_bev_downsampled(
                                hdmap_bev_path,
                                size=self.hdmap_bev_downsample_size,
                                downsample_mode=self.hdmap_downsample_mode,
                                channel_scale=self.hdmap_channel_scale,
                                return_valid_mask=True,
                            )
                            if bev_ds is not None:
                                data_dict['hdmap_bev_raw'] = torch.as_tensor(bev_ds, dtype=torch.float32)
                                data_dict['hdmap_valid_mask'] = torch.tensor(valid_mask, dtype=torch.float32)
                            else:
                                h, w = self.hdmap_bev_downsample_size
                                data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)
                                data_dict['hdmap_valid_mask'] = torch.tensor(0.0, dtype=torch.float32)
                        else:
                            h, w = self.hdmap_bev_downsample_size
                            data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)
                            data_dict['hdmap_valid_mask'] = torch.tensor(0.0, dtype=torch.float32)

                    hdmap_context = ""
                    if self.use_hdmap_image:
                        hdmap_context += "An HD map bird's-eye view is provided showing lane layout, junctions, and road boundaries. "
                    if self.use_hdmap_proprio or self.use_hdmap_cnn:
                        hdmap_context += "Road structure features (lane density, junction proximity, lane boundaries) are encoded in the vehicle state. "

                    nav_desc = all_data.get('navigation_command_descreption')
                    if nav_desc is not None:
                        nav_desc = np.asarray(nav_desc).reshape(-1)
                        if hdmap_context:
                            data_dict['language_instruction'] = f"{hdmap_context}Predict trajectory based on the HD map and navigation command: {str(nav_desc[0])}"
                        else:
                            data_dict['language_instruction'] = f"Predict trajectory based on the navigation command: {str(nav_desc[0])}"
                    else:
                        if hdmap_context:
                            data_dict["language_instruction"] = f"{hdmap_context}Predict trajectory considering the road structure."
                        else:
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
                        pred_traj_real = pred_traj_real.float().cpu().numpy()
                    
                    # 获取真实轨迹
                    # 新数据格式：fur_trajectory ndarray [N, 2]，存储顺序为 [y, x]
                    if "fur_trajectory" in all_data:
                        fur_raw = np.asarray(all_data["fur_trajectory"], dtype=np.float32)  # [N, 2] (y, x)
                        gt_traj_real = np.stack([fur_raw[:, 1], fur_raw[:, 0]], axis=-1).astype(np.float32)  # 转为 [N, 2] (x, y)
                    else:
                        # 兼容旧格式 fur_x / fur_y
                        fur_x_gt = all_data["fur_x"]
                        fur_y_gt = all_data["fur_y"]
                        fur_x_gt_arr = np.array(fur_x_gt) if not isinstance(fur_x_gt, np.ndarray) else fur_x_gt
                        fur_y_gt_arr = np.array(fur_y_gt) if not isinstance(fur_y_gt, np.ndarray) else fur_y_gt
                        if fur_x_gt_arr.ndim == 2:
                            fur_x_gt_arr = fur_x_gt_arr[-1]
                            fur_y_gt_arr = fur_y_gt_arr[-1]
                        gt_traj_real = np.stack([fur_x_gt_arr, fur_y_gt_arr], axis=-1).astype(np.float32)
                    
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
        # 检测目录结构：支持多种目录结构
        # 1. 根目录（包含多个时间戳目录，每个有 drivepi 子目录）
        # 2. drivepi 目录（直接包含 pkl 文件）
        # 3. 扁平目录（直接包含 timestamp_stepN.pkl 格式的文件）
        # ------------------------------------------------------------------
        timestamp_dirs = []
        
        # 首先检查目录下是否直接有 pkl 文件（扁平化结构）
        pkl_files_in_dir = list(val_data_path.glob("*.pkl"))
        is_flat_structure = len(pkl_files_in_dir) > 0
        
        if is_flat_structure:
            # 扁平化结构：直接包含 pkl 文件
            # 按时间戳前缀分组（如果有多个时间戳）
            timestamp_groups = {}
            for pkl_file in pkl_files_in_dir:
                # 尝试匹配时间戳_stepN格式
                match = re.match(r"(\d+)_step(\d+)\.pkl$", pkl_file.name)
                if match:
                    timestamp = match.group(1)
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = []
                    timestamp_groups[timestamp].append(pkl_file)
                else:
                    # 如果不是时间戳格式，使用目录名作为时间戳
                    timestamp = val_data_path.name if val_data_path.name else "default"
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = []
                    timestamp_groups[timestamp].append(pkl_file)
            
            # 为每个时间戳创建一个条目（使用原目录和时间戳）
            for timestamp, files in timestamp_groups.items():
                timestamp_dirs.append((timestamp, val_data_path, True))  # 第三个参数表示是扁平化结构
            print(f"Found {len(timestamp_groups)} timestamp groups in flat directory structure")
        
        # 如果不是扁平化结构，检查是否是根目录或 drivepi 目录
        elif val_data_path.name != "drivepi":
            # 可能是根目录，查找所有时间戳目录下的 drivepi 子目录
            for item in val_data_path.iterdir():
                if item.is_dir():
                    # 检查是否有 drivepi 子目录
                    drivepi_dir = item / "drivepi"
                    if drivepi_dir.exists() and drivepi_dir.is_dir():
                        timestamp_dirs.append((item.name, drivepi_dir, False))  # 第三个参数表示不是扁平化结构
        else:
            # 已经是 drivepi 目录，直接处理
            # 获取父目录名作为时间戳名称
            timestamp_name = val_data_path.parent.name
            timestamp_dirs.append((timestamp_name, val_data_path, False))  # 不是扁平化结构
        
        if not timestamp_dirs:
            print(f"[WARNING] No valid timestamp directories or pkl files found in {val_data_path}")
            return

        # 多卡数据分片：按 rank 划分 timestamp_dirs
        total = len(timestamp_dirs)
        if world_size > 1:
            # 将所有任务均匀分配给各进程
            timestamp_dirs = timestamp_dirs[rank::world_size]
            print(f"[Multi-GPU] rank={rank}/{world_size}: 负责 {len(timestamp_dirs)}/{total} 个 timestamp 目录")

        print(f"Found {len(timestamp_dirs)} timestamp directories/groups to process")
        
        # 遍历每个时间戳目录（类似 inference.py 遍历场景）
        for timestamp_info in tqdm(timestamp_dirs, desc="Processing timestamps"):
            if len(timestamp_info) == 3:
                timestamp_name, pkl_dir, is_flat = timestamp_info
            else:
                # 兼容旧格式（没有 is_flat 标志）
                timestamp_name, pkl_dir = timestamp_info
                is_flat = any(pkl_dir.glob("*_step*.pkl"))
            
            # 对于扁平化目录结构，传递时间戳过滤器
            timestamp_filter = timestamp_name if is_flat else None
            
            result = process_pkl_directory(pkl_dir, timestamp_name, timestamp_filter=timestamp_filter)
            
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
    from src.data.utils.hdmap_features import extract_hdmap_features, load_hdmap_image, load_hdmap_bev_downsampled, HDMAP_PROPRIO_DIM, HDMAP_BEV_CHANNELS
    
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
    parser.add_argument("--rank", type=int, default=0,
                        help="当前进程的 rank（0-indexed），用于多卡并行推理")
    parser.add_argument("--world_size", type=int, default=1,
                        help="总进程数（GPU 数量），用于多卡并行推理")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)
    statistic_path = args.statistic_path

    # 多卡推理：每个进程通过 CUDA_VISIBLE_DEVICES=X 隔离 GPU，
    # gpu_id 统一设为 0（进程内唯一的 GPU）
    if args.world_size > 1:
        OmegaConf.update(config, "gpu_id", 0, merge=True)
        print(f"[Multi-GPU] rank={args.rank}/{args.world_size}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
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

            # 与 dataset.py 保持一致
            if 'navigation_id' not in all_data:
                from src.data.dataset import build_navigation_id_sequence
                target_len = np.asarray(all_data['his_theta']).shape[0]
                all_data = dict(all_data)
                all_data['navigation_id'] = build_navigation_id_sequence(all_data, target_len)

            # 方案 B: 提取 hdmap 特征
            if inference_agent.use_hdmap_proprio:
                hdmap_bev_path = all_data.get('hdmap_bev')
                if hdmap_bev_path is not None:
                    all_data['hdmap_features'] = extract_hdmap_features(hdmap_bev_path)
                else:
                    all_data['hdmap_features'] = np.zeros(HDMAP_PROPRIO_DIM, dtype=np.float32)

            state = normalize.prepare_state(
                all_data,
                nav_onehot=inference_agent.nav_onehot,
                use_his_trajectory=inference_agent.use_his_trajectory,
                use_hdmap_proprio=inference_agent.use_hdmap_proprio,
                state_frame_index=getattr(inference_agent, "state_frame_index", None),
            )

            data_dict = {}
            # 4 cameras x 2 time steps = 8 images（与训练一致）
            cam_key_map = {
                'front': ('cam_front', 'cam_front_prev_1'),
                'right': ('cam_right', 'cam_right_prev_1'),
                'left':  ('cam_left',  'cam_left_prev_1'),
                'back':  ('cam_back',  'cam_back_prev_1'),
            }
            for cam_name, (cur_key, prev_key) in cam_key_map.items():
                # 兼容新旧格式
                cur_key_fallback = f'his_cam_{cam_name}'
                prev_key_fallback = f'his_cam_{cam_name}_time'
                cur = cur_key if cur_key in all_data else cur_key_fallback
                prev = prev_key if prev_key in all_data else prev_key_fallback
                cam_path = extract_image_path(all_data[cur])
                data_dict[f'image_{cam_name}'] = image_normalization(cam_path, inference_agent.cam_image_size)
                cam_time_path = extract_image_path(all_data[prev])
                data_dict[f'image_{cam_name}_time'] = image_normalization(cam_time_path, inference_agent.cam_image_size)

            # 方案 A: 加载 HDMap BEV 图像
            if inference_agent.use_hdmap_image:
                hdmap_bev_path = all_data.get('hdmap_bev')
                if hdmap_bev_path is not None:
                    hdmap_img = load_hdmap_image(hdmap_bev_path, size=inference_agent.hdmap_image_size)
                    if hdmap_img is not None:
                        data_dict['image_hdmap_bev'] = torch.as_tensor(hdmap_img, dtype=torch.uint8)
                    else:
                        h, w = inference_agent.hdmap_image_size
                        data_dict['image_hdmap_bev'] = torch.zeros((h, w, 3), dtype=torch.uint8)

            # 方案 C: 加载下采样 BEV 数据
            if inference_agent.use_hdmap_cnn:
                hdmap_bev_path = all_data.get('hdmap_bev')
                if hdmap_bev_path is not None:
                    bev_ds, valid_mask = load_hdmap_bev_downsampled(
                        hdmap_bev_path,
                        size=inference_agent.hdmap_bev_downsample_size,
                        downsample_mode=inference_agent.hdmap_downsample_mode,
                        channel_scale=inference_agent.hdmap_channel_scale,
                        return_valid_mask=True,
                    )
                    if bev_ds is not None:
                        data_dict['hdmap_bev_raw'] = torch.as_tensor(bev_ds, dtype=torch.float32)
                        data_dict['hdmap_valid_mask'] = torch.tensor(valid_mask, dtype=torch.float32)
                    else:
                        h, w = inference_agent.hdmap_bev_downsample_size
                        data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)
                        data_dict['hdmap_valid_mask'] = torch.tensor(0.0, dtype=torch.float32)
                else:
                    h, w = inference_agent.hdmap_bev_downsample_size
                    data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)
                    data_dict['hdmap_valid_mask'] = torch.tensor(0.0, dtype=torch.float32)

            hdmap_context = ""
            if inference_agent.use_hdmap_image:
                hdmap_context += "An HD map bird's-eye view is provided showing lane layout, junctions, and road boundaries. "
            if inference_agent.use_hdmap_proprio or inference_agent.use_hdmap_cnn:
                hdmap_context += "Road structure features (lane density, junction proximity, lane boundaries) are encoded in the vehicle state. "

            if hdmap_context:
                data_dict["language_instruction"] = f"{hdmap_context}Predict trajectory considering the road structure."
            else:
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
            max_samples=args.max_samples,
            rank=args.rank,
            world_size=args.world_size,
        )
        
        print(f"Batch inference completed. Results saved to: {output_dir}")
