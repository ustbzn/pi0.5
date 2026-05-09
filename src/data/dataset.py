from typing import Optional
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from src.data.camera_scenario_map import get_cam_id_by_label, get_cam_id_by_name, get_scenario_id
from src.data.utils.image import image_normalization
from src.data.utils.normalization import Normalize
from src.data.utils.hdmap_features import extract_hdmap_features, load_hdmap_image, load_hdmap_bev_downsampled, HDMAP_PROPRIO_DIM, HDMAP_BEV_CHANNELS



NAVIGATION_COMMAND_TO_ID = {
    "straight": 0,
    "turn left": 1,
    "turn right": 2,
    "u turn": 3,
    "park": 4,
    "none": 0,
}


def infer_navigation_id(all_data):
    nav = all_data.get("navigation_id")
    if nav is not None:
        nav = np.asarray(nav)
        if nav.size > 0:
            return int(nav.reshape(-1)[0])

    cmd = all_data.get("navigation_command")
    if cmd is None:
        return 0
    cmd = np.asarray(cmd)
    if cmd.size == 0:
        return 0
    token = str(cmd.reshape(-1, cmd.shape[-1])[0][0]).strip().lower() if cmd.ndim >= 2 else str(cmd.reshape(-1)[0]).strip().lower()
    return NAVIGATION_COMMAND_TO_ID.get(token, 0)


def build_navigation_id_sequence(all_data, target_len):
    nav_id = infer_navigation_id(all_data)
    return np.full((target_len,), nav_id, dtype=np.float32)

def decode_path_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, dict):
        if '_bytes_value' in value:
            bytes_value = value['_bytes_value']
            if isinstance(bytes_value, bytes):
                return bytes_value.decode('utf-8')
            if isinstance(bytes_value, str):
                return bytes_value
            return str(bytes_value)
        return str(value)
    if hasattr(value, 'numpy'):
        try:
            numpy_value = value.numpy()
            if isinstance(numpy_value, bytes):
                return numpy_value.decode('utf-8')
            if isinstance(numpy_value, str):
                return numpy_value
            if isinstance(numpy_value, np.ndarray) and numpy_value.dtype == np.uint8:
                return numpy_value.tobytes().decode('utf-8')
            return str(numpy_value)
        except Exception:
            return str(value)
    return str(value)


def build_aux_targets(all_data):
    """从 fur_trajectory 自动构造 action-grounded 辅助监督标签。

    生成 5 个辅助目标（方案 A）：
      1. maneuver_cls:  straight(0) / left(1) / right(2) / stop(3) / u_turn_like(4)
      2. stop_cls:      non-stop(0) / stop(1)
      3. endpoint:      future trajectory 末端 (x_end, y_end)
      4. progress:      总位移 ||fur[-1]||
      5. curvature:     平均二阶差分范数（转弯程度）

    同时输出用于方案 D 分桶评估的 trajectory_bucket:
      straight(0) / turn(1) / static(2) / high_curvature(3)

    Args:
        all_data: 包含 'fur_trajectory' 字段的样本字典，
                  fur_trajectory shape = [H, 2]，列顺序 [y, x]
    Returns:
        dict: 包含上述 5 个辅助标签和 trajectory_bucket
    """
    fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)  # [H, 2], [y, x]
    final_y, final_x = float(fur[-1, 0]), float(fur[-1, 1])
    displacement = float(np.sqrt(final_x ** 2 + final_y ** 2))

    # --- curvature: 平均二阶差分范数 ---
    diffs = fur[1:] - fur[:-1]
    curvature = float(np.mean(np.linalg.norm(diffs[1:] - diffs[:-1], axis=-1))) if len(fur) >= 3 else 0.0

    # --- maneuver / stop 分类 ---
    stop = displacement < 1.0
    if stop:
        maneuver = 3  # stop
    elif abs(final_y) > 2.0 and final_y > 0:
        maneuver = 1  # left
    elif abs(final_y) > 2.0 and final_y < 0:
        maneuver = 2  # right
    else:
        maneuver = 0  # straight

    # --- trajectory_bucket: 用于方案 D 分桶 eval ---
    # 0=straight, 1=turn, 2=static, 3=high_curvature
    if stop:
        bucket = 2  # static
    elif abs(final_y) > 2.0:
        bucket = 1  # turn
    elif curvature > 0.3:
        bucket = 3  # high_curvature
    else:
        bucket = 0  # straight

    return {
        "maneuver_cls": maneuver,
        "stop_cls": int(stop),
        "endpoint": np.array([final_x, final_y], dtype=np.float32),
        "progress": np.array([displacement], dtype=np.float32),
        "curvature": np.array([curvature], dtype=np.float32),
        "trajectory_bucket": bucket,
    }


class TorchDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        statistics_path: str,
        use_fixed_images: bool,
        num_of_action_experts: int,
        norm_mode: str = "percentile",
        nav_onehot: bool = False,
        use_his_trajectory: bool = False,
        state_frame_index: Optional[int] = None,
        use_hdmap_image: bool = False, 
        use_hdmap_proprio: bool = False, 
        use_hdmap_cnn: bool = False, 
        cam_image_size: tuple = (224, 224), 
        hdmap_image_size: tuple = (192, 352), 
        hdmap_bev_downsample_size: tuple = (48, 90), 
        hdmap_downsample_mode=None, 
        hdmap_channel_scale=None, 
    ):
        super(TorchDataset, self).__init__()
        self.data_path = data_path
        self.statistics_path = statistics_path
        self.use_fixed_images = use_fixed_images
        self.num_of_action_experts = num_of_action_experts
        self.norm_mode = norm_mode
        self.nav_onehot = nav_onehot
        self.use_his_trajectory = use_his_trajectory
        self.state_frame_index = state_frame_index
        self.use_hdmap_image = use_hdmap_image
        self.use_hdmap_proprio = use_hdmap_proprio
        self.use_hdmap_cnn = use_hdmap_cnn
        self.cam_image_size = tuple(cam_image_size)      # (H, W) 相机图像缩放目标
        self.hdmap_image_size = tuple(hdmap_image_size)   # (H, W) HDMap BEV 缩放目标
        self.hdmap_bev_downsample_size = tuple(hdmap_bev_downsample_size)  # (H, W) 方案 C BEV 下采样目标
        self.hdmap_downsample_mode = hdmap_downsample_mode
        self.hdmap_channel_scale = hdmap_channel_scale
        self.list_data_dict = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.list_data_dict)

    def get_nav_labels(self):
        labels = []
        for fname in self.list_data_dict:
            pkl_path = os.path.join(self.data_path, fname)
            try:
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                labels.append(infer_navigation_id(d))
            except Exception:
                labels.append(0)
        return labels

    def get_trajectory_weights(
        self,
        static_threshold: float = 1.0,
        turn_threshold: float = 2.0,
        static_weight: float = 2.0,
        turn_weight: float = 3.0,
        straight_weight: float = 1.0,
        hard_case_weight_path: str = None,
        hard_case_alpha: float = 1.0,
    ):
        """根据轨迹运动模式为每个样本计算采样权重。

        分类规则（按优先级）：
          1. 转弯: fur_trajectory 末端横向位移 > turn_threshold
          2. 静止: fur_trajectory 末端总位移 < static_threshold
          3. 直行: 其余

        方案 D 扩展: 如果提供了 hard_case_weight_path（state-only 模型
        离线推理得到的逐样本 L1 error），则额外乘以
        (1 + hard_case_alpha * normalized_error)，让 VLM 训练更聚焦
        state-only 模型做不好的样本。

        Args:
            static_threshold: 判定静止的位移阈值 (米)
            turn_threshold:   判定转弯的横向偏移阈值 (米)
            static_weight:    静止样本的采样权重
            turn_weight:      转弯样本的采样权重
            straight_weight:  直行样本的采样权重
            hard_case_weight_path: 可选，state-only 逐样本 error 的 json 路径
            hard_case_alpha:  hard case 权重缩放系数

        Returns:
            weights: list[float], 与 self.list_data_dict 等长的权重列表
            stats:   dict, 各类别的样本计数
        """
        # --- 可选: 加载 state-only hard case scores ---
        hard_scores = {}
        if hard_case_weight_path and os.path.exists(hard_case_weight_path):
            with open(hard_case_weight_path, 'r') as f:
                hard_scores = json.load(f)
            # 归一化到 [0, 1]
            if hard_scores:
                max_score = max(hard_scores.values())
                min_score = min(hard_scores.values())
                score_range = max_score - min_score if max_score > min_score else 1.0
                hard_scores = {k: (v - min_score) / score_range for k, v in hard_scores.items()}

        weights = []
        stats = {"straight": 0, "turn": 0, "static": 0}
        for fname in self.list_data_dict:
            pkl_path = os.path.join(self.data_path, fname)
            try:
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                fur = np.asarray(d["fur_trajectory"], dtype=np.float32)  # (6, 2) [y, x]
                final_y, final_x = fur[-1, 0], fur[-1, 1]
                displacement = np.sqrt(final_y ** 2 + final_x ** 2)
                lateral = abs(final_y)

                # 转弯优先判定（转弯样本位移一般也 > static_threshold）
                if lateral > turn_threshold:
                    w = turn_weight
                    stats["turn"] += 1
                elif displacement < static_threshold:
                    w = static_weight
                    stats["static"] += 1
                else:
                    w = straight_weight
                    stats["straight"] += 1

                # 方案 D: 乘以 hard case 加权
                if fname in hard_scores:
                    w *= (1.0 + hard_case_alpha * hard_scores[fname])
                weights.append(w)
            except Exception:
                weights.append(straight_weight)
                stats["straight"] += 1
        return weights, stats

    @staticmethod
    def _pick_image_path(all_data, *keys):
        for key in keys:
            if key in all_data:
                path = decode_path_value(all_data[key])
                if path and path.strip():
                    return path
        return None

    def __getitem__(self, i):
        data_dict = {}
        pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            normalize = Normalize.get_instance(self.statistics_path, norm_mode=self.norm_mode)
            if 'navigation_id' not in all_data:
                all_data = dict(all_data)
                target_len = np.asarray(all_data['his_theta']).shape[0]
                all_data['navigation_id'] = build_navigation_id_sequence(all_data, target_len)

            # 方案 B: 从 hdmap NPZ 提取标量特征，放入 all_data 供 prepare_state 使用
            if self.use_hdmap_proprio:
                hdmap_bev_path = all_data.get('hdmap_bev')
                if hdmap_bev_path is not None:
                    all_data['hdmap_features'] = extract_hdmap_features(hdmap_bev_path)
                else:
                    all_data['hdmap_features'] = np.zeros(HDMAP_PROPRIO_DIM, dtype=np.float32)

            state = normalize.prepare_state(
                all_data,
                nav_onehot=self.nav_onehot,
                use_his_trajectory=self.use_his_trajectory,
                state_frame_index=self.state_frame_index,
                use_hdmap_proprio=self.use_hdmap_proprio,
            )
            data_dict['state'] = state

            if 'cam_front' in all_data:
                cam_key_map = {
                    'front': ('cam_front', 'cam_front_prev_1'),
                    'right': ('cam_right', 'cam_right_prev_1'),
                    'left': ('cam_left', 'cam_left_prev_1'),
                    'back': ('cam_back', 'cam_back_prev_1'),
                }
                for cam_name, (current_key, history_key) in cam_key_map.items():
                    path = self._pick_image_path(all_data, current_key)
                    path_time = self._pick_image_path(all_data, history_key)
                    if not path or not path_time:
                        return self.__getitem__((i + 1) % len(self))
                    data_dict[f'image_{cam_name}'] = image_normalization(path, self.cam_image_size)
                    data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, self.cam_image_size)
            elif 'cam_id' not in all_data.keys():
                for cam_name in ['front', 'right', 'left', 'back']:
                    cam_key = f'his_cam_{cam_name}'
                    cam_time_key = f'his_cam_{cam_name}_time'
                    path = decode_path_value(all_data[cam_key])
                    path_time = decode_path_value(all_data[cam_time_key])
                    if not path or not path.strip() or not path_time or not path_time.strip():
                        return self.__getitem__((i + 1) % len(self))
                    data_dict[f'image_{cam_name}'] = image_normalization(path, self.cam_image_size)
                    data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, self.cam_image_size)
            elif not self.use_fixed_images:
                image_front_path = decode_path_value(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, self.cam_image_size)
                image_back_path = decode_path_value(all_data['his_cam_back'])
                data_dict['image_back'] = image_normalization(image_back_path, self.cam_image_size)
                image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                data_dict['image_front_time'] = image_normalization(image_front_time_path, self.cam_image_size)
                image_front_left_path = decode_path_value(all_data['his_cam_front_left'])
                data_dict['image_front_left'] = image_normalization(image_front_left_path, self.cam_image_size)
                image_front_right_path = decode_path_value(all_data['his_cam_front_right'])
                data_dict['image_front_right'] = image_normalization(image_front_right_path, self.cam_image_size)
                image_back_left_path = decode_path_value(all_data['his_cam_back_left'])
                data_dict['image_back_left'] = image_normalization(image_back_left_path, self.cam_image_size)
                image_back_right_path = decode_path_value(all_data['his_cam_back_right'])
                data_dict['image_back_right'] = image_normalization(image_back_right_path, self.cam_image_size)
                data_dict['cam_id'] = get_cam_id_by_label(decode_path_value(all_data['cam_id']))
            else:
                image_front_path = decode_path_value(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, self.cam_image_size)
                image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                data_dict['image_front_time'] = image_normalization(image_front_time_path, self.cam_image_size)
                skill_cam_path = decode_path_value(all_data['skill_cam'])
                data_dict['skill_cam'] = image_normalization(skill_cam_path, self.cam_image_size)
                cam_id = pkl_data_path.split('/')[-1].split('_')[-1]
                cam_id = cam_id.split('.')[0]
                cam_id = get_cam_id_by_name(cam_id)
                data_dict['cam_id'] = cam_id

            trajectory = normalize.prepare_traj(all_data)
            data_dict['trajectory'] = trajectory

            # 方案 A: 加载 HDMap BEV 图像作为额外视觉输入
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

            # 方案 C: 加载下采样原始 BEV 数据，由模型 CNN 编码
            if self.use_hdmap_cnn:
                h, w = self.hdmap_bev_downsample_size
                data_dict['hdmap_valid_mask'] = torch.tensor(0.0, dtype=torch.float32)
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
                        data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)
                else:
                    data_dict['hdmap_bev_raw'] = torch.zeros((HDMAP_BEV_CHANNELS, h, w), dtype=torch.float32)

            hdmap_context = ""
            if self.use_hdmap_image:
                hdmap_context += "An HD map bird's-eye view is provided showing lane layout, junctions, and road boundaries. "
            if self.use_hdmap_proprio or self.use_hdmap_cnn:
                hdmap_context += "Road structure features (lane density, junction proximity, lane boundaries) are encoded in the vehicle state. "

            # --- 方案 A & D: 自动构造辅助监督标签 ---
            aux = build_aux_targets(all_data)
            for k, v in aux.items():
                data_dict[k] = torch.tensor(v)

            nav_desc = all_data.get('navigation_command_descreption')
            if nav_desc is not None:
                nav_desc = np.asarray(nav_desc).reshape(-1)
                if hdmap_context:
                    data_dict['language_instruction'] = f"{hdmap_context}Predict trajectory based on the HD map and navigation command: {str(nav_desc[0])}"
                else:
                    data_dict['language_instruction'] = f"Predict trajectory based on the navigation command: {str(nav_desc[0])}"
            else:
                if hdmap_context:
                    data_dict['language_instruction'] = f"{hdmap_context}Predict trajectory considering the road structure."
                else:
                    data_dict['language_instruction'] = 'predict trajectory'
        return data_dict


def prepare_b2d_dataset(
    work_dir: str,
    statistics_path: str,
    use_fixed_images: bool,
    num_of_action_experts: int,
    split: Optional[str] = None,
    dynamic_dir_name: str = 'b2d_dynamic_camera',
    fixed_dir_name: str = 'b2d_fixed_camera',
    norm_mode: str = "percentile",
    nav_onehot: bool = False,
    use_his_trajectory: bool = False,
    state_frame_index: Optional[int] = None,
    use_hdmap_image: bool = False, 
    use_hdmap_proprio: bool = False, 
    use_hdmap_cnn: bool = False, 
    cam_image_size: tuple = (224, 224), 
    hdmap_image_size: tuple = (192, 352), 
    hdmap_bev_downsample_size: tuple = (48, 90), 
    hdmap_downsample_mode=None, 
    hdmap_channel_scale=None, 
):
    common_kwargs = dict(
        statistics_path=statistics_path,
        num_of_action_experts=num_of_action_experts,
        norm_mode=norm_mode,
        nav_onehot=nav_onehot,
        use_his_trajectory=use_his_trajectory,
        state_frame_index=state_frame_index,
        use_hdmap_image=use_hdmap_image,
        use_hdmap_proprio=use_hdmap_proprio,
        use_hdmap_cnn=use_hdmap_cnn,
        cam_image_size=cam_image_size,
        hdmap_image_size=hdmap_image_size,
        hdmap_bev_downsample_size=hdmap_bev_downsample_size,
        hdmap_downsample_mode=hdmap_downsample_mode,
        hdmap_channel_scale=hdmap_channel_scale,
    )
    if use_fixed_images:
        data_path = os.path.join(work_dir, fixed_dir_name)
        return TorchDataset(data_path=data_path, use_fixed_images=True, **common_kwargs)
    data_path = os.path.join(work_dir, dynamic_dir_name)
    if split == 'train':
        train_data_path = os.path.join(data_path, 'train')
        return TorchDataset(data_path=os.path.join(data_path, 'train'), use_fixed_images=False, **common_kwargs)
    if split == 'val':
        val_data_path = os.path.join(data_path, 'val')
        return TorchDataset(data_path=os.path.join(data_path, 'val'), use_fixed_images=False, **common_kwargs)
    raise ValueError("split type must be 'train' or 'val'")
