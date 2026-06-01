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

    同时输出用于方案 D 分桶评估的 trajectory_bucket：
      straight(0) / turn(1) / static(2) / high_curvature(3)

    Args:
        all_data: 支持两种轨迹格式：
                  1) fur_trajectory shape = [H, 2]，列顺序 [y, x]
                  2) fur_x/fur_y，旧 Bench2Drive 格式，shape 可为 [1, H] 或 [H]
    Returns:
        dict: 包含上述 5 个辅助标签和 trajectory_bucket
    """
    if "fur_trajectory" in all_data:
        fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)  # [H, 2], [y, x]
        fur_y = fur[:, 0]
        fur_x = fur[:, 1]
    else:
        fur_x = np.asarray(all_data["fur_x"], dtype=np.float32).reshape(-1)
        fur_y = np.asarray(all_data["fur_y"], dtype=np.float32).reshape(-1)
        fur = np.stack([fur_y, fur_x], axis=1)

    final_y, final_x = float(fur_y[-1]), float(fur_x[-1])
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
        state_only: bool = False,
        image_augment: bool = True,
        camera_mode: Optional[str] = None,
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
        self.state_only = state_only
        self.image_augment = image_augment
        self.camera_mode = camera_mode
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

    @staticmethod
    def _previous_frame_path(path: str):
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        stem, ext = os.path.splitext(filename)
        if not stem.isdigit():
            return None
        previous_stem = str(max(int(stem) - 1, 0)).zfill(len(stem))
        return os.path.join(dirname, f"{previous_stem}{ext}")

    def __getitem__(self, i):
        data_dict = {}
        pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            normalize = Normalize.get_instance(self.statistics_path, norm_mode=self.norm_mode)

            # 确保 navigation_id 存在
            if 'navigation_id' not in all_data:
                all_data = dict(all_data)
                target_len = np.asarray(all_data['his_theta']).shape[0]
                all_data['navigation_id'] = build_navigation_id_sequence(all_data, target_len)

            state = normalize.prepare_state(
                all_data,
                nav_onehot=self.nav_onehot,
                use_his_trajectory=self.use_his_trajectory,
                state_frame_index=self.state_frame_index,
            )
            data_dict['state'] = state

            # state_only 模式下跳过全部图片加载，大幅减少 CPU 开销
            if not self.state_only:
                # 新格式数据（cam_front / cam_right / cam_left / cam_back）
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
                        data_dict[f'image_{cam_name}'] = image_normalization(path, (224, 224), augment=self.image_augment)
                        data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, (224, 224), augment=self.image_augment)
                # B2D action 格式：6 路当前帧图像；历史帧由当前帧路径回推上一帧编号。
                elif 'his_image_front' in all_data:
                    if self.camera_mode in ("front_history_2", "b2d_front_history_2"):
                        image_key_map = {'front': 'his_image_front'}
                    else:
                        image_key_map = {
                            'front': 'his_image_front',
                            'front_left': 'his_image_front_left',
                            'front_right': 'his_image_front_right',
                            'back': 'his_image_back',
                            'back_left': 'his_image_back_left',
                            'back_right': 'his_image_back_right',
                        }
                    needs_history = self.camera_mode in (
                        "front_history_2",
                        "b2d_front_history_2",
                        "b2d_surround_history_12",
                        "surround_history_12",
                        "mogo_8",
                        "surround_history_8",
                    )
                    for cam_name, key in image_key_map.items():
                        path = self._pick_image_path(all_data, key)
                        if not path:
                            return self.__getitem__((i + 1) % len(self))
                        data_dict[f'image_{cam_name}'] = image_normalization(
                            path, (224, 224), augment=self.image_augment
                        )
                        if needs_history:
                            history_path = self._previous_frame_path(path)
                            if not history_path:
                                return self.__getitem__((i + 1) % len(self))
                            data_dict[f'image_{cam_name}_time'] = image_normalization(
                                history_path, (224, 224), augment=self.image_augment
                            )
                    # 兼容其他 camera_mode 的通用命名。
                    if 'image_front_left' in data_dict:
                        data_dict['image_left'] = data_dict['image_front_left']
                    if 'image_front_right' in data_dict:
                        data_dict['image_right'] = data_dict['image_front_right']
                    if 'image_front_left_time' in data_dict:
                        data_dict['image_left_time'] = data_dict['image_front_left_time']
                    if 'image_front_right_time' in data_dict:
                        data_dict['image_right_time'] = data_dict['image_front_right_time']
                    if 'his_camera_id' in all_data:
                        data_dict['cam_id'] = get_cam_id_by_label(
                            decode_path_value(all_data['his_camera_id'])
                        )
                # drive-pi0 旧格式
                elif 'cam_id' not in all_data.keys():
                    # 4 cameras x 2 time steps = 8 images
                    for cam_name in ['front', 'right', 'left', 'back']:
                        cam_key = f'his_cam_{cam_name}'
                        cam_time_key = f'his_cam_{cam_name}_time'
                        path = decode_path_value(all_data[cam_key])
                        path_time = decode_path_value(all_data[cam_time_key])
                        if not path or not path.strip() or not path_time or not path_time.strip():
                            # 跳过坏样本，随机取下一个
                            return self.__getitem__((i + 1) % len(self))
                        data_dict[f'image_{cam_name}'] = image_normalization(path, (224, 224), augment=self.image_augment)
                        data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, (224, 224), augment=self.image_augment)

                elif not self.use_fixed_images:
                    image_front_path = decode_path_value(all_data['his_cam_front'])
                    data_dict['image_front'] = image_normalization(image_front_path, (224, 224), augment=self.image_augment)
                    image_back_path = decode_path_value(all_data['his_cam_back'])
                    data_dict['image_back'] = image_normalization(image_back_path, (224, 224), augment=self.image_augment)
                    image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                    data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224), augment=self.image_augment)
                    image_back_time_path = self._pick_image_path(all_data, 'his_cam_back_time')
                    if image_back_time_path:
                        data_dict['image_back_time'] = image_normalization(image_back_time_path, (224, 224), augment=self.image_augment)
                    image_front_left_path = decode_path_value(all_data['his_cam_front_left'])
                    data_dict['image_front_left'] = image_normalization(image_front_left_path, (224, 224), augment=self.image_augment)
                    image_front_right_path = decode_path_value(all_data['his_cam_front_right'])
                    data_dict['image_front_right'] = image_normalization(image_front_right_path, (224, 224), augment=self.image_augment)
                    image_back_left_path = decode_path_value(all_data['his_cam_back_left'])
                    data_dict['image_back_left'] = image_normalization(image_back_left_path, (224, 224), augment=self.image_augment)
                    image_back_right_path = decode_path_value(all_data['his_cam_back_right'])
                    data_dict['image_back_right'] = image_normalization(image_back_right_path, (224, 224), augment=self.image_augment)
                    # 兼容训练侧统一的 front/right/left/back 命名；旧 B2D 没有 right/left 后向历史帧，
                    # 默认使用当前帧占位，具体使用哪些图像由 camera_mode 决定。
                    data_dict['image_left'] = data_dict['image_front_left']
                    data_dict['image_right'] = data_dict['image_front_right']
                    data_dict['image_left_time'] = data_dict['image_left']
                    data_dict['image_right_time'] = data_dict['image_right']
                    if 'image_back_time' not in data_dict:
                        data_dict['image_back_time'] = data_dict['image_back']
                    data_dict['cam_id'] = get_cam_id_by_label(decode_path_value(all_data['cam_id']))
                else:
                    image_front_path = decode_path_value(all_data['his_cam_front'])
                    data_dict['image_front'] = image_normalization(image_front_path, (224, 224), augment=self.image_augment)
                    image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                    data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224), augment=self.image_augment)
                    skill_cam_path = decode_path_value(all_data['skill_cam'])
                    data_dict['skill_cam'] = image_normalization(skill_cam_path, (224, 224), augment=self.image_augment)
                    cam_id = pkl_data_path.split('/')[-1].split('_')[-1]
                    cam_id = cam_id.split('.')[0]
                    cam_id = get_cam_id_by_name(cam_id)
                    data_dict['cam_id'] = cam_id

            trajectory = normalize.prepare_traj(all_data)
            data_dict['trajectory'] = trajectory

            # --- 方案 A & D: 自动构造辅助监督标签 ---
            aux = build_aux_targets(all_data)
            for k, v in aux.items():
                data_dict[k] = torch.tensor(v)

            nav_desc = all_data.get('navigation_command_descreption')
            if nav_desc is not None:
                nav_desc = np.asarray(nav_desc).reshape(-1)
                data_dict['language_instruction'] = f"Predict trajectory based on the navigation command: {str(nav_desc[0])}"
            elif "command_far" in all_data:
                command = np.asarray(all_data["command_far"]).reshape(-1)
                command_id = int(command[-1]) if command.size else 0
                data_dict['language_instruction'] = f"Predict trajectory based on the navigation command id: {command_id}"
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
    state_only: bool = False,
    image_augment: bool = True,
    camera_mode: Optional[str] = None,
):
    """
    Select the dataset directory based on camera type and split.
    Directory names are configurable, defaulting to legacy values.
    """
    if use_fixed_images:
        data_path = os.path.join(work_dir, fixed_dir_name)
        return TorchDataset(
            data_path=data_path,
            statistics_path=statistics_path,
            use_fixed_images=True,
            num_of_action_experts=num_of_action_experts,
            norm_mode=norm_mode,
            nav_onehot=nav_onehot,
            use_his_trajectory=use_his_trajectory,
            state_frame_index=state_frame_index,
            state_only=state_only,
            image_augment=image_augment,
            camera_mode=camera_mode,
        )
    data_path = os.path.join(work_dir, dynamic_dir_name)
    if split == 'train':
        train_data_path = os.path.join(data_path, 'train')
        return TorchDataset(
            data_path=train_data_path,
            statistics_path=statistics_path,
            use_fixed_images=False,
            num_of_action_experts=num_of_action_experts,
            norm_mode=norm_mode,
            nav_onehot=nav_onehot,
            use_his_trajectory=use_his_trajectory,
            state_frame_index=state_frame_index,
            state_only=state_only,
            image_augment=image_augment,
            camera_mode=camera_mode,
        )
    if split == 'val':
        val_data_path = os.path.join(data_path, 'val')
        return TorchDataset(
            data_path=val_data_path,
            statistics_path=statistics_path,
            use_fixed_images=False,
            num_of_action_experts=num_of_action_experts,
            norm_mode=norm_mode,
            nav_onehot=nav_onehot,
            use_his_trajectory=use_his_trajectory,
            state_frame_index=state_frame_index,
            state_only=state_only,
            image_augment=image_augment,
            camera_mode=camera_mode,
        )
    raise ValueError("split type must be 'train' or 'val'")
