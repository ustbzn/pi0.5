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


class TorchDataset(Dataset):
    def __init__(self, data_path: str, statistics_path: str, use_fixed_images: bool, num_of_action_experts: int, norm_mode: str = "percentile", nav_onehot: bool = False, use_his_trajectory: bool = False):
        super(TorchDataset, self).__init__()
        self.data_path = data_path
        self.statistics_path = statistics_path
        self.use_fixed_images = use_fixed_images
        self.num_of_action_experts = num_of_action_experts
        self.norm_mode = norm_mode
        self.nav_onehot = nav_onehot
        self.use_his_trajectory = use_his_trajectory
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
            state = normalize.prepare_state(
                all_data,
                nav_onehot=self.nav_onehot,
                use_his_trajectory=self.use_his_trajectory,
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
                    data_dict[f'image_{cam_name}'] = image_normalization(path, (224, 224))
                    data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, (224, 224))
            elif 'cam_id' not in all_data.keys():
                for cam_name in ['front', 'right', 'left', 'back']:
                    cam_key = f'his_cam_{cam_name}'
                    cam_time_key = f'his_cam_{cam_name}_time'
                    path = decode_path_value(all_data[cam_key])
                    path_time = decode_path_value(all_data[cam_time_key])
                    if not path or not path.strip() or not path_time or not path_time.strip():
                        return self.__getitem__((i + 1) % len(self))
                    data_dict[f'image_{cam_name}'] = image_normalization(path, (224, 224))
                    data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, (224, 224))
            elif not self.use_fixed_images:
                image_front_path = decode_path_value(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                image_back_path = decode_path_value(all_data['his_cam_back'])
                data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224))
                image_front_left_path = decode_path_value(all_data['his_cam_front_left'])
                data_dict['image_front_left'] = image_normalization(image_front_left_path, (224, 224))
                image_front_right_path = decode_path_value(all_data['his_cam_front_right'])
                data_dict['image_front_right'] = image_normalization(image_front_right_path, (224, 224))
                image_back_left_path = decode_path_value(all_data['his_cam_back_left'])
                data_dict['image_back_left'] = image_normalization(image_back_left_path, (224, 224))
                image_back_right_path = decode_path_value(all_data['his_cam_back_right'])
                data_dict['image_back_right'] = image_normalization(image_back_right_path, (224, 224))
                data_dict['cam_id'] = get_cam_id_by_label(decode_path_value(all_data['cam_id']))
            else:
                image_front_path = decode_path_value(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224))
                skill_cam_path = decode_path_value(all_data['skill_cam'])
                data_dict['skill_cam'] = image_normalization(skill_cam_path, (224, 224))
                cam_id = pkl_data_path.split('/')[-1].split('_')[-1]
                cam_id = cam_id.split('.')[0]
                cam_id = get_cam_id_by_name(cam_id)
                data_dict['cam_id'] = cam_id

            trajectory = normalize.prepare_traj(all_data)
            data_dict['trajectory'] = trajectory

            nav_desc = all_data.get('navigation_command_descreption')
            if nav_desc is not None:
                nav_desc = np.asarray(nav_desc).reshape(-1)
                data_dict['language_instruction'] = str(nav_desc[0])
            else:
                data_dict['language_instruction'] = 'predict trajectory'
        return data_dict


def prepare_b2d_dataset(work_dir: str, statistics_path: str, use_fixed_images: bool, num_of_action_experts: int, split: Optional[str] = None, dynamic_dir_name: str = 'b2d_dynamic_camera', fixed_dir_name: str = 'b2d_fixed_camera', norm_mode: str = "percentile", nav_onehot: bool = False, use_his_trajectory: bool = False):
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
        )
    raise ValueError("split type must be 'train' or 'val'")
