from typing import Sequence, Optional
import os
import torch
import pickle
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from torch.utils.data import Dataset
from src.data.utils.image import read_resize_encode_image_pytorch, concatenate_images, image_normalization
from src.data.utils.normalization import Normalize
from src.data.camera_scenario_map import get_scenario_id, get_cam_id_by_name, get_cam_id_by_label

def decode_path_value(value):
    """
    兼容多种格式的路径解码函数
    
    支持格式：
    1. 旧格式：有 .numpy() 方法的对象（如 TensorFlow tensor）
    2. 新格式：dict 类型（从 StringWrapper 序列化而来，有 _bytes_value 字段）
    3. 直接是字符串类型
    4. bytes 类型
    
    参数:
        value: 需要解码的值
    
    返回:
        str: 解码后的字符串路径
    """
    if isinstance(value, str):
        # 直接是字符串，直接返回
        return value
    elif isinstance(value, bytes):
        # 直接是 bytes，解码返回
        return value.decode('utf-8')
    elif isinstance(value, dict):
        # 新格式：dict 类型（从 StringWrapper 序列化而来）
        if '_bytes_value' in value:
            # 从 StringWrapper 序列化而来的 dict
            bytes_value = value['_bytes_value']
            if isinstance(bytes_value, bytes):
                return bytes_value.decode('utf-8')
            elif isinstance(bytes_value, str):
                return bytes_value
            else:
                # 尝试转换为字符串
                return str(bytes_value)
        else:
            # 其他 dict 类型，尝试转换为字符串
            return str(value)
    elif hasattr(value, 'numpy'):
        # 旧格式：有 .numpy() 方法的对象（如 TensorFlow tensor）
        try:
            numpy_value = value.numpy()
            if isinstance(numpy_value, bytes):
                return numpy_value.decode('utf-8')
            elif isinstance(numpy_value, str):
                return numpy_value
            elif isinstance(numpy_value, np.ndarray):
                # numpy array，尝试解码
                if numpy_value.dtype == np.uint8:
                    return numpy_value.tobytes().decode('utf-8')
                else:
                    return str(numpy_value)
            else:
                return str(numpy_value)
        except Exception as e:
            # 如果 .numpy() 调用失败，尝试直接转换
            return str(value)
    else:
        # 其他类型，尝试转换为字符串
        return str(value)

class TorchDataset(Dataset):
    def __init__(self, data_path: str, statistics_path: str, use_fixed_images: bool, num_of_action_experts: int, norm_mode: str = "percentile", nav_onehot: bool = False):
        super(TorchDataset, self).__init__()
        self.data_path = data_path
        self.statistics_path = statistics_path
        self.use_fixed_images = use_fixed_images
        self.num_of_action_experts = num_of_action_experts
        self.norm_mode = norm_mode
        self.nav_onehot = nav_onehot
        self.list_data_dict = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
    
    def __len__(self):
        return len(self.list_data_dict)

    def get_nav_labels(self):
        """预扫描所有样本的 navigation_id，用于构建 WeightedRandomSampler"""
        labels = []
        for fname in self.list_data_dict:
            pkl_path = os.path.join(self.data_path, fname)
            try:
                with open(pkl_path, 'rb') as f:
                    d = pickle.load(f)
                nav = d.get('navigation_id')
                if nav is not None and hasattr(nav, '__len__') and len(nav) > 0:
                    labels.append(int(nav[0]))
                else:
                    labels.append(0)
            except Exception:
                labels.append(0)
        return labels

    def __getitem__(self, i):
        data_dict = {}
        pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            normalize = Normalize.get_instance(self.statistics_path, norm_mode=self.norm_mode)
            state = normalize.prepare_state(all_data, nav_onehot=self.nav_onehot)
            data_dict['state'] = state
            # drive-pi0
            if 'cam_id' not in all_data.keys():
                # 4 cameras x 2 time steps = 8 images
                for cam_name in ['front', 'right', 'left', 'back']:
                    cam_key = f'his_cam_{cam_name}'
                    cam_time_key = f'his_cam_{cam_name}_time'
                    path = decode_path_value(all_data[cam_key])
                    path_time = decode_path_value(all_data[cam_time_key])
                    if not path or not path.strip() or not path_time or not path_time.strip():
                        # 跳过坏样本，随机取下一个
                        return self.__getitem__((i + 1) % len(self))
                    data_dict[f'image_{cam_name}'] = image_normalization(path, (224, 224))
                    data_dict[f'image_{cam_name}_time'] = image_normalization(path_time, (224, 224))

            elif not self.use_fixed_images:
                print("走第二个分支！！！")
                image_front_path = decode_path_value(all_data['his_cam_front'])
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                image_back_path = decode_path_value(all_data['his_cam_back'])
                data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                image_front_time_path = decode_path_value(all_data['his_cam_front_time'])
                data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224))
                image_front_left_path = decode_path_value(all_data['his_cam_front_left'])
                data_dict['image_front_left'] = image_normalization(image_front_left_path, (224, 224))
                image_front_right_path = decode_path_value(all_data['his_cam_front_right'])
                data_dict['image_front_right']= image_normalization(image_front_right_path, (224, 224))
                image_back_left_path = decode_path_value(all_data['his_cam_back_left'])
                data_dict['image_back_left'] = image_normalization(image_back_left_path, (224, 224))
                image_back_right_path = decode_path_value(all_data['his_cam_back_right'])
                data_dict['image_back_right']= image_normalization(image_back_right_path, (224, 224))
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
            # scenario_id = pkl_data_path.split('/')[-1].split('_')[0]
            # scenario_id = get_scenario_id(self.num_of_action_experts, scenario_id)
            # data_dict['scenario_id'] = scenario_id
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
        )
    raise ValueError("split type must be 'train' or 'val'")