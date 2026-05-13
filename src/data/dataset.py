import os
import torch
import pickle
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.utils.image import image_normalization
from src.data.utils.normalization import Normalize
from src.data.camera_scenario_map import get_cam_id, get_scenario_id

class TorchDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        statistics_path: str, 
        return_camera_id: bool,
        return_scenario_id: bool,
        is_drivemoe: bool, 
        scene_priority=None
    ):
        super(TorchDataset, self).__init__()
        self.data_path = data_path
        self.statistics_path = statistics_path
        self.return_camera_id = return_camera_id
        self.return_scenario_id = return_scenario_id
        self.is_drivemoe = is_drivemoe
        self.list_data_dict = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
        
        self.sample_weights = []
        if scene_priority:
            for filename in self.list_data_dict:
                scene_name = filename.split('_')[0]
                weight = 1.0
                for scene_keyword, priority_val in scene_priority.items():
                    if scene_keyword == scene_name:
                        weight = priority_val
                        break
                self.sample_weights.append(weight)
                
    def get_sampler(self):
        if not self.sample_weights:
            return None
        weights_tensor = torch.DoubleTensor(self.sample_weights)
        return WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data_dict = {}
        pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            normalize = Normalize.get_instance(self.statistics_path)
            state = normalize.prepare_state(all_data)
            data_dict['state'] = state
            
            image_front_path = all_data['his_image_front'].numpy().decode('utf-8')
            data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                
            # Extract current frame number from image path
            current_step_str = all_data["his_image_front"].numpy().decode('utf-8').split("/")[-1].split(".")[0]
            his_step = int(current_step_str) - 1  # Get previous frame index
            his_step = max(his_step, 0)  # Ensure not negative
            his_step_str = str(his_step).zfill(5)  # Format as 5-digit string
            
            # Generate path for previous frame image
            cam_his = all_data["his_image_front"].numpy().decode('utf-8').replace(current_step_str, his_step_str)
                
            data_dict['image_front_time'] = image_normalization(cam_his, (224, 224))
                
            # drivemoe
            if self.is_drivemoe:
                image_back_path = all_data['his_image_back'].numpy().decode('utf-8')
                data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                image_front_left_path = all_data['his_image_front_left'].numpy().decode('utf-8')
                data_dict['image_front_left'] = image_normalization(image_front_left_path, (224, 224))
                image_front_right_path = all_data['his_image_front_right'].numpy().decode('utf-8')
                data_dict['image_front_right']= image_normalization(image_front_right_path, (224, 224))
                image_back_left_path = all_data['his_image_back_left'].numpy().decode('utf-8')
                data_dict['image_back_left'] = image_normalization(image_back_left_path, (224, 224))
                image_back_right_path = all_data['his_image_back_right'].numpy().decode('utf-8')
                data_dict['image_back_right']= image_normalization(image_back_right_path, (224, 224))
                
                data_dict['waypoints'] = normalize.prepare_nav_points(all_data)
                
                if self.return_camera_id:
                    data_dict['cam_id'] = get_cam_id(all_data['his_camera_id'].numpy().decode('utf-8'))
                if self.return_scenario_id:
                    data_dict['scenario_id'] = get_scenario_id(all_data['his_scenario_label'].numpy().decode('utf-8'))
            
            trajectory = normalize.prepare_traj(all_data)
            data_dict['trajectory'] = trajectory
            data_dict['language_instruction'] = 'predict trajectory'
        return data_dict

def prepare_b2d_dataset(
    work_dir: str,
    statistics_path: str,
    split: str,
    return_camera_id: bool = False,
    return_scenario_id: bool = False,
    is_drivemoe: bool = False,
    scene_priority=None,
):  
    if split == 'train':
        train_data_path = os.path.join(work_dir, 'train')
        return TorchDataset(
            data_path=train_data_path, 
            statistics_path=statistics_path, 
            return_camera_id=return_camera_id,
            return_scenario_id=return_scenario_id,
            is_drivemoe=is_drivemoe,
            scene_priority=scene_priority
        )
    elif split == 'val':
        val_data_path = os.path.join(work_dir, 'val')
        return TorchDataset(
            data_path=val_data_path, 
            statistics_path=statistics_path, 
            return_camera_id=return_camera_id,
            return_scenario_id=return_scenario_id,
            is_drivemoe=is_drivemoe,
            scene_priority=None
        )
    else:
        raise ValueError(f"split type must be 'train' or 'val'")