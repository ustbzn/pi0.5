import tensorflow as tf
import numpy as np
from PIL import Image
from src.data.dataset import prepare_b2d_dataset
from src.utils.monitor import log_execution_time
from torch.utils.data import DataLoader


tf.config.set_visible_devices([], "GPU")

class Bench2DriveDataset:
    @log_execution_time()
    def __init__(self, config):
        dataset = prepare_b2d_dataset(
            work_dir=config.work_dir,
            statistics_path=config.statistics_path,
            use_fixed_images=config.use_fixed_images,
            num_of_action_experts=config.num_of_action_experts,
            split=config.get("split", None),
            dynamic_dir_name=config.get("dynamic_dir", "b2d_dynamic_camera"),
            fixed_dir_name=config.get("fixed_dir", "b2d_fixed_camera"),
            norm_mode=config.get("norm_mode", "percentile"),
            nav_onehot=config.get("nav_onehot", False),
            use_his_trajectory=config.get("use_his_trajectory", False),
            state_frame_index=config.get("state_frame_index", None),
            use_hdmap_image=config.get("use_hdmap_image", False),
            use_hdmap_proprio=config.get("use_hdmap_proprio", False),
            use_hdmap_cnn=config.get("use_hdmap_cnn", False),
            cam_image_size=tuple(config.get("cam_image_size", [224, 224])),
            hdmap_image_size=tuple(config.get("hdmap_image_size", [352, 192])),
            hdmap_bev_downsample_size=tuple(config.get("hdmap_bev_downsample_size", [48, 90])),
            hdmap_downsample_mode=config.get("hdmap_downsample_mode", None),
            hdmap_channel_scale=config.get("hdmap_channel_scale", None),
        )
        self.dataset = dataset


if __name__ == '__main__':
    from omegaconf import OmegaConf
    fixed_camera_config = OmegaConf.load("config/test_dataloader/fixed_camera.yaml")
    dynamic_camera_train_config = OmegaConf.load("config/test_dataloader/dynamic_camera_train.yaml")
    dynamic_camera_eval_config = OmegaConf.load("config/test_dataloader/dynamic_camera_eval.yaml")
    fixed_camera_dataset = Bench2DriveDataset(fixed_camera_config.data.train).dataset
    dynamic_camera_train_dataset = Bench2DriveDataset(dynamic_camera_train_config.data.train).dataset
    dynamic_camera_eval_dataset = Bench2DriveDataset(dynamic_camera_eval_config.data.val).dataset
    
    fixed_camera_dataloader = DataLoader(fixed_camera_dataset, batch_size=1, shuffle=False)
    dynamic_camera_train_dataloader = DataLoader(dynamic_camera_train_dataset, batch_size=1, shuffle=False)
    dynamic_camera_eval_dataloader = DataLoader(dynamic_camera_eval_dataset, batch_size=1, shuffle=False)
    print('-----------------------fixed camera data-----------------------')
    for batch in fixed_camera_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        print('cam_id: ', batch['cam_id'])
        break
    print('-----------------------dynamic camera data-----------------------')
    print('train')
    for batch in dynamic_camera_train_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break
    print('eval')
    for batch in dynamic_camera_eval_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break
