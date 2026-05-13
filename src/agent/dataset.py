import tensorflow as tf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import prepare_b2d_dataset
from src.utils.monitor import log_execution_time

tf.config.set_visible_devices([], "GPU")

SCENE_PRIORITOY = {}

class Bench2DriveDataset:
    @log_execution_time()
    def __init__(self, config):
        dataset = prepare_b2d_dataset(
            work_dir=config.work_dir,
            statistics_path=config.statistics_path,
            return_camera_id=config.return_camera_id,
            return_scenario_id=config.return_scenario_id,
            split=config.split,
            is_drivemoe=config.is_drivemoe,
            scene_priority=SCENE_PRIORITOY if config.set_scene_priority else None,
        )
        self.dataset = dataset

if __name__ == '__main__':
    drivepi_config = OmegaConf.load("config/train/DrivePi0/base.yaml")
    drivepi_dataset = Bench2DriveDataset(drivepi_config.data.train).dataset
    drivepi_sampler = drivepi_dataset.get_sampler()
    drivepi_dataloader = DataLoader(
        drivepi_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False, 
        sampler=drivepi_sampler
    )
    for batch in drivepi_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break
    drivemoe_config = OmegaConf.load("config/train/DriveMoE/stage1_closed_loop.yaml")
    drivemoe_dataset = Bench2DriveDataset(drivemoe_config.data.train).dataset
    drivemoe_sampler = drivemoe_dataset.get_sampler()
    drivemoe_dataloader = DataLoader(
        drivemoe_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False,
        sampler=drivemoe_sampler
    )
    for batch in drivemoe_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        print('cam id', batch['cam_id'])
        print('scenario id', batch['scenario_id'])
        print('waypoints', batch['waypoints'])
        break