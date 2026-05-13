import os
import ray
import uuid
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

class Bench2DriveGenerator:
    def __init__(self, work_dir, is_train, window_size, horizon):
        """Initialize the data generator with processing parameters.
        
        Args:
            work_dir: Root directory for input/output data
            is_train: Flag for training/validation mode
            window_size: Number of historical timesteps to use
            horizon: Number of future timesteps to predict
        """
        self.window_size = window_size
        self.horizon = horizon
        self.is_train = is_train
        self.dataset_size = 0

        # Initialize Ray for parallel processing if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Configure input/output paths
        self._setup_paths(work_dir)
        
        # Load the raw episode data from pickle file
        with open(self.pkl_file_path, 'rb') as f:
            all_episode_data = pickle.load(f)
        
        # Process all episodes in parallel
        self.data = self._parallel_process_episodes(all_episode_data)
        
        # Clean up the temporary pickle file
        os.remove(self.pkl_file_path)

    def _setup_paths(self, work_dir):
        """Configure input/output paths based on mode (train/val)."""
        mode = 'train' if self.is_train else 'val'
        self.action_data_save_path = os.path.join(work_dir, 'b2d_action', mode)
        self.pkl_file_path = os.path.join(work_dir, 'b2d_action', mode, f'{mode}.pkl')
        os.makedirs(self.action_data_save_path, exist_ok=True)

    def _parallel_process_episodes(self, all_episode_data):
        """Process all episodes in parallel using Ray."""
        # Create list of tuples with all arguments needed for processing
        processing_args = [
            (episode_data, idx, self.action_data_save_path, self.window_size, self.horizon)
            for idx, episode_data in enumerate(all_episode_data)
        ]
    
        # Process episodes in parallel using Ray
        futures = [self._process_single_episode_remote.remote(self, *args) for args in processing_args]
    
        # Collect results with progress bar
        results = []
        with tqdm(total=len(futures), desc="Processing episodes") as pbar:
            while futures:
                done, futures = ray.wait(futures)
                results.extend(ray.get(done))
                pbar.update(len(done))
    
        return [sample for episode_samples in results for sample in episode_samples]

    @ray.remote
    def _process_single_episode_remote(self, episode_data, episode_idx, save_path, window_size, horizon):
        """Process a single episode in parallel."""
        # Unpack episode components
        components = [
            'x_pure', 'y_pure', 'theta', 'speed', 'x_command_far', 'y_command_far',
            'acceleration', 'angular_velocity', 'image_front', 'image_front_left',
            'image_front_right', 'image_back', 'image_back_left', 'image_back_right',
            'camera_id', 'scenario_label', 'scenario_id', 'town_id', 'route_id', 'weather_id'
        ]
        episode = dict(zip(components, episode_data))

        # Window all components
        keys_to_process = components[:-4]
        windowed = {}
        for k in keys_to_process:
            windowed[k] = self._state_window(episode[k], window_size)
    
        # Create future trajectories
        trajectory_x = self._trajectory_window(episode['x_pure'], 1, horizon)
        trajectory_y = self._trajectory_window(episode['y_pure'], 1, horizon)

        samples = []
        for sample_idx in range(len(trajectory_x)):
            # Create sample dictionary with all features
            sample = {
                **{f'his_{k}': windowed[k][sample_idx] for k in windowed},
                'fur_x': trajectory_x[sample_idx],
                'fur_y': trajectory_y[sample_idx],
                'episode_idx': episode_idx,
                'sample_idx': sample_idx,
                'unique_id': str(uuid.uuid4())[:8]
            }
        
            # Process the sample (coordinate transforms, etc.)
            processed_sample = self._process_individual_sample(sample)
        
            # Save the processed sample
            filename = f"{episode['scenario_id']}_{episode['town_id']}_{episode['route_id']}_{episode['weather_id']}_step{sample_idx}.pkl"
            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(processed_sample, f)
        
            samples.append(processed_sample)
    
        return samples

    @staticmethod
    def _state_window(traj, window_size):
        """Create sliding window view of sequential data.
        
        Args:
            traj: Input sequence (1D array)
            window_size: Number of timesteps in the window
            
        Returns:
            2D array where each row contains window_size consecutive elements
        """
        traj_len = len(traj)
        indices = tf.range(traj_len)[:, None] + tf.range(-window_size + 1, 1)
        indices = tf.maximum(indices, 0)
        return tf.gather(traj, indices)

    @staticmethod
    def _trajectory_window(traj, window_size, horizon):
        """Create future trajectory prediction windows.
        
        Args:
            traj: Input sequence (1D array)
            window_size: History window size
            horizon: Prediction horizon
            
        Returns:
            3D array of shape [num_samples, window_size, horizon]
        """
        traj_len = len(traj)
        history_indices = tf.range(traj_len)[:, None] + tf.range(-window_size + 1, 1)
        future_indices = tf.range(traj_len)[:, None] + tf.range(horizon)
        future_indices = tf.minimum(future_indices, traj_len - 1)
        
        future_traj = tf.gather(traj, future_indices)
        return tf.gather(future_traj, history_indices)[:traj_len - horizon + 1]

    @staticmethod
    def world2ego(ego_theta, ego_x, ego_y, point_x, point_y):
        """Convert world coordinates to ego vehicle frame.
        
        Args:
            ego_theta: Vehicle heading angle
            ego_x, ego_y: Vehicle position
            point_x, point_y: Point to transform
            
        Returns:
            Transformed (x,y) in ego vehicle coordinates
        """
        R = np.array([
            [np.cos(ego_theta), np.sin(ego_theta)],
            [-np.sin(ego_theta), np.cos(ego_theta)]
        ])
        point_in_ego = np.array([point_x - ego_x, point_y - ego_y])
        return R.dot(point_in_ego)

    def _process_individual_sample(self, sample):
        """Process a single windowed sample (coordinate transforms, etc.)."""
        # Get current ego state
        ego_x = sample['his_x_pure'][-1]
        ego_y = sample['his_y_pure'][-1]
        his_theta = [theta - np.pi/2 if not np.isnan(theta) else 0 for theta in sample['his_theta']]
        ego_theta = his_theta[-1]
        
        # Process features
        processed = {
            # Convert angles to ego frame
            'his_theta': np.array([ta - ego_theta for ta in his_theta], dtype=np.float32),
            
            # Convert commands to ego frame
            **self._process_commands(sample, ego_theta, ego_x, ego_y),
            
            # Convert future trajectory to ego frame
            **self._process_trajectory(sample, ego_theta, ego_x, ego_y),
            
            # Copy other features directly
            'his_speed': np.array(sample['his_speed'], dtype=np.float32),
            'his_acceleration': np.array(sample['his_acceleration'], dtype=np.float32),
            'his_angular_velocity': np.array(sample['his_angular_velocity'], dtype=np.float32),
            
            # Camera data
            **{k: sample[k][-1] for k in sample if k.startswith('his_image_')},
            
            # Camera id
            **{k: sample[k][-1] for k in sample if k.startswith('his_camera_id')},
            
            # Scenario id
            **{k: sample[k][-1] for k in sample if k.startswith('his_scenario_label')},
            
            # Metadata
            'episode_id': sample['episode_idx'],
            'sample_id': sample['sample_idx'],
            'unique_id': sample['unique_id']
        }
        
        return processed

    def _process_commands(self, sample, ego_theta, ego_x, ego_y):
        """Convert command waypoints to ego vehicle coordinates."""
        far_xy = [self.world2ego(ego_theta, ego_x, ego_y, x, y) 
                 for x, y in zip(sample['his_x_command_far'], sample['his_y_command_far'])]
        return {
            'x_command_far': np.array([x for x, _ in far_xy], dtype=np.float32),
            'y_command_far': np.array([y for _, y in far_xy], dtype=np.float32)
        }

    def _process_trajectory(self, sample, ego_theta, ego_x, ego_y):
        """Convert future trajectory to ego vehicle coordinates."""
        fur_x, fur_y = [], []
        for i in range(len(sample['fur_x'])):
            tmp = [self.world2ego(ego_theta, ego_x, ego_y, x, y) 
                  for x, y in zip(sample['fur_x'][i], sample['fur_y'][i])]
            fur_x.append([t[0].tolist() for t in tmp])
            fur_y.append([t[1].tolist() for t in tmp])
        return {
            'fur_x': np.array(fur_x, dtype=np.float32),
            'fur_y': np.array(fur_y, dtype=np.float32)
        }

    def start(self):
        """Main method to process and save all data."""
        # Data is already processed during initialization
        self.dataset_size = len(self.data)
        return self.dataset_size


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bench2Drive Data Processing')
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Root directory for data files")
    parser.add_argument("--window_size", type=int, required=True,
                        help="Number of historical timesteps to use")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Number of future timesteps to predict")
    parser.add_argument("--num_cpus", type=int, default=None,
                        help="Number of CPUs to use for parallel processing")
    args = parser.parse_args()

    # Initialize Ray with specified resources
    ray.init(num_cpus=args.num_cpus, num_gpus=0)

    print("\nProcessing training data:")
    train_generator = Bench2DriveGenerator(
        work_dir=args.work_dir,
        is_train=True,
        window_size=args.window_size,
        horizon=args.horizon
    )
    train_size = train_generator.start()

    print("\nProcessing validation data:")
    val_generator = Bench2DriveGenerator(
        work_dir=args.work_dir,
        is_train=False,
        window_size=args.window_size,
        horizon=args.horizon
    )
    val_size = val_generator.start()

    print(f"\nDataset sizes:\nTrain: {train_size}\nValidation: {val_size}")
    
    # Clean up Ray resources
    ray.shutdown()