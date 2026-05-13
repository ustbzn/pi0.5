import os
import ray
import json
import argparse
import joblib
from tqdm import tqdm

from data_split import VAL_LIST
from load_utils import load_json_gz

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

@ray.remote
def get_data_info(route_path: str, camera_label_dir: str, scenario_label_dir: str):
    # Extract basic information from route path
    route_name = os.path.basename(route_path)
    scenario_id, town_id, route_id, weather_id = route_name.split('_')[:4]
    anno_dir = os.path.join(route_path, "anno")
    
    camera_label_file_name = str(route_name) + ".json"
    camera_label_path = os.path.join(camera_label_dir, camera_label_file_name)
    scenario_label_path = os.path.join(scenario_label_dir, camera_label_file_name)
    
    # Get total number of steps from annotation directory
    step_sum = len(os.listdir(anno_dir))
    
    # Load all annotation files at once using list comprehension
    anno_list = [load_json_gz(os.path.join(anno_dir, f"{str(step).zfill(5)}.json.gz")) 
                for step in range(step_sum)]
    
    cam_id_list = []
    scenario_id_list = []
    try:
        with open(camera_label_path, 'r') as f:
            cam_data = json.load(f)
            if(len(cam_data) != step_sum):
                raise Exception(f'cam id num in {route_name} is wrong!')
            for step in range(step_sum):
                cam_id_list.append(cam_data[f"{step}"])            
    except FileNotFoundError:
        print(f"file {camera_label_path} does not exist, skip it")
        return None
    try:
        with open(scenario_label_path, 'r') as f:
            scenario_data = json.load(f)
            if(len(scenario_data) != step_sum):
                raise Exception(f'scenario id num in {route_name} is wrong!')
            for step in range(step_sum):
                scenario_id_list.append(scenario_data[f"{step}"])            
    except FileNotFoundError:
        print(f"file {scenario_label_path} does not exist, skip it")
        return None
    
    # Generate paths for all camera images (6 cameras)
    camera_dirs = {
        'front': 'rgb_front',
        'front_left': 'rgb_front_left',
        'front_right': 'rgb_front_right',
        'back': 'rgb_back',
        'back_left': 'rgb_back_left',
        'back_right': 'rgb_back_right'
    }
    
    # Create image path lists for each camera
    camera_paths = {
        name: [os.path.join(route_path, "camera", subdir, f"{str(step).zfill(5)}.jpg")
              for step in range(step_sum)]
        for name, subdir in camera_dirs.items()
    }
    
    # Extract relevant data from annotations
    episode_data = {
        'x_pure': [anno['bounding_boxes'][0]['location'][0] for anno in anno_list],
        'y_pure': [anno['bounding_boxes'][0]['location'][1] for anno in anno_list],
        'theta': [anno['theta'] for anno in anno_list],
        'speed': [anno['speed'] for anno in anno_list],
        'x_command_far': [anno['x_command_far'] for anno in anno_list],
        'y_command_far': [anno['y_command_far'] for anno in anno_list],
        'acceleration': [anno['acceleration'] for anno in anno_list],
        'angular_velocity': [anno['angular_velocity'] for anno in anno_list]
    }
    
    # Prepare final output - all lists except actions are shifted by 1 (current obs -> next obs)
    return (
        episode_data['x_pure'][:-1],      # Current x position
        episode_data['y_pure'][:-1],      # Current y position
        episode_data['theta'][:-1],       # Current orientation
        episode_data['speed'][:-1],       # Current speed
        episode_data['x_command_far'][:-1],  # Far command x position
        episode_data['y_command_far'][:-1],  # Far command y position
        episode_data['acceleration'][:-1],   # Current acceleration
        episode_data['angular_velocity'][:-1], # Current angular velocity
        camera_paths['front'][:-1],       # Front camera images
        camera_paths['front_left'][:-1],  # Front-left camera images
        camera_paths['front_right'][:-1], # Front-right camera images
        camera_paths['back'][:-1],        # Back camera images
        camera_paths['back_left'][:-1],   # Back-left camera images
        camera_paths['back_right'][:-1],  # Back-right camera images
        cam_id_list[:-1],                 # Camera ids
        scenario_id_list[:-1],            # Scenario ids
        scenario_id,
        town_id,
        route_id,
        weather_id
    )

def deal_with_data(data_path: str, work_dir: str, camera_label_dir: str, scenario_label_dir: str):
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(num_cpus=min(32, os.cpu_count()), num_gpus=0)  # Use up to 32 CPUs
    
    try:
        # Get all route directories (first level only)
        route_dirs = []
        for dirpath, dirnames, _ in os.walk(data_path):
            route_dirs = [os.path.join(dirpath, name) for name in dirnames]
            break  # Only process top-level directories
        
        # Submit all processing tasks in parallel
        futures = []
        for route_dir in route_dirs:
            route_name = os.path.basename(route_dir)
            # Submit task to Ray cluster with proper tagging
            future = get_data_info.remote(route_dir, camera_label_dir, scenario_label_dir)
            futures.append((future, 'val' if route_name in VAL_LIST else 'train'))
        
        # Process results as they complete
        train_data = []
        val_data = []
        completed_count = 0
        
        with tqdm(total=len(futures), desc="Processing routes") as progress_bar:
            for future, data_type in futures:
                try:
                    result = ray.get(future)  # Get result from worker
                    if result is not None:  # Skip failed/None results
                        if data_type == 'train':
                            import pdb; pdb.set_trace()
                            train_data.append(result)
                        else:
                            val_data.append(result)
                except Exception as e:
                    print(f"Error processing {future}: {str(e)}")
                finally:
                    completed_count += 1
                    progress_bar.update(1)
        
        # Save processed data
        target_dir = os.path.join(work_dir, 'b2d_action')
        os.makedirs(target_dir, exist_ok=True)
        
        # Save training data
        train_dir = os.path.join(target_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        train_file_path = os.path.join(train_dir, 'train.pkl')
        with open(train_file_path, 'wb') as f:
            joblib.dump(train_data, f)
        
        # Save validation data
        val_dir = os.path.join(target_dir, 'val')
        os.makedirs(val_dir, exist_ok=True)
        val_file_path = os.path.join(val_dir, 'val.pkl')
        with open(val_file_path, 'wb') as f:
            joblib.dump(val_data, f)
            
    finally:
        # Ensure Ray is properly shutdown
        if ray.is_initialized():
            ray.shutdown()
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2d_dataset_dir", type=str, default="data/Bench2Drive-Base")
    parser.add_argument("--camera_label_dir", type=str, default="data/camera_labels")
    parser.add_argument("--scenario_label_dir", type=str, default="data/scenario_labels")
    parser.add_argument("--work_dir", type=str, default="exp")
    args = parser.parse_args()
    deal_with_data(data_path=args.b2d_dataset_dir, work_dir=args.work_dir, camera_label_dir=args.camera_label_dir, scenario_label_dir=args.scenario_label_dir)