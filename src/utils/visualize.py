import os
import re
import cv2
import glob
import pickle
import numpy as np
from PIL import Image
from typing import Tuple

from src.data.utils.image import mosaic_driver_cameras
from src.data.camera_scenario_map import get_cam_id

class VisualizeAgent:
    def __init__(
        self,
        b2d_exp_dir: str = "exp/b2d_action",
    ):
        self.b2d_exp_dir = b2d_exp_dir
        
    def get_scenario_pkl_files(self, scenario_id: str):
        search_pattern = os.path.join(self.b2d_exp_dir, "**", f"{scenario_id}*.pkl")
        pkl_files = glob.glob(search_pattern, recursive=True)
        
        def sort_key(file_path):
            match = re.search(r'step(\d+)\.pkl', file_path)
            return int(match.group(1)) if match else 0

        sorted_files = sorted(pkl_files, key=sort_key)
    
        return (sorted_files, scenario_id)
    
    def generate_video(
        self, 
        scenario_data: Tuple,
    ):
        pkl_files, scenario_id = scenario_data
        video = None
        
        for pkl_file in pkl_files:
            current_frame_pil = self._visualize_pkl_data(pkl_file)
            if video is None:
                w, h = current_frame_pil.size
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(f"visualize/{scenario_id}.mp4", fourcc, 10, (w, h))
            
            img_bgr = cv2.cvtColor(np.array(current_frame_pil), cv2.COLOR_RGB2BGR)
            video.write(img_bgr)
        
        if video:
            video.release()
        
    def get_length(self, scenario_data: Tuple):
        pkl_files, _ = scenario_data
        return len(pkl_files)

    def _visualize_pkl_data(
        self,
        pkl_path: str,
    ):
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
            image_front_left_path = all_data['his_image_front_left'].numpy().decode('utf-8')
            image_front_left_pil = Image.open(image_front_left_path)
            image_front_path = all_data['his_image_front'].numpy().decode('utf-8')
            image_front_pil = Image.open(image_front_path)
            image_front_right_path = all_data['his_image_front_right'].numpy().decode('utf-8')
            image_front_right_pil = Image.open(image_front_right_path)
            image_back_left_path = all_data['his_image_back_left'].numpy().decode('utf-8')
            image_back_left_pil = Image.open(image_back_left_path)
            image_back_path = all_data['his_image_back'].numpy().decode('utf-8')
            image_back_pil = Image.open(image_back_path)
            image_back_right_path = all_data['his_image_back_right'].numpy().decode('utf-8')
            image_back_right_pil = Image.open(image_back_right_path)
            images_list = [image_front_left_pil, image_front_pil, image_front_right_pil, image_back_left_pil, image_back_pil, image_back_right_pil]
            cam_id = get_cam_id(all_data['his_camera_id'].numpy().decode('utf-8'))
            image_cat = mosaic_driver_cameras(images_list, cam_id)
            
        return image_cat
        
if __name__ == "__main__":
    agent = VisualizeAgent()
    scenario_data = agent.get_scenario_pkl_files("Accident_Town05_Route218_Weather10")
    print(agent.get_length(scenario_data))
    agent.generate_video(scenario_data)