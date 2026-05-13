import os
import cv2
import json
import math
import time
import carla
import torch
import einops
import datetime
import pathlib
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import deque
from omegaconf import OmegaConf
from scipy.optimize import fsolve
from transformers import AutoTokenizer
from torchvision import transforms as T

from team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent

from src.utils.pid import PIDController
from src.data.utils.normalization import Normalize
from src.model.DrivePi0.processing import VLAProcessor
from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.data.utils.augmentations import augment_image


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
print('*'*10)
print(PLANNER_TYPE) # None
print('*'*10)

EARTH_RADIUS_EQUA = 6378137.0
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

def get_entry_point():
    return 'DrivePiZeroAgent'

class DrivePiZeroAgent(autonomous_agent.AutonomousAgent):
    def load_checkpoint(self, path):
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
        self.model.load_state_dict(data["model"], strict=True)
    
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        
        self.data_queue = deque()
        self.data_queue_len = 11 ### 20 Hz!!!
        self.pid_controller = PIDController()

        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
            self.config_path = path_to_conf_file.split('+')[0]
        else:
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = OmegaConf.load(self.config_path)
        self.device = torch.device(f"cuda:{self.config.gpu_id}")
        self.dtype = torch.bfloat16 if self.config.get("use_bf16", False) else torch.float32
        self.model = DrivePiZeroInference(self.config, use_ddp=False)
        self.load_checkpoint(self.config.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=self.config.vision.config.num_image_tokens,
            max_seq_len=self.config.max_seq_len,
            tokenizer_padding=self.config.tokenizer_padding,
        )
        self.debug = self.config.debug
        self.save_images = self.config.save_images

        self.coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0],
                                      [0.0, -1.0,  0.0,  0.0], 
                                      [0.0,  0.0, -1.0, 50.0], 
                                      [0.0,  0.0,  0.0,  1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

        self.save_path = None

        # save images for debugging
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = self.save_name

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            if self.save_images:
                (self.save_path / 'rgb_front').mkdir()
                (self.save_path / 'bev_traj').mkdir()
            (self.save_path / 'meta').mkdir()
            
    
    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            E = EARTH_RADIUS_EQUA
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * E) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * E * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * E * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        print(self.lat_ref, self.lon_ref, self.save_name)
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self._waypoint_planner = RoutePlanner(7.5, 25.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref) # far
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self.initialized = True
        self.metric_info = {}
    
    def sensors(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
                },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
                },
            # gps
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
                },
            # imu
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # speed
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
                },
        ]
        if IS_BENCH2DRIVE:
            sensors += [
                {	
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1

        rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_back = cv2.cvtColor(input_data['CAM_BACK'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        _, rgb_front = cv2.imencode('.jpg', rgb_front, encode_param)
        rgb_front = cv2.imdecode(rgb_front, cv2.IMREAD_COLOR)
        _, rgb_back = cv2.imencode('.jpg', rgb_back, encode_param)
        rgb_back = cv2.imdecode(rgb_back, cv2.IMREAD_COLOR)
        
        gps = input_data['GPS'][1][:3]
        speed = input_data['SPEED'][1]['speed']
        acceleration = input_data['IMU'][1][:3]
        compass = input_data['IMU'][1][-1]
        angular_velocity = input_data['IMU'][1][3:6]
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
        
        result = {
			'rgb_front': rgb_front,
            'rgb_back': rgb_back,
			'gps': gps,
			'speed': speed,
			'compass': compass,
			'bev': bev,
			'acceleration': acceleration,
            'angular_velocity': angular_velocity
			}
        pos = self.gps_to_location(result['gps'])
        far_node, _ = self._waypoint_planner.run_step(pos[:2])
        result['x'] = pos[0]
        result['y'] = pos[1]
        result['z'] = pos[2]
        result['theta'] = compass - np.pi/2
        result['command_x'] = far_node[0]
        result['command_y'] = far_node[1]
        return result


    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        if len(self.data_queue) >= self.data_queue_len:
            self.data_queue.popleft()
        self.data_queue.append(tick_data)
        if self.step < self.data_queue_len:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.last_control = control
            return control
        
        if (self.step - self.data_queue_len) % 2 != 0:
            return self.last_control

        # Process
        ego_x = self.data_queue[-1]['x']
        ego_y = self.data_queue[-1]['y']
        ego_z = self.data_queue[-1]['z']
        ego_theta = self.data_queue[-1]['theta']
        speed = self.data_queue[-1]['speed']

        his_theta = []
        his_speed = []
        his_acceleration = []
        his_angular_velocity = []
        command_x = []
        command_y = []
        
        for index in range(len(self.data_queue)):
            if index % 2 == 0:
                continue
            R = np.array([
                [np.cos(ego_theta), np.sin(ego_theta)],
                [-np.sin(ego_theta),  np.cos(ego_theta)]
            ])
            command_x_y = np.array([self.data_queue[index]['command_x']-ego_x, self.data_queue[index]['command_y']-ego_y])
            command_x_y = R.dot(command_x_y)
            command_x.append(command_x_y[0])
            command_y.append(command_x_y[1])
            his_theta.append(self.data_queue[index]['theta']-ego_theta)
            his_speed.append(self.data_queue[index]['speed'])
            his_acceleration.append(self.data_queue[index]['acceleration'])
            his_angular_velocity.append(self.data_queue[index]['angular_velocity'])
            
        all_data = {}
        all_data['his_speed'] = np.array(his_speed, dtype=np.float32)
        all_data['his_acceleration'] = np.array(his_acceleration, dtype=np.float32)
        all_data['his_angular_velocity'] = np.array(his_angular_velocity, dtype=np.float32)
        all_data['his_theta'] = np.array(his_theta, dtype=np.float32)
        all_data['x_command_far'] = np.array(command_x, dtype=np.float32)
        all_data['y_command_far'] = np.array(command_y, dtype=np.float32)

        # change to tensor
        state_normalize = Normalize.get_instance(self.config.data.statistics_path)
        state = state_normalize.prepare_state(all_data).unsqueeze(0)
        texts = ["predict trajectory"]

        # prepare image
        rgb_front = self.data_queue[-1]['rgb_front']
        rgb_front_history = self.data_queue[-3]['rgb_front']

        images_front = self.image_preprocess(rgb_front).unsqueeze(0)
        images_front = einops.rearrange(images_front, "B H W C -> B C H W").unsqueeze(1)
        images_front_history = self.image_preprocess(rgb_front_history).unsqueeze(0)
        images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W").unsqueeze(1)

        images = torch.cat((images_front, images_front_history), dim=1)
        model_inputs = self.processor(text=texts, images=images)

        # build causal mask and position ids for trajectory
        causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": state_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": state.to(self.dtype),
        }
        image_text_proprio_mask, action_mask = (
                self.model.split_full_mask_into_submasks(causal_mask)
            )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pred = self.model.infer_action(**inputs)
        x_pred, y_pred = state_normalize.infer_traj(pred)
        # move to cpu
        x_pred = x_pred.cpu()
        y_pred = y_pred.cpu()
        pred = np.concatenate((x_pred[:, np.newaxis], y_pred[:, np.newaxis]), axis=-1)
        self.pred_traj_x = x_pred
        self.pred_traj_y = y_pred
        self.pred_traj_z = ego_z
        self.ego_x = ego_x
        self.ego_y = ego_y
        self.ego_theta = ego_theta

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pid_controller.control_pid(pred, speed)

        if self.debug:
            print('steer:', steer_traj)
            print('speed:', speed)
        
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control = carla.VehicleControl()
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        
        if abs(control.steer) > 0.15: ##0.15 In turning
            speed_threshold = 4.5 ## Avoid stuck during turning
        else:
            speed_threshold = 7.0 ## Avoid pass stop/red light/collision
        if float(tick_data['speed']) > speed_threshold:
            max_throttle = 0.05
        else:
            max_throttle = 0.75
        control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)
        
        if control.brake > 0.5:
            control.throttle = float(0)
            
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info

        if SAVE_PATH is not None and (self.step -  self.data_queue_len) % 2 == 0:
            self.save(tick_data, pred)

        self.last_control = control
        return control
    
    
    def ego2world(self, ego_theta, ego_x, ego_y, point_ego_x, point_ego_y):
        R_inv = np.array([
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta),  np.cos(ego_theta)]
        ])

        point_in_world = R_inv.dot(np.array([point_ego_x, point_ego_y]))
    
        point_in_world[0] += ego_x
        point_in_world[1] += ego_y
    
        return point_in_world

    def image_preprocess(self, rgb_image):
        resize_transform = T.Resize((224, 224))
        pil_image = Image.fromarray(rgb_image)
        image = resize_transform(pil_image)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image), dtype=torch.float32)
        image = torch.round(image)
        image = torch.clamp(image, 0, 255)
        image = image.to(torch.uint8)
        np_array = image.numpy()
        tf_tensor = tf.convert_to_tensor(np_array)
        image_augment_kwargs = dict(random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1],),
            random_brightness=[0.1],
            random_contrast=[0.9, 1.1],
            random_saturation=[0.9, 1.1],
            random_hue=[0.05],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
                ],
            )
        output = augment_image(image=tf_tensor, **image_augment_kwargs)
        np_image = output.numpy().astype(np.uint8)
        image = torch.as_tensor(np_image)
        return image
    
    def draw_traj_bev(self, traj, raw_img, canvas_size=(512,512), thickness=3, is_ego=False, hue_start=120, hue_end=80):
        if is_ego:
            line = np.concatenate([np.zeros((1,2)),traj],axis=0)
        else:
            line = traj
        img = raw_img.copy()        
        pts_4d = np.stack([line[:,0],line[:,1],np.zeros((line.shape[0])),np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask,0:2]
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        num_points = len(smoothed_pts)
        for i in range(num_points-1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
            if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
                cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
            elif i==0:
                break
        return img


    def save(self, tick_data, pred):
        frame = self.step // 2 - 5
        # images
        if self.save_images:
            Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
            image_bev_traj = self.draw_traj_bev(pred, tick_data['bev'])
            Image.fromarray(image_bev_traj).save(self.save_path / 'bev_traj' / ('%04d.png' % frame))

        # meta
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
    
    def destroy(self):
        del self.model
        del self.tokenizer
        del self.processor
        torch.cuda.empty_cache()
  
    def gps_to_location(self, gps):
        # gps content: numpy array: [lat, lon, alt]
        lat, lon, alt = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        z = alt
        return np.array([x, y, z])