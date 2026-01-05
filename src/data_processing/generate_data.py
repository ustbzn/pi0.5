import json
import os
import argparse
import multiprocessing as mp
import joblib
import argparse
from tqdm import *

from src.data_processing.utils import load_jpg, load_json_gz, load_npz


'''
Bench2Drive 数据预处理：
- 原始数据每条 route 下有多帧标注/相机图片/专家评估等。
- 这里把 route 划分为 train / val（val_list 硬编码），并把每条 route 组织成列表后序列化为 pkl。
输出：
- b2d_dynamic_camera/train/train_data.pkl
- b2d_dynamic_camera/val/val.pkl
后续 window.py 会按窗口切片并落盘单条样本。
'''

val_list = [  # 作为 open-loop 测试集的路线列表（硬编码）
    'StaticCutIn_Town05_Route226_Weather18',
    'MergerIntoSlowTrafficV2_Town12_Route857_Weather25',
    'YieldToEmergencyVehicle_Town04_Route166_Weather10',
    'ConstructionObstacle_Town10HD_Route74_Weather22',
    'VehicleTurningRoutePedestrian_Town15_Route445_Weather11',
    'VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21',
    'SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2',
    'LaneChange_Town06_Route307_Weather21',
    'ConstructionObstacleTwoWays_Town12_Route1093_Weather1',
    'HazardAtSideLaneTwoWays_Town12_Route1151_Weather7',
    'OppositeVehicleTakingPriority_Town04_Route214_Weather6',
    'NonSignalizedJunctionRightTurn_Town03_Route126_Weather18',
    'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9',
    'ParkedObstacle_Town06_Route282_Weather22',
    'ControlLoss_Town10HD_Route378_Weather14',
    'ControlLoss_Town04_Route170_Weather14',
    'OppositeVehicleRunningRedLight_Town04_Route180_Weather23',
    'InterurbanAdvancedActorFlow_Town06_Route324_Weather2',
    'HighwayCutIn_Town12_Route1029_Weather15',
    'MergerIntoSlowTraffic_Town06_Route317_Weather5',
    'NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3',
    'AccidentTwoWays_Town12_Route1115_Weather23',
    'ParkingCrossingPedestrian_Town13_Route545_Weather25',
    'VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8',
    'ParkingExit_Town12_Route922_Weather12',
    'VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23',
    'HardBreakRoute_Town01_Route32_Weather6',
    'DynamicObjectCrossing_Town01_Route3_Weather3',
    'ConstructionObstacle_Town12_Route78_Weather0',
    'EnterActorFlow_Town03_Route132_Weather2',
    'HazardAtSideLane_Town10HD_Route373_Weather9',
    'InvadingTurn_Town02_Route95_Weather9',
    'TJunction_Town05_Route260_Weather0',
    'VehicleTurningRoute_Town15_Route504_Weather10',
    'DynamicObjectCrossing_Town02_Route11_Weather11',
    'TJunction_Town06_Route306_Weather20',
    'ParkedObstacleTwoWays_Town13_Route1333_Weather26',
    'SignalizedJunctionRightTurn_Town03_Route118_Weather14',
    'NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13',
    'VehicleOpensDoorTwoWays_Town12_Route1203_Weather7',
    'CrossingBicycleFlow_Town12_Route977_Weather15',
    'SignalizedJunctionLeftTurn_Town04_Route173_Weather26',
    'HighwayExit_Town06_Route312_Weather0',
    'Accident_Town05_Route218_Weather10',
    'ParkedObstacle_Town10HD_Route372_Weather8',
    'InterurbanActorFlow_Town12_Route1291_Weather1',
    'ParkingCutIn_Town13_Route1343_Weather1',
    'VehicleTurningRoutePedestrian_Town15_Route481_Weather19',
    'PedestrianCrossing_Town13_Route747_Weather19',
    'BlockedIntersection_Town03_Route135_Weather5',
]

def get_data_info(route_path, cam_id_path):
    """
    读取单条 route 的所有帧，返回该 route 的序列数据（未切片）。
    - route_path: 形如 /.../<RouteName>，内部有 anno/ camera/ expert_assessment/ 等。
    - cam_id_path: camera_labels 根目录，用于读取 <RouteName>.json 的相机标签。
    """
    route_name = (route_path.split('/')[-1])
    cam_id_path = os.path.join(cam_id_path, route_name)
    anno_dir = os.path.join(route_path, "anno")
    # 只统计压缩标注文件，避免同时存在 .json/.json.gz 时重复计数
    anno_files = sorted(
        f for f in os.listdir(anno_dir)
        if f.endswith(".json.gz")
    )
    step_sum = len(anno_files)  # 帧数

    # 逐帧读取标注与专家评估
    anno_list = []
    for step in range(step_sum):
        anno_list.append(load_json_gz(os.path.join(route_path, "anno", f"{str(step).zfill(5)}.json.gz")))

    # 为每帧构建六路相机图片路径（始终保留，不受 cam_id 标签影响）
    camera_front_list = []
    for step in range(step_sum):
        camera_front_list.append(os.path.join(route_path, "camera", "rgb_front", f"{str(step).zfill(5)}.jpg"))
    
    camera_front_left_list = []
    for step in range(step_sum):
        camera_front_left_list.append(os.path.join(route_path, "camera", "rgb_front_left", f"{str(step).zfill(5)}.jpg"))
    
    camera_front_right_list = []
    for step in range(step_sum):
        camera_front_right_list.append(os.path.join(route_path, "camera", "rgb_front_right", f"{str(step).zfill(5)}.jpg"))
    
    camera_back_list = []
    for step in range(step_sum):
        camera_back_list.append(os.path.join(route_path, "camera", "rgb_back", f"{str(step).zfill(5)}.jpg"))
    
    camera_back_left_list = []
    for step in range(step_sum):
        camera_back_left_list.append(os.path.join(route_path, "camera", "rgb_back_left", f"{str(step).zfill(5)}.jpg"))
    
    camera_back_right_list = []
    for step in range(step_sum):
        camera_back_right_list.append(os.path.join(route_path, "camera", "rgb_back_right", f"{str(step).zfill(5)}.jpg"))

    # 专家评估（步号从 -1 开始填充，首帧会对齐下一步）
    expert_list = []  # 每帧专家动作评估（长度等于 step_sum）
    for step in range(step_sum):
        expert_list.append(load_npz(os.path.join(route_path, "expert_assessment", f"{str(step-1).zfill(5)}.npz")))
    
    # 相机标签：每帧一个 cam_id（可为 NULL），缺文件则跳过该 route
    cam_id_list = []
    try:
        with open((cam_id_path + '.json'), 'r') as f:
            cam_data = json.load(f)
            if(len(cam_data) != step_sum):
                raise Exception(f'cam id num in {route_name} is wrong!')
            for step in range(step_sum):
                cam_id_list.append(cam_data[f"{step}"])            
    except FileNotFoundError:
        print(f"file {cam_id_path + '.json'} does not exist, skip it")
        return None

    # 从目录名解析元信息
    scenario_id = route_path.split('/')[-1].split('_')[0]
    town_id = route_path.split('/')[-1].split('_')[1]
    route_id = route_path.split('/')[-1].split('_')[2]
    weather_id = route_path.split('/')[-1].split('_')[3]

    episode_x_list = []
    episode_y_list = []
    episode_x_pure_list = []
    episode_y_pure_list = []
    episode_theta_list = []
    episode_speed_list = []
    episode_x_command_far_list = []
    episode_y_command_far_list = []
    episode_command_far_list = []
    episode_x_command_near_list = []
    episode_y_command_near_list = []
    episode_command_near_list = []
    episode_x_target_list = []
    episode_y_target_list = []
    episode_acceleration_list = []
    episode_angular_velocity_list = []
    episode_action_list = []

    # 将标注字段展开为列表（对齐 step_sum）
    for anno in anno_list:
        episode_x_list.append(anno['x'])
        episode_y_list.append(anno['y'])
        episode_x_pure_list.append(anno['bounding_boxes'][0]['location'][0])
        episode_y_pure_list.append(anno['bounding_boxes'][0]['location'][1])
        episode_theta_list.append(anno['theta'])
        episode_speed_list.append(anno['speed'])
        episode_x_command_far_list.append(anno['x_command_far'])
        episode_y_command_far_list.append(anno['y_command_far'])
        episode_command_far_list.append(anno['command_far'])
        episode_x_command_near_list.append(anno['x_command_near'])
        episode_y_command_near_list.append(anno['y_command_near'])
        episode_command_near_list.append(anno['command_near'])
        episode_x_target_list.append(anno['x_target'])
        episode_y_target_list.append(anno['y_target'])
        episode_angular_velocity_list.append(anno['angular_velocity'])
        episode_acceleration_list.append(anno['acceleration'])
    
    # 观测长度比动作多 1，因此去掉最后一帧观测，使 obs_t + act_t -> obs_{t+1} 对齐
    episode_image_front_list = camera_front_list[:-1]
    episode_image_front_left_list = camera_front_left_list[:-1]
    episode_image_front_right_list = camera_front_right_list[:-1]
    episode_image_back_list = camera_back_list[:-1]
    episode_image_back_left_list = camera_back_left_list[:-1]
    episode_image_back_right_list = camera_back_right_list[:-1]
    
    cam_id_list = cam_id_list[:-1]

    # 动作序列：去掉首帧（首帧无前一动作）
    for expert in expert_list:
        episode_action_list.append(expert[-1])
    
    # obs + action -> next obs，对齐长度
    episode_x_list = episode_x_list[:-1]
    episode_y_list = episode_y_list[:-1]
    episode_x_pure_list = episode_x_pure_list[:-1]
    episode_y_pure_list = episode_y_pure_list[:-1]

    episode_theta_list = episode_theta_list[:-1]
    episode_speed_list = episode_speed_list[:-1]
    episode_x_command_far_list = episode_x_command_far_list[:-1]
    episode_y_command_far_list = episode_y_command_far_list[:-1]
    episode_command_far_list = episode_command_far_list[:-1]
    episode_x_command_near_list = episode_x_command_near_list[:-1]
    episode_y_command_near_list = episode_y_command_near_list[:-1]
    episode_command_near_list = episode_command_near_list[:-1]
    episode_x_target_list = episode_x_target_list[:-1]
    episode_y_target_list = episode_y_target_list[:-1]
    episode_acceleration_list = episode_acceleration_list[:-1]
    episode_angular_velocity_list = episode_angular_velocity_list[:-1]
    episode_action_list = episode_action_list[1:]

    return  episode_x_list, \
            episode_y_list, \
            episode_x_pure_list, \
            episode_y_pure_list, \
            episode_theta_list, \
            episode_speed_list, \
            episode_x_command_far_list, \
            episode_y_command_far_list, \
            episode_command_far_list, \
            episode_x_command_near_list, \
            episode_y_command_near_list, \
            episode_command_near_list, \
            episode_x_target_list, \
            episode_y_target_list, \
            episode_acceleration_list, \
            episode_angular_velocity_list, \
            episode_action_list, \
            episode_image_front_list, \
            episode_image_front_left_list, \
            episode_image_front_right_list, \
            episode_image_back_list, \
            episode_image_back_left_list, \
            episode_image_back_right_list, \
            cam_id_list, \
            scenario_id, \
            town_id, \
            route_id, \
            weather_id


def deal_with_data(data_path: str, cam_id_path: str, work_dir: str, target_dir_name: str):
    train_data = mp.Manager().list()
    val_data = mp.Manager().list()
    for dirpath, dirnames, filenames in os.walk(data_path):
        for name in tqdm(dirnames):
            if name in val_list:
                val_data.append(get_data_info(os.path.join(dirpath, name), cam_id_path))
            else:
                train_data.append(get_data_info(os.path.join(dirpath, name), cam_id_path))
        break
    # save data
    target_dir = os.path.join(work_dir, target_dir_name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    train_file_path = os.path.join(target_dir, 'train', 'train_data.pkl')
    with open(train_file_path, 'wb') as file:
        joblib.dump(list(train_data), file)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    val_file_path = os.path.join(target_dir, 'val', 'val.pkl')
    with open(val_file_path, 'wb') as file:
        joblib.dump(list(val_data), file)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--cam_id_path", type=str, required=True)
    parser.add_argument("--target_dir", type=str, default='b2d_dynamic_camera', help="relative output directory under work_dir")
    args = parser.parse_args()
    bench2drive_path = args.dataset_path
    work_dir = args.work_dir
    cam_id_path = args.cam_id_path
    target_dir_name = args.target_dir
    print('------------------------------dealing with data------------------------------')
    deal_with_data(data_path=bench2drive_path, cam_id_path=cam_id_path, work_dir=work_dir, target_dir_name=target_dir_name)