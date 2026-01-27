'''
Senna driving QA dataset based on nuScenes
using LlaVA format
using LLaVA-1.6-34b for generate surround img description
'''

import re
import os
import uuid
import json
import math
import copy
import requests
import argparse
from io import BytesIO
from os import path as osp

import sys
senna_path = '/D2/NXT/vlm/miaobinqi/Senna'
if senna_path not in sys.path:
    sys.path.append(senna_path)

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from llava_next.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava_next.llava.conversation import conv_templates, SeparatorStyle
from llava_next.llava.utils import disable_torch_init
from data_tools.senna_qa_utils import get_obj_acc_or_dec, \
    get_obj_turn_or_lane_change, get_obj_rel_position, load_image

idx_class_mapping = ["car", "cyclist", "pedestrian"]

pedal_status = {
    'const': 'KEEP',
    'accelerate': 'ACCELERATE',
    'decelerate': 'DECELERATE',
    'stop': 'STOP'
}

path_status = {
    'right turn': 'RIGHT_TURN',
    'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN',
    'left lane change': 'LEFT_CHANGE',
    'straight': 'STRAIGHT'
}

def _get_image_format(image_path: str) -> str:
    """获取图片格式"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif']:
        return ext[1:]  # 去掉点号
    raise ValueError(f"不支持的图片格式: {ext}")


def encode_image_to_base64(image_path: str) -> str:
    import base64
    """将图片文件编码为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/{_get_image_format(image_path)};base64,{encoded}"
    except FileNotFoundError:
        raise ValueError(f"图片文件未找到: {image_path}")
    except Exception as e:
        raise RuntimeError(f"图片编码失败: {str(e)}")

#     args = type('Args', (), {"query": question, "img_file": img_path,})()
def eval_doubao_init(args):
    from volcenginesdkarkruntime import Ark
    api_key = '0116c602-09b1-459a-a5bf-5555aa95ddea'
    #api_key = 'b9c7a721-5e2c-4bfe-ae4f-8be7628ccb13'
    client = Ark(api_key=api_key)
    image = encode_image_to_base64(args.img_file)
    question = args.query
    model="doubao-1.5-vision-pro-250328"
    message = [
        {
            "role":"user",
            "content":[
                {
                "type":"image_url",
                "image_url":{
                    "url":image,
                }
                },

                {
                    "type":"text",
                    "text":question,
                }
            ]
        }
     ] 
    response = client.chat.completions.create(
        model=model,
        messages=message
    )
    text_outputs = response.choices[0].message.content  
    # print('doubao output:',text_outputs)  
    return text_outputs


def eval_llava_34b_wo_init(args, tokenizer, model, image_processor):
    # Model
    disable_torch_init()
    image = load_image(args.img_file)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]

    conv_template = "chatml_direct" # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + args.query
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=512,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    print('llava output:',text_outputs)
    return text_outputs


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

ego_width, ego_length = 1.85, 4.084

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                        #   version='v1.0-trainval',
                          version='v1.0-mini',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    # 功能：从 nuScenes 原始数据生成训练/验证信息文件（json），包含传感器、轨迹、标注等详细信息。
    # 实现：
    # 加载 nuScenes 数据和 CAN bus 信息
    # 按场景划分训练/验证集
    # 遍历每一帧，提取激光雷达、相机、标注、轨迹等信息
    # 处理多帧历史/未来轨迹、目标检测框、速度、类别等
    # 生成多模态QA数据（调用 generate_drive_qa）
    # 保存为 json 文件
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps,output_path=out_path)

    metadata = dict(version=version)

    if test:
        # print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.json'.format(info_prefix))
        with open(info_path, "w") as f:
            json.dump(data, f)
    else:
        info_path = osp.join(out_path,
                             '{}_train.json'.format(info_prefix))
        with open(info_path, "w") as f:
            json.dump(train_nusc_infos, f)
        info_val_path = osp.join(out_path,
                                 '{}_val.json'.format(info_prefix))
        with open(info_val_path, "w") as f:
            json.dump(val_nusc_infos, f)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

json_count = 0
def _get_can_bus_info(nusc, nusc_can_bus, sample):
    # 功能：获取每帧的 CAN bus 信息（如车辆速度、方向等）。
    # 实现：根据时间戳匹配最近的 CAN bus 数据。
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=6,
                         his_ts=2,
                         use_doubao=True,
                         output_path=None):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # 功能：生成训练/验证集的详细信息列表。
    # 实现：遍历每一帧，提取并处理所有传感器、标注、轨迹、未来目标等信息，生成 QA 数据。
    global json_count
    if not use_doubao:
        # load vlm model and generate
        from llava_next.llava.model.builder import load_pretrained_model
        pretrained = "/D2/NXT/huy/hugging/llava-v1.6-34b-2"
        #pretrained = "/D2/NXT/huy/hugging/qwen2.5-72b"
        #model_name = "qwen-v-72b"
        model_name = "llava-v1.6-34b"
        device_map = "auto"
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            device_map=device_map,
            # load_4bit=True,
            load_8bit=True,
            attn_implementation=None)
        model.eval()
        model.tie_weights()
    else:
        tokenizer, model, image_processor = None,None,None

    
    # init json count 
    def count_json_file(json_path):
        count = 0
        for filename in os.listdir(json_path):
            if filename.endswith('.json'):
                count += 1
        return count

    json_count = count_json_file(output_path)


    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for dic in nusc.category:
        cat2idx[dic['name']] = dic['index']
    count = 0
    for sample in tqdm(nusc.sample):
        # print('0 sample:',sample)
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        #print('lidar path:',lidar_path)

        # 查看当前处理的sample信息
        print(f"Processing sample 001: {sample['token']}")
  
        # assert os.path.isfile(lidar_path)
        if not os.path.isfile(lidar_path):
            continue
        
        if count < json_count:
            count+=1
            continue
        
        print(f"Test state: {test}")
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            print(f"Processing sample 002: {sample['token']}")
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            # get future coords for each box
            # [num_box, fut_ts*2]
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_trajs_vcs = np.zeros((num_box, fut_ts, 2))
            gt_fut_yaw = np.zeros((num_box, fut_ts))
            gt_fut_yaw_vcs = np.zeros((num_box, fut_ts))
            gt_fut_masks = np.zeros((num_box, fut_ts))
            gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)
            # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
            agent_lcf_feat = np.zeros((num_box, 9))
            gt_fut_goal = np.zeros((num_box))
            for i, anno in enumerate(annotations):
                cur_box = boxes[i]
                box_lcf_vcs = Box(anno['translation'],
                                  anno['size'],
                                  Quaternion(anno['rotation']))
                box_lcf_vcs.translate(-np.array(cs_record['translation']))
                box_lcf_vcs.rotate(Quaternion(cs_record['rotation']).inverse)
                cur_anno = anno
                lcf_anno = copy.deepcopy(anno)
                agent_lcf_feat[i, 0:2] = cur_box.center[:2]	
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                agent_lcf_feat[i, 3:5] = velocity[i]
                agent_lcf_feat[i, 5:8] = anno['size'] # width,length,height
                agent_lcf_feat[i, 8] = cat2idx[anno['category_name']] if anno['category_name'] in cat2idx.keys() else -1
                for j in range(fut_ts):
                    if cur_anno['next'] != '':
                        anno_next = nusc.get('sample_annotation', cur_anno['next'])
                        box_next = Box(
                            anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                        )
                        # Move box to vehicle lcf coord system.
                        box_next_vcs = copy.deepcopy(box_next)
                        box_next_vcs.translate(-np.array(lcf_anno['translation']))
                        box_next_vcs.rotate(Quaternion(lcf_anno['rotation']).inverse)
                        # Move box to sensor coord system.
                        box_next_vcs.translate(-np.array(cs_record['translation']))
                        box_next_vcs.rotate(Quaternion(cs_record['rotation']).inverse)
                        # Move box to ego vehicle coord system.
                        box_next.translate(-np.array(pose_record['translation']))
                        box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                        # Move box to sensor coord system.
                        box_next.translate(-np.array(cs_record['translation']))
                        box_next.rotate(Quaternion(cs_record['rotation']).inverse)
                        gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                        gt_fut_masks[i, j] = 1
                        gt_fut_trajs_vcs[i, j] = box_next_vcs.center[:2]
                        # add yaw diff
                        _, _, box_yaw = quart_to_rpy([cur_box.orientation.x, cur_box.orientation.y,
                                                      cur_box.orientation.z, cur_box.orientation.w])
                        _, _, box_yaw_next = quart_to_rpy([box_next.orientation.x, box_next.orientation.y,
                                                           box_next.orientation.z, box_next.orientation.w])
                        gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                        _, _, box_lcf_yaw_vcs = quart_to_rpy([box_lcf_vcs.orientation.x, box_lcf_vcs.orientation.y,
                                                           box_lcf_vcs.orientation.z, box_lcf_vcs.orientation.w])
                        _, _, box_yaw_vcs_next = quart_to_rpy([box_next_vcs.orientation.x, box_next_vcs.orientation.y,
                                                           box_next_vcs.orientation.z, box_next_vcs.orientation.w])
                        gt_fut_yaw_vcs[i, j] = box_yaw_vcs_next - box_lcf_yaw_vcs

                        cur_anno = anno_next
                        cur_box = box_next
                    else:
                        gt_fut_trajs[i, j:] = 0
                        break
                # get agent goal
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                if coord_diff.max() < 1.0: # static
                    gt_fut_goal[i] = 9
                else:
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

            # get ego history traj (offset format)
            ego_his_trajs = np.zeros((his_ts+1, 3))
            ego_his_trajs_diff = np.zeros((his_ts+1, 3))
            sample_cur = sample
            for i in range(his_ts, -1, -1):
                if sample_cur is not None:
                    pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                    ego_his_trajs[i] = pose_mat[:3, 3]
                    has_prev = sample_cur['prev'] != ''
                    has_next = sample_cur['next'] != ''
                    if has_next:
                        sample_next = nusc.get('sample', sample_cur['next'])
                        pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                        ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                    sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
                else:
                    ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                    ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]
            
            # global to ego at lcf
            ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            # ego to lidar at lcf
            ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

            # get ego futute traj (offset format)
            ego_fut_trajs = np.zeros((fut_ts+1, 3))
            ego_fut_masks = np.zeros((fut_ts+1))
            sample_cur = sample
            for i in range(fut_ts+1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            # global to ego at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # Save current frame position (starting point) before converting to offset format
            # This will have small non-zero values due to floating point precision
            current_frame_pos = ego_fut_trajs[0, :2].copy()  # shape: (2,)
            # drive command according to final fut step offset from lcf
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            # offset from lcf -> per-step offset
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]


            ### Senna: get navi info from long-horzion (6s) future trajs
            long_fut_ts = 12
            ego_navi_trajs = np.zeros((long_fut_ts+1, 3))
            sample_cur = sample
            for i in range(13):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_navi_trajs[i] = pose_mat[:3, 3]
                if sample_cur['next'] == '':
                    ego_navi_trajs[i+1:] = ego_navi_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            # global to ego at lcf
            ego_navi_trajs = ego_navi_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T
            # ego to lidar at lcf
            ego_navi_trajs = ego_navi_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T#当前帧+之后12帧的所有坐标
            # drive command according to final fut step offset from lcf
            
            ego_navi_trajs = ego_navi_trajs[:, :2]
            ego_navi_trajs = ego_navi_trajs[1:, :2] # discard current timestamp
            target_point = ego_navi_trajs[[-1], :2]
            # if target_point[0, 0] >= 20.0 and target_point[0, 1] >= 10.0:
            #     ego_navi_cmd = 'go straight and turn left'
            # elif target_point[0, 0] >= 20.0 and target_point[0, 1] <= -10.0:
            #     ego_navi_cmd = 'go straight and turn right'
            # elif target_point[0, 0] < 20.0 and target_point[0, 1] >= 10.0:
            #     ego_navi_cmd = 'turn left'
            # elif target_point[0, 0] < 20.0 and target_point[0, 1] <= -10.0:
            #     ego_navi_cmd = 'turn right'
            # else:
            #     ego_navi_cmd = 'go straight'
            # print('target 1:',target_point[0,1],',target  0:',target_point[0,0])
            # 坐标系转换
            target_point_x = target_point[0, 1]
            target_point_y = -target_point[0, 0]
            if target_point_x >= 20.0 and target_point_y >= 10.0:
                ego_navi_cmd = 'go straight and turn left'
            elif target_point_x >= 20.0 and target_point_y <= -10.0:
                ego_navi_cmd = 'go straight and turn right'
            elif target_point_x < 20.0 and target_point_y >= 10.0:
                ego_navi_cmd = 'turn left'
            elif target_point_x < 20.0 and target_point_y <= -10.0:
                ego_navi_cmd = 'turn right'
            else:
                ego_navi_cmd = 'go straight'
            # print('ego_navi_cmd:')


            ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer)
            ego_lcf_feat = np.zeros(9)
            _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
            ego_pos = np.array(pose_record['translation'])
            if pose_record_prev is not None:
                _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
                ego_pos_prev = np.array(pose_record_prev['translation'])
            if pose_record_next is not None:
                _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
                ego_pos_next = np.array(pose_record_next['translation'])
            assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'
            if pose_record_prev is not None:
                ego_w = (ego_yaw - ego_yaw_prev) / 0.5
                ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
            else:
                ego_w = (ego_yaw_next - ego_yaw) / 0.5
                ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

            ref_scene = nusc.get("scene", sample['scene_token'])
            try:
                pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
                steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
                pose_uts = [msg['utime'] for msg in pose_msgs]
                steer_uts = [msg['utime'] for msg in steer_msgs]
                ref_utime = sample['timestamp']
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                steer_index = locate_message(steer_uts, ref_utime)
                steer_data = steer_msgs[steer_index]
                # initial speed2
                v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
                # curvature (positive: turn left)
                steering = steer_data["value"]
                # flip x axis if in left-hand traffic (singapore)
                flip_flag = True if map_location.startswith('singapore') else False
                if flip_flag:
                    steering *= -1
                Kappa = 2 * steering / 2.588
            except:
                delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                v0 = np.sqrt(delta_x**2 + delta_y**2)
                Kappa = 0

            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy]) #can_bus[13:15]
            ego_lcf_feat[2:4] = can_bus[7:9]
            ego_lcf_feat[4] = ego_w #can_bus[12]
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = v0
            ego_lcf_feat[8] = Kappa

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)

            info['gt_agent_fut_trajs_vcs'] = gt_fut_trajs_vcs.astype(np.float32)
            info['gt_agent_fut_yaw_vcs'] = gt_fut_yaw_vcs.astype(np.float32)


            ######## DrivePi-Senna data processing ########
            
            # ========== 检查未来帧是否足够 ==========
            # 未来轨迹需要7个点：当前帧 + 6个未来帧
            # 如果未来帧不足，跳过该sample，不生成数据
            fut_ts_needed = 6  # 需要6个未来帧
            sample_cur = sample
            has_enough_future_frames = True
            for i in range(fut_ts_needed):
                if sample_cur['next'] == '':
                    # 未来帧不足，标记并跳出检查循环
                    has_enough_future_frames = False
                    break
                sample_cur = nusc.get('sample', sample_cur['next'])
            
            # 如果未来帧数量不足，跳过该sample
            if not has_enough_future_frames:
                continue
            
            # ========== 字段1: scene_name (场景名称，用于文件命名) ==========
            scene_rec = nusc.get('scene', sample['scene_token'])
            scene_name = scene_rec['name']
            
            # ========== 字段2: images (当前帧的6个相机图像路径) ==========
            camera_types = ['CAM_FRONT',
                            'CAM_FRONT_RIGHT',
                            'CAM_FRONT_LEFT',
                            'CAM_BACK',
                            'CAM_BACK_LEFT',
                            'CAM_BACK_RIGHT']
            info['images'] = [info['cams'][idx]['data_path'] for idx in camera_types]
            
            # ========== 字段3: his_image, his_images (历史帧图像路径) ==========
            # 获取前一帧的图像路径，如果不存在则使用当前帧图像
            his_image = info['images'][0]  # Default to current frame
            his_images = info['images'].copy()  # Default to current frame images
            if sample['prev'] != '':
                sample_prev = nusc.get('sample', sample['prev'])
                his_images = []
                for cam in camera_types:
                    cam_token = sample_prev['data'][cam]
                    cam_path, _, _ = nusc.get_sample_data(cam_token)
                    his_images.append(str(cam_path))
                his_image = his_images[0] if his_images else info['images'][0]
            
            # ========== 字段4: meta_action, meta_action_x, meta_action_y (meta action命令和坐标) ==========
            # 计算6帧后的meta action点坐标和命令
            meta_action_ts = 6
            meta_action_trajs = np.zeros((meta_action_ts+1, 3))
            sample_cur = sample
            for i in range(meta_action_ts+1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                meta_action_trajs[i] = pose_mat[:3, 3]
                if sample_cur['next'] == '':
                    meta_action_trajs[i+1:] = meta_action_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            # 转换到自车坐标系
            meta_action_trajs = meta_action_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            meta_action_trajs = np.dot(rot_mat, meta_action_trajs.T).T
            meta_action_trajs = meta_action_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            meta_action_trajs = np.dot(rot_mat, meta_action_trajs.T).T
            meta_action_trajs = meta_action_trajs[1:, :2]  # discard current timestamp
            meta_action_point = meta_action_trajs[meta_action_ts-1, :2]  # 6 frames ahead (index 5, since we removed first frame)
            
            # 根据6帧后的坐标计算meta action命令
            meta_action_x = meta_action_point[1]  # swap x,y
            meta_action_y = -meta_action_point[0]
            if meta_action_x >= 20.0 and meta_action_y >= 10.0:
                meta_action_cmd = 'go straight and turn left'
            elif meta_action_x >= 20.0 and meta_action_y <= -10.0:
                meta_action_cmd = 'go straight and turn right'
            elif meta_action_x < 20.0 and meta_action_y >= 10.0:
                meta_action_cmd = 'turn left'
            elif meta_action_x < 20.0 and meta_action_y <= -10.0:
                meta_action_cmd = 'turn right'
            else:
                meta_action_cmd = 'go straight'
            
            # ========== 字段5: navigation_x, navigation_y (导航点坐标，12帧后) ==========
            # ego_navi_trajs已经在前面计算过，形状为(12, 2)，索引11对应第12帧
            navi_point = ego_navi_trajs[11, :2]  # 12 frames ahead (last element, index 11)
            
            # ========== 字段6-9: his_speed, his_acceleration, his_angular_velocity, his_theta (历史状态特征) ==========
            # 历史帧数量：总共5帧（4个历史帧 + 1个当前帧）
            # 顺序：[4帧前, 3帧前, 2帧前, 1帧前, 当前帧]
            history_frames = 5
            num_history_frames = history_frames - 1  # 4 history frames (excluding current)
            
            # 步骤1: 收集所有历史帧和当前帧（按时间顺序）
            all_frames = []
            sample_cur = sample
            for _ in range(num_history_frames):
                if sample_cur['prev'] != '':
                    sample_cur = nusc.get('sample', sample_cur['prev'])
                    all_frames.insert(0, sample_cur)  # Insert at beginning to keep oldest first
                else:
                    break
            all_frames.append(sample)  # Add current frame at the end
            
            # 步骤2: 收集每帧的位置、旋转、速度数据
            frame_data = []
            dt = 0.5
            current_pose_mat = get_global_sensor_pose(sample, nusc, inverse=False)
            current_pos = current_pose_mat[:3, 3]
            _, _, current_yaw = quart_to_rpy(pose_record['rotation'])
            
            # 获取下一帧位置（用于计算当前帧速度）
            next_pos = None
            if sample['next'] != '':
                sample_next = nusc.get('sample', sample['next'])
                next_pose_mat = get_global_sensor_pose(sample_next, nusc, inverse=False)
                next_pos = next_pose_mat[:3, 3]
            
            for i, frame_sample in enumerate(all_frames):
                sd_rec = nusc.get('sample_data', frame_sample['data']['LIDAR_TOP'])
                pose_record_frame = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                pose_mat = get_global_sensor_pose(frame_sample, nusc, inverse=False)
                pos = pose_mat[:3, 3]
                rot = Quaternion(pose_record_frame['rotation'])
                roll, pitch, yaw = quart_to_rpy(pose_record_frame['rotation'])
                
                # 计算速度（通过位置差分）
                if i + 1 < len(all_frames):
                    next_frame_pos = get_global_sensor_pose(all_frames[i + 1], nusc, inverse=False)[:3, 3]
                    vel_global = (next_frame_pos - pos) / dt
                elif i == len(all_frames) - 1 and next_pos is not None:
                    vel_global = (next_pos - pos) / dt
                else:
                    vel_global = np.zeros(3)
                
                frame_data.append({
                    'pos': pos,
                    'rot': rot,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'vel_global': vel_global
                })
            
            # 步骤3: 如果历史帧不足5帧，用第一帧的数据进行填充（与window.py的state_window逻辑一致）
            # state_window使用tf.maximum(history_indices, 0)，会重复第一帧（索引0）
            # 对于window_size=5，历史索引为[t-4, t-3, t-2, t-1, t]，不足时填充为[0, 0, ..., t-1, t]
            if len(frame_data) > 0:
                first_frame = frame_data[0]  # Use the oldest available frame for padding
                while len(frame_data) < history_frames:
                    # Deep copy the first frame data to avoid reference issues
                    frame_data.insert(0, {
                        'pos': first_frame['pos'].copy() if isinstance(first_frame['pos'], np.ndarray) else first_frame['pos'],
                        'rot': first_frame['rot'],
                        'roll': first_frame['roll'],
                        'pitch': first_frame['pitch'],
                        'yaw': first_frame['yaw'],
                        'vel_global': first_frame['vel_global'].copy() if isinstance(first_frame['vel_global'], np.ndarray) else first_frame['vel_global']
                    })
            else:
                # 如果完全没有历史帧，使用当前帧数据填充
                while len(frame_data) < history_frames:
                    frame_data.insert(0, {
                        'pos': current_pos,
                        'rot': Quaternion(pose_record['rotation']),
                        'roll': 0, 'pitch': 0, 'yaw': current_yaw,
                        'vel_global': np.zeros(3)
                    })

            # 步骤4: 追加下一帧数据，用于计算当前帧的加速度和角速度
            # 这样可以保持历史窗口长度为history_frames，同时允许对最后一帧（当前帧）使用前向差分
            if sample['next'] != '':
                sample_next_full = nusc.get('sample', sample['next'])
                sd_next_full = nusc.get('sample_data', sample_next_full['data']['LIDAR_TOP'])
                pose_next_full = nusc.get('ego_pose', sd_next_full['ego_pose_token'])
                pose_mat_next_full = get_global_sensor_pose(sample_next_full, nusc, inverse=False)
                pos_next_full = pose_mat_next_full[:3, 3]
                rot_next_full = Quaternion(pose_next_full['rotation'])
                roll_next_full, pitch_next_full, yaw_next_full = quart_to_rpy(pose_next_full['rotation'])

                # 计算下一帧的速度：使用再下一帧的位置（如果存在）
                # 未来帧足够（需要6个未来帧），再下一帧肯定存在
                if sample_next_full['next'] != '':
                    sample_next_next = nusc.get('sample', sample_next_full['next'])
                    pose_mat_next_next = get_global_sensor_pose(sample_next_next, nusc, inverse=False)
                    pos_next_next = pose_mat_next_next[:3, 3]
                    vel_next_full = (pos_next_next - pos_next_full) / dt
                else:
                    # 如果下一帧是最后一帧，使用当前帧到下一帧的位移（虽然不准确，但至少不是0）
                    vel_next_full = (pos_next_full - current_pos) / dt

                frame_data.append({
                    'pos': pos_next_full,
                    'rot': rot_next_full,
                    'roll': roll_next_full,
                    'pitch': pitch_next_full,
                    'yaw': yaw_next_full,
                    'vel_global': vel_next_full
                })
            
            # 步骤5: 计算历史特征（his_speed, his_acceleration, his_angular_velocity, his_theta）
            rot_mat_global_to_ego = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            
            def normalize_angle(angle):
                """Normalize angle to [-pi, pi]"""
                while angle > np.pi:
                    angle -= 2 * np.pi
                while angle < -np.pi:
                    angle += 2 * np.pi
                return angle
            
            # 初始化历史特征数组
            # 字段6: 历史速度，shape=(5,)
            his_speed = np.zeros(history_frames, dtype=np.float32)
            # 字段7: 历史加速度，shape=(5,3)，[x,y,z]
            his_acceleration = np.zeros((history_frames, 3), dtype=np.float32)
            # 字段8: 历史角速度，shape=(5,3)，[roll,pitch,yaw]
            his_angular_velocity = np.zeros((history_frames, 3), dtype=np.float32)
            # 字段9: 历史航向角（相对于当前帧），shape=(5,)
            his_theta = np.zeros(history_frames, dtype=np.float32)
            
            for i in range(history_frames):
                frame = frame_data[i]
                vel_global = frame['vel_global']
                
                # 字段6: his_speed - 计算速度大小（自车坐标系下的横向和纵向速度的模）
                vel_ego = np.dot(rot_mat_global_to_ego, vel_global)
                his_speed[i] = np.linalg.norm(vel_ego[:2])
                
                # 字段7: his_acceleration - 计算加速度（通过速度差分）
                # 由于已经检查过未来帧足够，frame_data应该包含下一帧（长度为6）
                # 当前帧索引为history_frames-1（即4），下一帧索引为5
                if i + 1 < len(frame_data):
                    # 有下一帧数据，使用下一帧的速度计算加速度
                    next_vel_global = frame_data[i + 1]['vel_global']
                    acc_global = (next_vel_global - vel_global) / dt
                    acc_ego = np.dot(rot_mat_global_to_ego, acc_global)
                    his_acceleration[i] = acc_ego
                else:
                    # 理论上不应该到这里（因为已检查过未来帧足够且已追加下一帧）
                    # 但为了安全起见，如果下一帧不在frame_data中，使用next_pos计算
                    if next_pos is not None:
                        next_vel_global = (next_pos - frame['pos']) / dt
                        acc_global = (next_vel_global - vel_global) / dt
                        acc_ego = np.dot(rot_mat_global_to_ego, acc_global)
                        his_acceleration[i] = acc_ego
                    else:
                        # 如果完全没有下一帧信息，设为0（理论上不应该发生）
                        his_acceleration[i] = np.zeros(3)
                
                # 字段8: his_angular_velocity - 计算角速度（通过旋转角度差分）
                if i + 1 < len(frame_data):
                    next_frame = frame_data[i + 1]
                    droll = normalize_angle(next_frame['roll'] - frame['roll'])
                    dpitch = normalize_angle(next_frame['pitch'] - frame['pitch'])
                    dyaw = normalize_angle(next_frame['yaw'] - frame['yaw'])
                    his_angular_velocity[i] = np.array([droll, dpitch, dyaw]) / dt
                elif i == len(frame_data) - 1 and sample['next'] != '':
                    # Current frame: use next frame
                    sample_next = nusc.get('sample', sample['next'])
                    next_sd_rec = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
                    next_pose_record = nusc.get('ego_pose', next_sd_rec['ego_pose_token'])
                    next_roll, next_pitch, next_yaw = quart_to_rpy(next_pose_record['rotation'])
                    droll = normalize_angle(next_roll - frame['roll'])
                    dpitch = normalize_angle(next_pitch - frame['pitch'])
                    dyaw = normalize_angle(next_yaw - frame['yaw'])
                    his_angular_velocity[i] = np.array([droll, dpitch, dyaw]) / dt
                
                # 字段9: his_theta - 计算航向角（相对于当前帧的航向角变化）
                his_theta[i] = normalize_angle(frame['yaw'] - current_yaw)
            
            # ========== 字段10: future_x, future_y (未来轨迹，7个点：当前帧 + 6个未来帧) ==========
            # fut_ts_needed已在前面定义（6个未来帧），这里直接使用
            ego_fut_trajs_abs = np.zeros((fut_ts_needed + 1, 3))
            sample_cur = sample
            for i in range(fut_ts_needed + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_fut_trajs_abs[i] = pose_mat[:3, 3]
                if sample_cur['next'] == '':
                    ego_fut_trajs_abs[i+1:] = ego_fut_trajs_abs[i]
                    break
                sample_cur = nusc.get('sample', sample_cur['next'])
            
            # 转换到自车坐标系
            ego_fut_trajs_abs = ego_fut_trajs_abs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs_abs = np.dot(rot_mat, ego_fut_trajs_abs.T).T
            ego_fut_trajs_abs = ego_fut_trajs_abs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs_abs = np.dot(rot_mat, ego_fut_trajs_abs.T).T
            
            # 提取未来轨迹点：当前帧 + fut_ts_needed个未来帧
            num_future_points = fut_ts_needed + 1  # 总点数 = 当前帧 + 未来帧数
            future_trajs = ego_fut_trajs_abs[:num_future_points, :2]  # shape: (num_future_points, 2)
            # 反转一下，与 navigation/meta_action 一致
            future_x = future_trajs[:, 1].reshape(1, num_future_points).astype(np.float32)
            future_y = future_trajs[:, 0].reshape(1, num_future_points).astype(np.float32)
            
            # ========== 字段组装: vlm_info (将所有字段组装成最终的数据字典) ==========
            vlm_info = {
                'token': sample['token'],
                'prev': sample['prev'],
                'next': sample['next'],
                'scene_token': sample['scene_token'],
                'image': info['images'][0],
                'images': info['images'],
                'ego_navi_cmd': ego_navi_cmd,
                'meta_action': meta_action_cmd,
                'his_image': his_image if his_image else '',
                'his_images': his_images if his_images else [],
                'his_speed': his_speed.tolist(), 
                'his_acceleration': his_acceleration.tolist(),
                'his_angular_velocity': his_angular_velocity.tolist(),
                'his_theta': his_theta.tolist(),
                'navigation_x': [float(navi_point[1])] * history_frames,
                'navigation_y': [float(navi_point[0])] * history_frames,
                'meta_action_x': [float(meta_action_point[1])] * history_frames,
                'meta_action_y': [float(meta_action_point[0])] * history_frames,
                'future_x': future_x.tolist(),
                'future_y': future_y.tolist()
            }
            
            # ========== 文件保存: 将vlm_info保存为JSON文件 ==========
            # 判断是训练集还是验证集
            is_train = sample['scene_token'] in train_scenes
            split_suffix = 'train' if is_train else 'val'
            
            # 获取场景内的step索引（从0开始计数该场景内的样本）
            # 初始化场景跟踪字典（如果不存在）
            if not hasattr(_fill_trainval_infos, '_scene_sample_indices'):
                _fill_trainval_infos._scene_sample_indices = {}
            
            # 为训练集和验证集使用独立的计数器
            scene_token = sample['scene_token']
            scene_key = f'{scene_token}_{split_suffix}'
            if scene_key not in _fill_trainval_infos._scene_sample_indices:
                _fill_trainval_infos._scene_sample_indices[scene_key] = 0
            
            step_index = _fill_trainval_infos._scene_sample_indices[scene_key]
            _fill_trainval_infos._scene_sample_indices[scene_key] += 1
            
            # 保存文件：scene_name_step{index}_{train|val}.json
            save_path = os.path.join(output_path, f'{scene_name}_step{step_index}_{split_suffix}.json')
            print(f'Save path: {save_path}')
            
            with open(save_path, "w") as f:
                json.dump([vlm_info], f, indent=2)
            
            # 同时添加到train/val列表中以保持兼容性
            if is_train:
                train_nusc_infos.append(vlm_info)
            else:
                val_nusc_infos.append(vlm_info)
            
            count += 1
                


    return train_nusc_infos, val_nusc_infos

def get_global_sensor_pose(rec, nusc, inverse=False):
    # 功能：计算传感器到全局的变换矩阵。
    # 实现：利用四元数和位移，构建变换矩阵。
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    #计算传感器到激光雷达的变换矩阵。
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


###  Senna nuScenes QA tool functions ###


def get_vru_qa(info, vru_dis_thresh=20.0):
    # 功能：生成关于易受伤道路使用者（如行人、自行车等）是否在前方的QA。
    # 实现：遍历目标，判断是否在前方且距离阈值内，生成描述。
    question = f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, " \
               "such as cyclists, motorcyclists, or pedestrians?"
    vru_list = []
    num_objects = info['gt_boxes'].shape[0]
    vru_classes = ['bicycle', 'motorcycle', 'pedestrian']

    for i in range(num_objects):
        obj_loc = info['gt_boxes'][i, :2]
        obj_rel_loc = get_obj_rel_position(obj_loc)
        if obj_rel_loc != "front":
            continue  # only consider VRUs at front

        obj_cls = info['gt_names'][i]
        lat_dis, log_dis = info['gt_boxes'][i, 0], info['gt_boxes'][i, 1]

        if obj_cls in vru_classes and np.linalg.norm(obj_loc) < vru_dis_thresh:
            if lat_dis <= -2.0:
                lat_pos = f" and {int(abs(lat_dis))} meters to the left"
            elif lat_dis >= 2.0:
                lat_pos = f" and {int(abs(lat_dis))} meters to the right"
            else:
                lat_pos = ""
            vru_description = f"a {obj_cls} located {int(abs(log_dis))} meters ahead of me{lat_pos}"
            vru_list.append(vru_description)

    if vru_list:
        answer = "Yes, I see " + ", and ".join(vru_list) + "."
    else:
        answer = "No, I don't see any vulnerable road users ahead of me, " \
                 "such as bicycles, motorcycles, or pedestrians."

    qa = format_qa(question, answer)

    return qa


def get_mot_pred_qa(info,
                    img_type='front',
                    dis_thresh=40.0):
    # 功能：生成关于动态目标未来行为预测的QA（速度决策、路径决策）。
    # 实现：遍历目标，提取位置、速度、未来轨迹，调用辅助函数判断行为，生成问答。

    question = "You are driving, I will now provide you with the location " \
               f"and velocity information of dynamic objects in the {img_type} view image. " \
               "Please predict their future driving behaviors, " \
               "which can be divided into SPEED decisions and PATH decisions. " \
               "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
               "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN." \
               "I will now provide you with the position and velocity information of the dynamic objects: \n"

    num_objects = info['gt_boxes'].shape[0]
    obj_cnt = 0
    answer = ""
    for i in range(num_objects):
        obj_loc = info['gt_boxes'][i, :2]
        obj_rel_loc = get_obj_rel_position(obj_loc)
        if obj_rel_loc != img_type:
            continue  # only care corresbouding view objects
        if np.linalg.norm(obj_loc, axis=-1) >= dis_thresh:
            continue  # only care objects with distance thresholds
        if np.any(info['gt_agent_fut_masks'][i] == 0):
            continue  # 如果object没有未来轨迹信息，则不考虑此object
        lat_dis, log_dis = info['gt_boxes'][i, 0], info['gt_boxes'][i, 1]
        obj_fut_traj = info["gt_agent_fut_trajs_vcs"][i]
        obj_pedal_status = get_obj_acc_or_dec(obj_fut_traj)
        obj_wheel_status = get_obj_turn_or_lane_change(obj_fut_traj)
        obj_speed_plan = pedal_status[obj_pedal_status]
        obj_path_plan = path_status[obj_wheel_status]
        obj_cls = info['gt_names'][i]
        obj_speed = np.linalg.norm(obj_fut_traj[1] - obj_fut_traj[0], axis=-1) / 0.5
        obj_cnt = obj_cnt + 1
        if log_dis >= 0:
            log_describe = f"{int(log_dis)} meters ahead"
        else:
            log_describe = f"{abs(int(log_dis))} meters behind"
        if lat_dis >= 0:
            lat_describe = f"{int(lat_dis)} meters to the right"
        else:
            lat_describe = f"{abs(int(lat_dis))} meters to the left"
        
        obj_info = f'Object {obj_cnt}: {obj_cls}, {log_describe}, {lat_describe}, speed of {int(obj_speed)} m/s.'

        question = question + obj_info + '\n'
        answer = answer + f"Object {obj_cnt}: {obj_speed_plan}, {obj_path_plan}\n"
    
    question_end = "Please predict the future driving behaviors of these objects " \
                   f"based on the {img_type} view image. " \
                   "For example, a well-formatted answer should be like:\n" \
                   "Object 1: KEEP, STRAIGHT\n" \
                   "Object 2: DECELERATE, RIGHT_TURN\n" \
                   "Object 3: ACCELERATE, LEFT_CHANGE\n"

    question = question + question_end
    
    qa = format_qa(question, answer)

    if obj_cnt == 0:
        return None

    return qa



def get_traffic_light_qa(cf_img_path,
                      tokenizer,
                      model,
                      image_processor,use_doubao=True):
    '''
    nuscenes does not contain traffic light labels, so we use pseudo labels generated by vlm
    '''
    # 功能：生成关于交通灯状态的QA（通过VLM模型伪标注）。
    # 实现：构造问题，调用大模型推理图片，返回答案。
    question = "Given the provided forward-facing image from a car's perspective, " \
               "identify if there is a traffic light that affects the car's behavior. " \
               "Respond with 'Red', 'Green', 'Yellow', or 'None'."

    args = type('Args', (), {
        "query": question,
        "img_file": cf_img_path,
    })()

    if not use_doubao:
        answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
        qa = format_qa(question, answer)
        print('llava qa:',qa)
    else:
        answer = eval_doubao_init(args)
        qa = format_qa(question,answer)
        # print('doubao_qa:',qa)

    return qa



def get_img_description_qa(img_path,
                           img_type,
                           tokenizer,
                           model,
                           image_processor,
                           use_doubao=True):

    question = "Suppose you are driving, and I'm providing you with the image " \
               f"captured by the car's {img_type}, generate a description of the driving scene " \
               "which includes the key factors for driving planning, including the positions " \
               "and movements of vehicles and pedestrians; prevailing weather conditions; " \
               "time of day, distinguishing between daylight and nighttime; road conditions, " \
               "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
               "which influence your decision making, specifying whether they are red or green. " \
               "The description should be concise, providing an accurate understanding " \
               "of the driving environment to facilitate informed decision-making."

    args = type('Args', (), {
        "query": question,
        "img_file": img_path,
    })()

    if not use_doubao:
        answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
        qa = format_qa(question, answer)
        # print('llava qa:',qa)
    else:
        answer = eval_doubao_init(args)
        qa = format_qa(question,answer)
        # print('doubao_qa:',qa)

    return qa


def get_plan_qa(info,use_doubao=True):
    
    if np.any(info['gt_ego_fut_masks'] == 0):
        return None

    ego_cur_vel = info['gt_ego_lcf_feat'][7]
    ego_navi_cmd = info['ego_navi_cmd']

    question = f"Your current speed is {int(ego_cur_vel)} m/s, " \
               f"the navigation command is '{ego_navi_cmd}', " \
                "based on the understanding of the driving scene and the navigation information, " \
                "what is your plan for the next three seconds? " \
                "Please answer your SPEED plan and your PATH plan. " \
                "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
                "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
                "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."


    ego_fut_traj = np.cumsum(info['gt_ego_fut_trajs'], axis=-2)
    ego_pedal_status = get_obj_acc_or_dec(ego_fut_traj)
    ego_speed_plan = pedal_status[ego_pedal_status]

    ego_path_plan = get_obj_turn_or_lane_change(ego_fut_traj)
    ego_path_plan = path_status[ego_path_plan]

    answer = ego_speed_plan + ', ' + ego_path_plan + '\n'

    qa = format_qa(question, answer)

    return qa

def get_plan_explaination_qa(info,
                             tokenizer,
                             model,
                             image_processor,use_doubao=True):

    cf_img_name = info['images'][0]

    ego_cur_vel = info['gt_ego_lcf_feat'][7]
    ego_navi_cmd = info['ego_navi_cmd']
    
    ego_fut_traj = np.cumsum(info['gt_ego_fut_trajs'], axis=-2)
    ego_pedal_status = get_obj_acc_or_dec(ego_fut_traj)
    ego_speed_plan = pedal_status[ego_pedal_status]
    ego_path_plan = get_obj_turn_or_lane_change(ego_fut_traj)
    ego_path_plan = path_status[ego_path_plan]

    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    path_decision = {
        'RIGHT_TURN': 'turn right',
        'RIGHT_CHANGE': 'change to the right lane',
        'LEFT_TURN': 'turn left',
        'LEFT_CHANGE': 'change to the left lane',
        'STRAIGHT': 'go straight'
    }

    if ego_speed_plan == 'STOP':
        decision = pedal_decision[ego_speed_plan]
    else:
        decision = pedal_decision[ego_speed_plan] + ' and ' + path_decision[ego_path_plan]


    question = "You are driving, " \
               f"your current speed is {int(ego_cur_vel)} m/s, " \
               f"and the navigation command is '{ego_navi_cmd}', " \
               "your driving decision for the next three seconds is to " \
               f"{decision}. " \
               "Based on the provided image of the driving environment, " \
               "explain the most likely reason for this decision in one or two concise sentence." \

    args = type('Args', (), {
        "query": question,
        "img_file": cf_img_name,
    })()

    if not use_doubao:
        answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
        qa = format_qa(question, answer)
        # print('llava qa:',qa)
    else:
        answer = eval_doubao_init(args)
        qa = format_qa(question,answer)
        # print('doubao_qa:',qa)

    return qa


def format_qa(question, answer):

    image_prompt = "<FRONT VIEW>:\n<image>\n" \
                   "<FRONT LEFT VIEW>:\n<image>\n" \
                   "<FRONT RIGHT VIEW>:\n<image>\n" \
                   "<BACK LEFT VIEW>:\n<image>\n" \
                   "<BACK RIGHT VIEW>:\n<image>\n" \
                   "<BACK VIEW>:\n<image>\n"

    question_dict, answer_dict = {}, {}
    question_dict["from"] = "human"
    question_dict["value"] = image_prompt + question
    answer_dict["from"] = "gpt"
    answer_dict["value"] = answer

    return [question_dict, answer_dict]

# todo
def generate_drive_qa(info, tokenizer, model, image_processor,use_doubao=True):
    import time
    drive_qa_info = []

    vru_qa = get_vru_qa(info)
    drive_qa_info.append(vru_qa)

    img_types = ["front", "front right", "front left", "back", "back left", "back right"]
    dis_thresh_imgs = [40.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    for type, thresh in zip(img_types, dis_thresh_imgs):
        mot_pred_qa = get_mot_pred_qa(info, img_type=type, dis_thresh=thresh)
        if mot_pred_qa is not None:
            drive_qa_info.append(mot_pred_qa)

    traffic_light_qa_starttime = time.time()
    cf_img_path = info['images'][0]
    traffic_light_qa = get_traffic_light_qa(cf_img_path, tokenizer, model, image_processor,use_doubao)
    drive_qa_info.append(traffic_light_qa)
    traffic_light_qa_endtime = time.time()
    # print('2.traffic_light qa infrence time:',traffic_light_qa_endtime - traffic_light_qa_starttime)
  

    cams = [
        'front camera',
        'front right camera',
        'front left camera',
        'back camera',
        'back left camera',
        'back right camera']

    for i, cam in enumerate(cams):
        img_path = info['images'][i]
        description_qa_starttime = time.time()
        description_qa = get_img_description_qa(
            img_path, cam, tokenizer, model, image_processor,use_doubao)
        description_qa_endtime = time.time()
        # print('3.description qa inference time:',description_qa_endtime - description_qa_starttime)
        drive_qa_info.append(description_qa)

    plan_qa = get_plan_qa(info)
    if plan_qa is not None:
        drive_qa_info.append(plan_qa)
    plan_explannation_qa_starttime = time.time()
    plan_explaination_qa = get_plan_explaination_qa(info, tokenizer, model, image_processor)
    plan_explannation_qa_endtime = time.time()
    # print('4.plan explantion qa time:',plan_explannation_qa_endtime - plan_explannation_qa_starttime)
    drive_qa_info.append(plan_explaination_qa)

    return drive_qa_info


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
