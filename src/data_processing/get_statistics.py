import numpy as np
from tqdm import *
import json
import pickle
import os
import argparse

def calculate(data):
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99 

def calc_minmax(data):
    return np.min(data), np.max(data)

def fmt(x):
    """转成 Python float，保证 JSON 可序列化（不做截断/四舍五入）"""
    return float(x)

def flatten(data):
    return np.concatenate([arr.flatten() for arr in data])

def generate_data(data_path, output_path='statistics.json'):
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    fur_x = []
    fur_y = []
    # command_near_x = []
    # command_near_y = []
    command_far_x = []
    command_far_y = []
    navigation_id = []
    print('calcuating percentile_01 and percentile_99 ... ')
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            his_theta.append(all_data['his_theta'])
            his_angular_velocity.append(all_data['his_angular_velocity'])
            his_acceleration.append(all_data['his_acceleration'])
            his_speed.append(all_data['his_speed'])
            fur_x.append(all_data['fur_x'])
            fur_y.append(all_data['fur_y'])
            command_far_x.append(all_data['x_command_far'])
            command_far_y.append(all_data['y_command_far'])
            if 'navigation_id' in all_data:
                navigation_id.append(all_data['navigation_id'])
            # command_near_x.append(all_data['x_command_near'])
            # command_near_y.append(all_data['y_command_near'])
    # 打印 flatten 之前的维度信息（数据保证非空）
    print('\n=== Flatten之前的维度信息 ===')
    print(f'fur_x: list长度={len(fur_x)}, 第一个元素shape={fur_x[0].shape}, dtype={type(fur_x[0])}')
    print(f'fur_y: list长度={len(fur_y)}, 第一个元素shape={fur_y[0].shape}, dtype={type(fur_y[0])}')
    # print(f'command_near_x: list长度={len(command_near_x)}, 第一个元素shape={command_near_x[0].shape}, dtype={type(command_near_x[0])}')
    # print(f'command_near_y: list长度={len(command_near_y)}, 第一个元素shape={command_near_y[0].shape}, dtype={type(command_near_y[0])}')
    print(f'command_far_x: list长度={len(command_far_x)}, 第一个元素shape={command_far_x[0].shape}, dtype={type(command_far_x[0])}')
    print(f'command_far_y: list长度={len(command_far_y)}, 第一个元素shape={command_far_y[0].shape}, dtype={type(command_far_y[0])}')
    print('=' * 80 + '\n')
    
    his_speed = flatten(his_speed)
    his_acceleration = flatten(his_acceleration)
    his_angular_velocity = flatten(his_angular_velocity)
    his_theta = flatten(his_theta)
    fur_x = flatten(fur_x)
    fur_y = flatten(fur_y)
    command_far_x = flatten(command_far_x)
    command_far_y = flatten(command_far_y)
    if navigation_id:
        navigation_id = flatten(navigation_id)
    # command_near_x = flatten(command_near_x)
    # command_near_y = flatten(command_near_y)
    # 打印 flatten 之后的维度信息（只关注四个字段）
    print('=== Flatten之后的维度信息 ===')
    print(f'fur_x: shape={fur_x.shape}, dtype={fur_x.dtype}')
    print(f'fur_y: shape={fur_y.shape}, dtype={fur_y.dtype}')
    # print(f'command_near_x: shape={command_near_x.shape}, dtype={command_near_x.dtype}')
    # print(f'command_near_y: shape={command_near_y.shape}, dtype={command_near_y.dtype}')
    print(f'command_far_x: shape={command_far_x.shape}, dtype={command_far_x.dtype}')
    print(f'command_far_y: shape={command_far_y.shape}, dtype={command_far_y.dtype}')
    print('=' * 80 + '\n')
    
    speed_01, speed_99 = calculate(his_speed)
    acceleration_01, acceleration_99 = calculate(his_acceleration)
    angular_velocity_01, angular_velocity_99 = calculate(his_angular_velocity)
    theta_01, theta_99 = calculate(his_theta)
    fur_x_01, fur_x_99 = calculate(fur_x)
    fur_y_01, fur_y_99 = calculate(fur_y)
    command_far_x_01, command_far_x_99 = calculate(command_far_x)
    command_far_y_01, command_far_y_99 = calculate(command_far_y)
    has_nav = isinstance(navigation_id, np.ndarray) and len(navigation_id) > 0
    if has_nav:
        navigation_id_01, navigation_id_99 = calculate(navigation_id)
    # command_near_x_01, command_near_x_99 = calculate(command_near_x)
    # command_near_y_01, command_near_y_99 = calculate(command_near_y)

    # min-max
    speed_min, speed_max = calc_minmax(his_speed)
    acceleration_min, acceleration_max = calc_minmax(his_acceleration)
    angular_velocity_min, angular_velocity_max = calc_minmax(his_angular_velocity)
    theta_min, theta_max = calc_minmax(his_theta)
    fur_x_min, fur_x_max = calc_minmax(fur_x)
    fur_y_min, fur_y_max = calc_minmax(fur_y)
    command_far_x_min, command_far_x_max = calc_minmax(command_far_x)
    command_far_y_min, command_far_y_max = calc_minmax(command_far_y)
    if has_nav:
        navigation_id_min, navigation_id_max = calc_minmax(navigation_id)
    # command_near_x_min, command_near_x_max = calc_minmax(command_near_x)
    # command_near_y_min, command_near_y_max = calc_minmax(command_near_y)

    print("=== Percentile (p01, p99) ===")
    print("speed:", speed_01, speed_99)
    print("acceleration", acceleration_01, acceleration_99)
    print("angular_velocity", angular_velocity_01, angular_velocity_99)
    print("theta:", theta_01, theta_99)
    print("fur_x:", fur_x_01, fur_x_99)
    print("fur_y:", fur_y_01, fur_y_99)
    print("command_far_x:", command_far_x_01, command_far_x_99)
    print("command_far_y:", command_far_y_01, command_far_y_99)
    if has_nav:
        print("navigation_id:", navigation_id_01, navigation_id_99)
    # print("command_near_x:", command_near_x_01, command_near_x_99)
    # print("command_near_y:", command_near_y_01, command_near_y_99)

    print("\n=== MinMax (min, max) ===")
    print("speed:", speed_min, speed_max)
    print("acceleration", acceleration_min, acceleration_max)
    print("angular_velocity", angular_velocity_min, angular_velocity_max)
    print("theta:", theta_min, theta_max)
    print("fur_x:", fur_x_min, fur_x_max)
    print("fur_y:", fur_y_min, fur_y_max)
    print("command_far_x:", command_far_x_min, command_far_x_max)
    print("command_far_y:", command_far_y_min, command_far_y_max)
    if has_nav:
        print("navigation_id:", navigation_id_min, navigation_id_max)
    # print("command_near_x:", command_near_x_min, command_near_x_max)
    # print("command_near_y:", command_near_y_min, command_near_y_max)
    
    data = {
        "percentile": {
            "speed": [fmt(speed_01), fmt(speed_99)],
            "acceleration": [fmt(acceleration_01), fmt(acceleration_99)],
            "angular_velocity": [fmt(angular_velocity_01), fmt(angular_velocity_99)],
            "theta": [fmt(theta_01), fmt(theta_99)],
            "fur_x": [fmt(fur_x_01), fmt(fur_x_99)],
            "fur_y": [fmt(fur_y_01), fmt(fur_y_99)],
            # "command_near_x": [fmt(command_near_x_01), fmt(command_near_x_99)],
            # "command_near_y": [fmt(command_near_y_01), fmt(command_near_y_99)],
            "command_far_x": [fmt(command_far_x_01), fmt(command_far_x_99)],
            "command_far_y": [fmt(command_far_y_01), fmt(command_far_y_99)],
            **({"navigation_id": [fmt(navigation_id_01), fmt(navigation_id_99)]} if has_nav else {}),
        },
        "minmax": {
            "speed": [fmt(speed_min), fmt(speed_max)],
            "acceleration": [fmt(acceleration_min), fmt(acceleration_max)],
            "angular_velocity": [fmt(angular_velocity_min), fmt(angular_velocity_max)],
            "theta": [fmt(theta_min), fmt(theta_max)],
            "fur_x": [fmt(fur_x_min), fmt(fur_x_max)],
            "fur_y": [fmt(fur_y_min), fmt(fur_y_max)],
            # "command_near_x": [fmt(command_near_x_min), fmt(command_near_x_max)],
            # "command_near_y": [fmt(command_near_y_min), fmt(command_near_y_max)],
            "command_far_x": [fmt(command_far_x_min), fmt(command_far_x_max)],
            "command_far_y": [fmt(command_far_y_min), fmt(command_far_y_max)],
            **({"navigation_id": [fmt(navigation_id_min), fmt(navigation_id_max)]} if has_nav else {}),
        },
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Statistics saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='statistics.json', 
                        help='Output path for statistics JSON file (default: statistics.json)')
    args = parser.parse_args()
    generate_data(args.data_path, args.output_path)

# 5-10
# speed: -0.021534712985157967 11.078238487243652
# acceleration -7.847959995269775 10.291497230529785
# angular_velocity -0.5025675892829895 0.4206865522265426
# theta: -0.18638522624969484 0.22664238452911345
# his_x: -3.7781866717338564 0.00403969112318009
# his_y: -0.11440461911261082 0.10973326697945517
# fur_x: -0.00909316767938435 8.093738679885869
# fur_y: -1.1331280183792114 0.9688449388742447