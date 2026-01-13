"""
# 获取整个2O-bags中的归一化数据
python src/data_processing/get_statistics_mogo_all.py \
    --data_path /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --output_path /D2/NXT/vlm/miaobinqi/Senna/mogo_test/statistics.json
"""

import numpy as np
from tqdm import *
import json
import pickle
import os
import argparse

def calculate(data):
    """
    使用 1% / 99% 百分位作为归一化范围（与原始逻辑保持一致）
    """
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99 

def flatten(data):
    return np.concatenate([arr.flatten() for arr in data])

def generate_data(data_path):
    """
    生成统计量：
    - 如果 data_path 直接包含 pkl 文件，则只使用该目录下的 pkl
    - 否则，将 data_path 视为根目录（如 2O-bags），遍历其每个子目录下的 drivepi 目录中的 pkl 文件
    """
    pkl_paths = []
    # 1) 先看当前目录是否直接有 pkl 文件（兼容旧用法：直接传 drivepi 目录）
    if os.path.isdir(data_path):
        direct_pkls = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith('.pkl')
        ]
        if direct_pkls:
            pkl_paths.extend(direct_pkls)
        else:
            # 2) 否则，将 data_path 视为根目录，遍历每个子目录下的 drivepi
            #    例如：/D2/.../2O-bags/<timestamp>/drivepi/*.pkl
            for sub in sorted(os.listdir(data_path)):
                sub_path = os.path.join(data_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                drivepi_dir = os.path.join(sub_path, "drivepi")
                if not os.path.isdir(drivepi_dir):
                    continue
                for f in os.listdir(drivepi_dir):
                    if f.endswith(".pkl"):
                        pkl_paths.append(os.path.join(drivepi_dir, f))
    else:
        raise ValueError(f"data_path is not a directory: {data_path}")

    if not pkl_paths:
        raise ValueError(f"No pkl files found under data_path: {data_path}")

    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    fur_x = []
    fur_y = []
    command_far_x = []
    command_far_y = []
    print('calcuating percentile_01 and percentile_99 ... ')
    for pkl_data_path in tqdm(pkl_paths):
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
    his_speed = flatten(his_speed)
    his_acceleration = flatten(his_acceleration)
    his_angular_velocity = flatten(his_angular_velocity)
    his_theta = flatten(his_theta)
    fur_x = flatten(fur_x)
    fur_y = flatten(fur_y)
    command_far_x = flatten(command_far_x)
    command_far_y = flatten(command_far_y)
    
    speed_01, speed_99 = calculate(his_speed)
    acceleration_01, acceleration_99 = calculate(his_acceleration)
    angular_velocity_01, angular_velocity_99 = calculate(his_angular_velocity)
    theta_01, theta_99 = calculate(his_theta)
    fur_x_01, fur_x_99 = calculate(fur_x)
    fur_y_01, fur_y_99 = calculate(fur_y)
    command_far_x_01, command_far_x_99 = calculate(command_far_x)
    command_far_y_01, command_far_y_99 = calculate(command_far_y)

    print('speed:', speed_01, speed_99)
    print('acceleration', acceleration_01, acceleration_99)
    print('angular_velocity', angular_velocity_01, angular_velocity_99)
    print('theta:', theta_01, theta_99)
    print('fur_x:', fur_x_01, fur_x_99)
    print('fur_y:', fur_y_01, fur_y_99)
    print('command_far_x:', command_far_x_01, command_far_x_99)
    print('command_far_y:', command_far_y_01, command_far_y_99)
    
    data = {
        'speed': [float(speed_01), float(speed_99)],
        'acceleration': [float(acceleration_01), float(acceleration_99)],
        'angular_velocity': [float(angular_velocity_01), float(angular_velocity_99)],
        'theta': [float(theta_01), float(theta_99)],
        'fur_x': [float(fur_x_01), float(fur_x_99)],
        'fur_y': [float(fur_y_01), float(fur_y_99)],
        'command_far_x': [float(command_far_x_01), float(command_far_x_99)],
        'command_far_y': [float(command_far_y_01), float(command_far_y_99)],
    }

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="statistics.json",
        help="Path to save computed statistics (default: ./statistics.json)",
    )
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = generate_data(data_path)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved statistics to: {output_path}")

# 5-10
# speed: -0.021534712985157967 11.078238487243652
# acceleration -7.847959995269775 10.291497230529785
# angular_velocity -0.5025675892829895 0.4206865522265426
# theta: -0.18638522624969484 0.22664238452911345
# his_x: -3.7781866717338564 0.00403969112318009
# his_y: -0.11440461911261082 0.10973326697945517
# fur_x: -0.00909316767938435 8.093738679885869
# fur_y: -1.1331280183792114 0.9688449388742447