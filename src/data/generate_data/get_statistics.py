"""
Generates statistical percentiles (1st and 99th) for Bench2Drive dataset features.

This script processes pickle files containing vehicle trajectory data and calculates:
- Speed percentiles
- Acceleration percentiles  
- Angular velocity percentiles
- Heading angle (theta) percentiles
- Future/predicted position (fur_x, fur_y) percentiles
- Commanded future position (command_far_x/y) percentiles

The results are saved as a JSON file ('b2d_statistics.json') containing the 1st and 99th 
percentile values for each feature, which can be used for data normalization or analysis.

Usage:
    python get_statistics.py --data_path /path/to/pkl_files
"""
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def calculate(data):
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99 

def flatten(data):
    flattened = np.concatenate([arr.flatten() for arr in data])
    return flattened

def generate_data(data_path):
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    fur_x = []
    fur_y = []
    command_far_x = []
    command_far_y = []
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
        'speed': [speed_01, speed_99],
        'acceleration': [acceleration_01, acceleration_99],
        'angular_velocity': [angular_velocity_01, angular_velocity_99],
        'theta': [theta_01, theta_99],
        'fur_x': [fur_x_01, fur_x_99],
        'fur_y': [fur_y_01, fur_y_99],
        'command_far_x': [command_far_x_01, command_far_x_99],
        'command_far_y': [command_far_y_01, command_far_y_99],
    }

    with open('b2d_statistics.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    generate_data(args.data_path)