import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm


def calculate(data):
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99


def calc_minmax(data):
    return np.min(data), np.max(data)


def fmt(x):
    return float(x)


def flatten(data):
    return np.concatenate([arr.flatten() for arr in data])


def split_future_xy(all_data):
    if "fur_trajectory" in all_data:
        fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)
        # fur_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
        return fur[:, 1], fur[:, 0]
    return np.asarray(all_data["fur_x"]), np.asarray(all_data["fur_y"])


def split_history_xy(all_data):
    his = all_data.get("his_trajectory")
    if his is None:
        return None, None
    his = np.asarray(his, dtype=np.float32)
    if his.ndim != 2 or his.shape[-1] != 2:
        raise ValueError(f"Unexpected his_trajectory shape: {his.shape}")
    # his_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
    return his[:, 1], his[:, 0]



NAVIGATION_COMMAND_TO_ID = {
    "straight": 0,
    "turn left": 1,
    "turn right": 2,
    "u turn": 3,
    "park": 4,
    "none": 0,
}


def infer_navigation_id(all_data):
    nav = all_data.get("navigation_id")
    if nav is not None:
        nav = np.asarray(nav)
        if nav.size > 0:
            return int(nav.reshape(-1)[0])

    cmd = all_data.get("navigation_command")
    if cmd is None:
        return 0
    cmd = np.asarray(cmd)
    if cmd.size == 0:
        return 0
    token = str(cmd.reshape(-1, cmd.shape[-1])[0][0]).strip().lower() if cmd.ndim >= 2 else str(cmd.reshape(-1)[0]).strip().lower()
    return NAVIGATION_COMMAND_TO_ID.get(token, 0)

def generate_data(data_path, output_path='statistics.json'):
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    his_x = []
    his_y = []
    fur_x = []
    fur_y = []
    navigation_id = []

    print('calcuating percentile_01 and percentile_99 ... ')
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            his_theta.append(np.asarray(all_data['his_theta']))
            his_angular_velocity.append(np.asarray(all_data['his_angular_velocity']))
            his_acceleration.append(np.asarray(all_data['his_acceleration']))
            his_speed.append(np.asarray(all_data['his_speed']))

            hx, hy = split_history_xy(all_data)
            if hx is not None:
                his_x.append(hx)
                his_y.append(hy)

            fx, fy = split_future_xy(all_data)
            fur_x.append(fx)
            fur_y.append(fy)

            nav_id = infer_navigation_id(all_data)
            navigation_id.append(np.asarray([nav_id], dtype=np.float32))

    his_speed = flatten(his_speed)
    his_acceleration = flatten(his_acceleration)
    his_angular_velocity = flatten(his_angular_velocity)
    his_theta = flatten(his_theta)
    fur_x = flatten(fur_x)
    fur_y = flatten(fur_y)
    has_his_traj = len(his_x) > 0
    if has_his_traj:
        his_x = flatten(his_x)
        his_y = flatten(his_y)
    has_nav = True
    if has_nav:
        navigation_id = flatten(navigation_id)

    speed_01, speed_99 = calculate(his_speed)
    acceleration_01, acceleration_99 = calculate(his_acceleration)
    angular_velocity_01, angular_velocity_99 = calculate(his_angular_velocity)
    theta_01, theta_99 = calculate(his_theta)
    fur_x_01, fur_x_99 = calculate(fur_x)
    fur_y_01, fur_y_99 = calculate(fur_y)
    if has_his_traj:
        his_x_01, his_x_99 = calculate(his_x)
        his_y_01, his_y_99 = calculate(his_y)
    if has_nav:
        navigation_id_01, navigation_id_99 = calculate(navigation_id)

    speed_min, speed_max = calc_minmax(his_speed)
    acceleration_min, acceleration_max = calc_minmax(his_acceleration)
    angular_velocity_min, angular_velocity_max = calc_minmax(his_angular_velocity)
    theta_min, theta_max = calc_minmax(his_theta)
    fur_x_min, fur_x_max = calc_minmax(fur_x)
    fur_y_min, fur_y_max = calc_minmax(fur_y)
    if has_his_traj:
        his_x_min, his_x_max = calc_minmax(his_x)
        his_y_min, his_y_max = calc_minmax(his_y)
    if has_nav:
        navigation_id_min, navigation_id_max = calc_minmax(navigation_id)

    print("=== Percentile (p01, p99) ===")
    print("speed:", speed_01, speed_99)
    print("acceleration", acceleration_01, acceleration_99)
    print("angular_velocity", angular_velocity_01, angular_velocity_99)
    print("theta:", theta_01, theta_99)
    if has_his_traj:
        print("his_x:", his_x_01, his_x_99)
        print("his_y:", his_y_01, his_y_99)
    print("fur_x:", fur_x_01, fur_x_99)
    print("fur_y:", fur_y_01, fur_y_99)
    if has_nav:
        print("navigation_id:", navigation_id_01, navigation_id_99)

    print("\n=== MinMax (min, max) ===")
    print("speed:", speed_min, speed_max)
    print("acceleration", acceleration_min, acceleration_max)
    print("angular_velocity", angular_velocity_min, angular_velocity_max)
    print("theta:", theta_min, theta_max)
    if has_his_traj:
        print("his_x:", his_x_min, his_x_max)
        print("his_y:", his_y_min, his_y_max)
    print("fur_x:", fur_x_min, fur_x_max)
    print("fur_y:", fur_y_min, fur_y_max)
    if has_nav:
        print("navigation_id:", navigation_id_min, navigation_id_max)

    data = {
        "percentile": {
            "speed": [fmt(speed_01), fmt(speed_99)],
            "acceleration": [fmt(acceleration_01), fmt(acceleration_99)],
            "angular_velocity": [fmt(angular_velocity_01), fmt(angular_velocity_99)],
            "theta": [fmt(theta_01), fmt(theta_99)],
            "fur_x": [fmt(fur_x_01), fmt(fur_x_99)],
            "fur_y": [fmt(fur_y_01), fmt(fur_y_99)],
            **({"his_x": [fmt(his_x_01), fmt(his_x_99)], "his_y": [fmt(his_y_01), fmt(his_y_99)]} if has_his_traj else {}),
            **({"navigation_id": [fmt(navigation_id_01), fmt(navigation_id_99)]} if has_nav else {}),
        },
        "minmax": {
            "speed": [fmt(speed_min), fmt(speed_max)],
            "acceleration": [fmt(acceleration_min), fmt(acceleration_max)],
            "angular_velocity": [fmt(angular_velocity_min), fmt(angular_velocity_max)],
            "theta": [fmt(theta_min), fmt(theta_max)],
            "fur_x": [fmt(fur_x_min), fmt(fur_x_max)],
            "fur_y": [fmt(fur_y_min), fmt(fur_y_max)],
            **({"his_x": [fmt(his_x_min), fmt(his_x_max)], "his_y": [fmt(his_y_min), fmt(his_y_max)]} if has_his_traj else {}),
            **({"navigation_id": [fmt(navigation_id_min), fmt(navigation_id_max)]} if has_nav else {}),
        },
    }

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Statistics saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='statistics.json', help='Output path for statistics JSON file (default: statistics.json)')
    args = parser.parse_args()
    generate_data(args.data_path, args.output_path)
