import numpy as np
from collections import deque

class PID(object):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window = deque([0 for _ in range(n)], maxlen=n)
        self.error = 0
        self.integral = 0
        self.derivative = 0
    
    def step(self, error):
        self.error = error
        self.window.append(error)
        if len(self.window) >= 2:
            self.integral = np.mean(self.window)
            self.derivative = self.window[-1] - self.window[-2]
        else:
            self.integral = 0.0
            self.derivative = 0.0
        return self.k_p * self.error + self.k_i * self.integral + self.k_d * self.derivative


class PIDController(object):
    def __init__(self, turn_KP=1.25, turn_KI=0.75, turn_KD=0.3, turn_n=20, speed_KP=5.0, speed_KI=0.5, speed_KD=1.0, speed_n=20, max_throttle=1.0, brake_speed=0.1, brake_ratio=1.1, clip_delta=0.25, aim_distance_threshold=5.5, aim_distance_slow=5.5, aim_distance_fast=8.5, clip_throttle=0.75):
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_distance_threshold = aim_distance_threshold
        self.aim_distance_slow = aim_distance_slow
        self.aim_distance_fast = aim_distance_fast
        self.clip_throttle = clip_throttle
        self.turn_controller = PID(k_p=turn_KP, k_i=turn_KI, k_d=turn_KD, n=turn_n)
        self.speed_controller = PID(k_p=speed_KP, k_i=speed_KI, k_d=speed_KD, n=speed_n)
    
    def control_pid(self, waypoints, speed):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        desired_speed = np.linalg.norm(waypoints[7] - waypoints[0]) * 2.0 # 
        brake = ((desired_speed < self.brake_speed) or ((speed / desired_speed) > self.brake_ratio))
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = throttle if not brake else 0.0
        throttle = np.clip(float(throttle), 0, self.clip_throttle)

        # To replicate the slow TransFuser behaviour we have a different distance
        # inside and outside of intersections (detected by desired_speed)
        if desired_speed < self.aim_distance_threshold:
            aim_distance = self.aim_distance_slow
        else:
            aim_distance = self.aim_distance_fast

        # aim_index = 2
        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break
        aim = waypoints[aim_index]
        # angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
        # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

        metadata = {
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            "aim_distance": float(aim_distance),
            'aim_index': int(aim_index),
            'speed': float(speed),
            "desired_speed": float(desired_speed.astype(np.float64)),
            "angle":float(angle),
            "error":float(self.turn_controller.error),
            "integral":float(self.turn_controller.integral),
            "derivative":float(self.turn_controller.derivative),
        }
        return steer, throttle, brake, metadata