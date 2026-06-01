import json
from typing import Optional

import numpy as np
import torch


class Normalize:
    """
    归一化与反归一化工具类

    支持三种模式（通过 norm_mode 控制）：
    1. 'percentile'        ：原始方式，使用分位数做归一化
    2. 'percentile_clip'   ：先按分位数归一化，再 clip 到 [-1, 1]
    3. 'minmax'            ：使用 min-max 归一化到 [-1, 1]

    注意：
    - 默认模式为 'percentile'，保证原有脚本代码行为不变。
    - min-max 模式需要在 statistics.json 中额外提供 *_min / *_max 字段，
      若不存在则回退到分位数模式。
    """

    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, percentile_path, norm_mode: str = "percentile"):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.percentile_path = percentile_path
        self.norm_mode = norm_mode or "percentile"

        with open(self.percentile_path, "r") as f:
            loaded_data = json.load(f)

        # 支持新旧两种 statistics.json 格式
        if "percentile" in loaded_data:
            pct = loaded_data["percentile"]
            mm = loaded_data.get("minmax", {})
        else:
            pct = loaded_data
            mm = {}

        # 分位数参数（原始实现）
        self.speed_01 = pct["speed"][0]
        self.speed_99 = pct["speed"][1]
        self.acceleration_01 = pct["acceleration"][0]
        self.acceleration_99 = pct["acceleration"][1]
        self.angular_velocity_01 = pct["angular_velocity"][0]
        self.angular_velocity_99 = pct["angular_velocity"][1]
        self.theta_01 = pct["theta"][0]
        self.theta_99 = pct["theta"][1]
        self.fur_x_01 = pct["fur_x"][0]
        self.fur_x_99 = pct["fur_x"][1]
        self.fur_y_01 = pct["fur_y"][0]
        self.fur_y_99 = pct["fur_y"][1]

        # 历史轨迹参数（可选）
        his_x = pct.get("his_x")
        his_y = pct.get("his_y")
        self.his_x_01 = his_x[0] if his_x else -abs(self.fur_x_99)
        self.his_x_99 = his_x[1] if his_x else 0.0
        self.his_y_01 = his_y[0] if his_y else self.fur_y_01
        self.his_y_99 = his_y[1] if his_y else self.fur_y_99

        # navigation_id 参数（可选）
        nav = pct.get("navigation_id")
        self.nav_id_01 = nav[0] if nav else 0.0
        self.nav_id_99 = nav[1] if nav else 2.0

        # command_far 参数（兼容旧格式）
        if "command_far_x" in pct:
            self.target_far_x_01 = pct["command_far_x"][0]
            self.target_far_x_99 = pct["command_far_x"][1]
            self.target_far_y_01 = pct["command_far_y"][0]
            self.target_far_y_99 = pct["command_far_y"][1]
        else:
            self.target_far_x_01 = 0.0
            self.target_far_x_99 = 1.0
            self.target_far_y_01 = 0.0
            self.target_far_y_99 = 1.0

        # min-max 参数（可选）：支持新旧两种格式
        def _get_mm(field, idx, fallback_key, fallback_val):
            v = mm.get(field)
            if v is not None and len(v) > idx:
                return v[idx]
            v2 = loaded_data.get(fallback_key)
            return v2 if v2 is not None else fallback_val

        self.speed_min = _get_mm("speed", 0, "speed_min", self.speed_01)
        self.speed_max = _get_mm("speed", 1, "speed_max", self.speed_99)
        self.acceleration_min = _get_mm("acceleration", 0, "acceleration_min", self.acceleration_01)
        self.acceleration_max = _get_mm("acceleration", 1, "acceleration_max", self.acceleration_99)
        self.angular_velocity_min = _get_mm("angular_velocity", 0, "angular_velocity_min", self.angular_velocity_01)
        self.angular_velocity_max = _get_mm("angular_velocity", 1, "angular_velocity_max", self.angular_velocity_99)
        self.theta_min = _get_mm("theta", 0, "theta_min", self.theta_01)
        self.theta_max = _get_mm("theta", 1, "theta_max", self.theta_99)
        self.fur_x_min = _get_mm("fur_x", 0, "fur_x_min", self.fur_x_01)
        self.fur_x_max = _get_mm("fur_x", 1, "fur_x_max", self.fur_x_99)
        self.fur_y_min = _get_mm("fur_y", 0, "fur_y_min", self.fur_y_01)
        self.fur_y_max = _get_mm("fur_y", 1, "fur_y_max", self.fur_y_99)
        self.his_x_min = _get_mm("his_x", 0, "his_x_min", self.his_x_01)
        self.his_x_max = _get_mm("his_x", 1, "his_x_max", self.his_x_99)
        self.his_y_min = _get_mm("his_y", 0, "his_y_min", self.his_y_01)
        self.his_y_max = _get_mm("his_y", 1, "his_y_max", self.his_y_99)
        self.nav_id_min = _get_mm("navigation_id", 0, "navigation_id_min", self.nav_id_01)
        self.nav_id_max = _get_mm("navigation_id", 1, "navigation_id_max", self.nav_id_99)
        self.target_far_x_min = _get_mm("command_far_x", 0, "command_far_x_min", self.target_far_x_01)
        self.target_far_x_max = _get_mm("command_far_x", 1, "command_far_x_max", self.target_far_x_99)
        self.target_far_y_min = _get_mm("command_far_y", 0, "command_far_y_min", self.target_far_y_01)
        self.target_far_y_max = _get_mm("command_far_y", 1, "command_far_y_max", self.target_far_y_99)

        self._initialized = True

    # -------- 基础函数：原始分位数归一化 / 反归一化 --------
    def standard_normalize(self, true_np_array, low, high):
        return 2 * (true_np_array - low) / (high - low) - 1

    def recover_trajectory(self, normalized_np_array, low, high):
        return (normalized_np_array + 1) * (high - low) / 2 + low

    # -------- 内部工具：按模式选择归一化 / 反归一化 --------
    def _normalize_by_mode(self, x, p01, p99, vmin, vmax):
        if self.norm_mode == "percentile":
            return self.standard_normalize(x, p01, p99)
        if self.norm_mode == "percentile_clip":
            y = self.standard_normalize(x, p01, p99)
            return np.clip(y, -1.0, 1.0)
        if self.norm_mode == "minmax":
            if np.allclose(vmin, vmax):
                return self.standard_normalize(x, p01, p99)
            y = 2 * (x - vmin) / (vmax - vmin) - 1
            return np.clip(y, -1.0, 1.0)
        return self.standard_normalize(x, p01, p99)

    def _denormalize_by_mode(self, x_norm, p01, p99, vmin, vmax):
        if self.norm_mode in ("percentile", "percentile_clip"):
            return self.recover_trajectory(x_norm, p01, p99)
        if self.norm_mode == "minmax":
            if np.allclose(vmin, vmax):
                return self.recover_trajectory(x_norm, p01, p99)
            return self.recover_trajectory(x_norm, vmin, vmax)
        return self.recover_trajectory(x_norm, p01, p99)

    NUM_NAV_CLASSES = 5

    def _nav_id_to_onehot(self, nav_id_array):
        ids = np.round(nav_id_array).astype(int)
        ids = np.clip(ids, 0, self.NUM_NAV_CLASSES - 1)
        onehot = np.zeros((len(ids), self.NUM_NAV_CLASSES), dtype=np.float32)
        onehot[np.arange(len(ids)), ids] = 1.0
        return onehot

    @staticmethod
    def _ensure_2d(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x[:, np.newaxis]
        return x

    def _prepare_history_xy(self, all_data, target_len: int):
        origin_x_norm = float(self._normalize_by_mode(
            np.array(0.0), self.his_x_01, self.his_x_99, self.his_x_min, self.his_x_max
        ))
        origin_y_norm = float(self._normalize_by_mode(
            np.array(0.0), self.his_y_01, self.his_y_99, self.his_y_min, self.his_y_max
        ))

        if "his_x" in all_data and "his_y" in all_data:
            his_x_raw = np.asarray(all_data["his_x"], dtype=np.float32).reshape(-1)
            his_y_raw = np.asarray(all_data["his_y"], dtype=np.float32).reshape(-1)
            hist_len = min(his_x_raw.shape[0], his_y_raw.shape[0])
            his_x = self._normalize_by_mode(
                his_x_raw[:hist_len], self.his_x_01, self.his_x_99, self.his_x_min, self.his_x_max
            )
            his_y = self._normalize_by_mode(
                his_y_raw[:hist_len], self.his_y_01, self.his_y_99, self.his_y_min, self.his_y_max
            )
            his_traj = np.stack([his_x, his_y], axis=1)
        elif "his_trajectory" in all_data:
            his_traj_raw = np.asarray(all_data["his_trajectory"], dtype=np.float32)
            # his_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
            his_y = self._normalize_by_mode(
                his_traj_raw[:, 0], self.his_y_01, self.his_y_99, self.his_y_min, self.his_y_max
            )
            his_x = self._normalize_by_mode(
                his_traj_raw[:, 1], self.his_x_01, self.his_x_99, self.his_x_min, self.his_x_max
            )
            his_traj = np.stack([his_x, his_y], axis=1)
        else:
            return np.full((target_len, 2), [origin_x_norm, origin_y_norm], dtype=np.float32)

        if his_traj.shape[0] < target_len:
            pad_len = target_len - his_traj.shape[0]
            pad = np.full((pad_len, 2), [origin_x_norm, origin_y_norm], dtype=np.float32)
            his_traj = np.concatenate([his_traj, pad], axis=0)
        elif his_traj.shape[0] > target_len:
            his_traj = his_traj[-target_len:]
        return his_traj.astype(np.float32)

    # -------- 对外接口：state / traj / infer_traj --------
    def prepare_state(
        self,
        all_data,
        nav_onehot: bool = False,
        use_his_trajectory: bool = False,
        state_frame_index: Optional[int] = None,
    ):
        """根据当前模式对状态进行归一化，支持 nav_onehot / use_his_trajectory / state_frame_index"""
        his_speed_raw = np.asarray(all_data["his_speed"])

        if his_speed_raw.ndim == 1 and "x_command_far" in all_data and "y_command_far" in all_data:
            # 旧 Bench2Drive/DriveMoE 格式：
            # speed(1) + acceleration(3) + angular_velocity(3) + theta(1)
            # + far command x/y(2) = state_dim 10。
            his_speed = self._ensure_2d(self._normalize_by_mode(
                his_speed_raw, self.speed_01, self.speed_99, self.speed_min, self.speed_max
            ))
            his_acceleration = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_acceleration"]),
                self.acceleration_01,
                self.acceleration_99,
                self.acceleration_min,
                self.acceleration_max,
            ))
            his_angular_velocity = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_angular_velocity"]),
                self.angular_velocity_01,
                self.angular_velocity_99,
                self.angular_velocity_min,
                self.angular_velocity_max,
            ))
            his_theta = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_theta"]), self.theta_01, self.theta_99, self.theta_min, self.theta_max
            ))
            target_far_x = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["x_command_far"]),
                self.target_far_x_01,
                self.target_far_x_99,
                self.target_far_x_min,
                self.target_far_x_max,
            ))
            target_far_y = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["y_command_far"]),
                self.target_far_y_01,
                self.target_far_y_99,
                self.target_far_y_min,
                self.target_far_y_max,
            ))
            parts = [
                his_speed,
                his_acceleration,
                his_angular_velocity,
                his_theta,
                target_far_x,
                target_far_y,
            ]
            if use_his_trajectory:
                parts.append(self._prepare_history_xy(all_data, target_len=his_theta.shape[0]))
        else:
            # MOGO/Naviv2 格式：speed/acceleration 取平面 2 维，yaw rate 取 z 轴。
            his_speed = self._ensure_2d(self._normalize_by_mode(
                his_speed_raw[:, :2], self.speed_01, self.speed_99, self.speed_min, self.speed_max
            ))
            his_acceleration = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_acceleration"])[:, :2],
                self.acceleration_01,
                self.acceleration_99,
                self.acceleration_min,
                self.acceleration_max,
            ))
            his_angular_velocity = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_angular_velocity"])[:, 2:3],
                self.angular_velocity_01,
                self.angular_velocity_99,
                self.angular_velocity_min,
                self.angular_velocity_max,
            ))
            his_theta = self._ensure_2d(self._normalize_by_mode(
                np.asarray(all_data["his_theta"]), self.theta_01, self.theta_99, self.theta_min, self.theta_max
            ))

            parts = [his_speed, his_acceleration, his_angular_velocity, his_theta]

            # 历史轨迹（可选）
            if use_his_trajectory:
                parts.append(self._prepare_history_xy(all_data, target_len=his_theta.shape[0]))

            # navigation_id（可选）
            if nav_onehot and "navigation_id" in all_data:
                parts.append(self._nav_id_to_onehot(np.asarray(all_data["navigation_id"])))
            elif "navigation_id" in all_data:
                navigation_id = self._normalize_by_mode(
                    np.asarray(all_data["navigation_id"]),
                    self.nav_id_01,
                    self.nav_id_99,
                    self.nav_id_min,
                    self.nav_id_max,
                )
                parts.append(self._ensure_2d(navigation_id))

        state_np = np.concatenate(parts, axis=1)

        # state_frame_index: 只取单帧
        if state_frame_index is not None:
            frame_idx = int(state_frame_index)
            if frame_idx < 0:
                frame_idx = state_np.shape[0] + frame_idx
            if frame_idx < 0 or frame_idx >= state_np.shape[0]:
                raise IndexError(
                    f"state_frame_index={state_frame_index} out of range for state length {state_np.shape[0]}"
                )
            state_np = state_np[frame_idx : frame_idx + 1]

        state = torch.tensor(state_np, dtype=torch.float32)
        return state

    def prepare_traj(self, all_data):
        """根据当前模式对 fur_trajectory / fur_x / fur_y 进行归一化"""
        if "fur_trajectory" in all_data:
            fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)
            # fur_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
            fur_y = self._normalize_by_mode(fur[:, 0], self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max)
            fur_x = self._normalize_by_mode(fur[:, 1], self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max)
            trajectory = torch.tensor(np.stack([fur_x, fur_y], axis=1), dtype=torch.float32)
            return trajectory

        fur_x = self._normalize_by_mode(
            np.asarray(all_data["fur_x"]), self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max
        )
        fur_y = self._normalize_by_mode(
            np.asarray(all_data["fur_y"]), self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max
        )
        trajectory = torch.tensor(
            np.concatenate((fur_x[:, :, np.newaxis], fur_y[:, :, np.newaxis]), axis=2), dtype=torch.float32
        )
        return trajectory

    def infer_traj(self, result):
        """
        将模型输出（[-1, 1] 空间）反归一化到真实坐标系
        result: [1, horizon, 2]
        """
        x_norm = result[0, :, 0]
        y_norm = result[0, :, 1]

        pred_x = self._denormalize_by_mode(
            x_norm, self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max
        )
        pred_y = self._denormalize_by_mode(
            y_norm, self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max
        )
        return pred_x, pred_y
