import json

import numpy as np
import torch


class Normalize:
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

        if "percentile" in loaded_data:
            pct = loaded_data["percentile"]
            mm = loaded_data.get("minmax", {})
        else:
            pct = loaded_data
            mm = {}

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
        his_x = pct.get("his_x")
        his_y = pct.get("his_y")
        self.his_x_01 = his_x[0] if his_x else 0.0
        self.his_x_99 = his_x[1] if his_x else 1.0
        self.his_y_01 = his_y[0] if his_y else 0.0
        self.his_y_99 = his_y[1] if his_y else 1.0
        nav = pct.get("navigation_id")
        self.nav_id_01 = nav[0] if nav else 0.0
        self.nav_id_99 = nav[1] if nav else 2.0

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

        self._initialized = True

    def standard_normalize(self, true_np_array, low, high):
        return 2 * (true_np_array - low) / (high - low) - 1

    def recover_trajectory(self, normalized_np_array, low, high):
        return (normalized_np_array + 1) * (high - low) / 2 + low

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

    def prepare_state(self, all_data, nav_onehot: bool = False, use_his_trajectory: bool = False):
        his_speed = self._ensure_2d(self._normalize_by_mode(
            np.asarray(all_data["his_speed"]), self.speed_01, self.speed_99, self.speed_min, self.speed_max
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

        parts = [his_speed, his_acceleration, his_angular_velocity, his_theta]

        if use_his_trajectory:
            his_traj = all_data.get("his_trajectory")
            if his_traj is not None:
                his_traj = np.asarray(his_traj, dtype=np.float32)
                his_x = self._normalize_by_mode(his_traj[:, 0], self.his_x_01, self.his_x_99, self.his_x_min, self.his_x_max)
                his_y = self._normalize_by_mode(his_traj[:, 1], self.his_y_01, self.his_y_99, self.his_y_min, self.his_y_max)
                his_traj = np.stack([his_x, his_y], axis=1)
                target_len = his_theta.shape[0]
                if his_traj.shape[0] < target_len:
                    pad = np.zeros((target_len - his_traj.shape[0], 2), dtype=np.float32)
                    his_traj = np.concatenate([his_traj, pad], axis=0)
                elif his_traj.shape[0] > target_len:
                    his_traj = his_traj[-target_len:]
                parts.append(his_traj)
            else:
                parts.append(np.zeros((his_theta.shape[0], 2), dtype=np.float32))

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

        state = torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)
        return state

    def prepare_traj(self, all_data):
        if "fur_trajectory" in all_data:
            fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)
            fur_x = self._normalize_by_mode(fur[:, 0], self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max)
            fur_y = self._normalize_by_mode(fur[:, 1], self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max)
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
        x_norm = result[0, :, 0]
        y_norm = result[0, :, 1]

        pred_x = self._denormalize_by_mode(
            x_norm, self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max
        )
        pred_y = self._denormalize_by_mode(
            y_norm, self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max
        )
        return pred_x, pred_y
