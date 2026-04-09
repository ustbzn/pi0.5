import numpy as np
import torch
import json


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
        """
        Args:
            percentile_path: 统计文件路径（通常是 statistics.json）
            norm_mode: 归一化模式
                - 'percentile'
                - 'percentile_clip'
                - 'minmax'
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.percentile_path = percentile_path
        self.norm_mode = norm_mode or "percentile"

        # load statistics
        with open(self.percentile_path, "r") as f:
            loaded_data = json.load(f)

        # 兼容两种 statistics.json 格式：
        #   嵌套格式: {"percentile": {"speed": [...]}, "minmax": {"speed": [...]}}
        #   扁平格式: {"speed": [...], "speed_min": ..., "speed_max": ...}
        if "percentile" in loaded_data:
            pct = loaded_data["percentile"]
            mm = loaded_data.get("minmax", {})
        else:
            pct = loaded_data
            mm = {}

        # 分位数参数
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
        self.target_far_x_01 = pct["command_far_x"][0]
        self.target_far_x_99 = pct["command_far_x"][1]
        self.target_far_y_01 = pct["command_far_y"][0]
        self.target_far_y_99 = pct["command_far_y"][1]
        # navigation_id: 离散值 (0=straight, 1=turn left, 2=turn right)
        nav = pct.get("navigation_id")
        self.nav_id_01 = nav[0] if nav else 0.0
        self.nav_id_99 = nav[1] if nav else 2.0

        # min-max 参数：优先从嵌套 minmax 字段读取，再从扁平 *_min/*_max 读取，最后回退到分位数
        def _get_mm(field, idx, fallback_key, fallback_val):
            """从 mm[field][idx] 或 loaded_data[fallback_key] 取值，都没有则用 fallback_val"""
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
        self.target_far_x_min = _get_mm("command_far_x", 0, "command_far_x_min", self.target_far_x_01)
        self.target_far_x_max = _get_mm("command_far_x", 1, "command_far_x_max", self.target_far_x_99)
        self.target_far_y_min = _get_mm("command_far_y", 0, "command_far_y_min", self.target_far_y_01)
        self.target_far_y_max = _get_mm("command_far_y", 1, "command_far_y_max", self.target_far_y_99)
        # navigation_id minmax
        self.nav_id_min = _get_mm("navigation_id", 0, "navigation_id_min", self.nav_id_01)
        self.nav_id_max = _get_mm("navigation_id", 1, "navigation_id_max", self.nav_id_99)

        self._initialized = True

    # -------- 基础函数：原始分位数归一化 / 反归一化 --------
    def standard_normalize(self, true_np_array, low, high):
        return 2 * (true_np_array - low) / (high - low) - 1

    def recover_trajectory(self, normalized_np_array, low, high):
        return (normalized_np_array + 1) * (high - low) / 2 + low

    # -------- 内部工具：按模式选择归一化 / 反归一化 --------
    def _normalize_by_mode(self, x, p01, p99, vmin, vmax):
        """根据 norm_mode 对单个标量序列做归一化"""
        if self.norm_mode == "percentile":
            return self.standard_normalize(x, p01, p99)
        elif self.norm_mode == "percentile_clip":
            y = self.standard_normalize(x, p01, p99)
            return np.clip(y, -1.0, 1.0)
        elif self.norm_mode == "minmax":
            # 如果 min/max 无效（相等），回退到分位数
            if np.allclose(vmin, vmax):
                return self.standard_normalize(x, p01, p99)
            y = 2 * (x - vmin) / (vmax - vmin) - 1
            return np.clip(y, -1.0, 1.0)
        else:
            # 未知模式时回退到原始实现
            return self.standard_normalize(x, p01, p99)

    def _denormalize_by_mode(self, x_norm, p01, p99, vmin, vmax):
        """根据 norm_mode 对单个标量序列做反归一化"""
        if self.norm_mode in ("percentile", "percentile_clip"):
            return self.recover_trajectory(x_norm, p01, p99)
        elif self.norm_mode == "minmax":
            if np.allclose(vmin, vmax):
                return self.recover_trajectory(x_norm, p01, p99)
            return self.recover_trajectory(x_norm, vmin, vmax)
        else:
            return self.recover_trajectory(x_norm, p01, p99)

    # -------- navigation_id one-hot 编码 --------
    NUM_NAV_CLASSES = 5  # straight=0, turn_left=1, turn_right=2, u_turn=3, park=4

    def _nav_id_to_onehot(self, nav_id_array):
        """将 navigation_id 数组转为 one-hot 编码

        Args:
            nav_id_array: shape (cond_steps,) 的数组，值为 0~4 的整数

        Returns:
            shape (cond_steps, NUM_NAV_CLASSES) 的 one-hot 数组
        """
        ids = np.round(nav_id_array).astype(int)
        ids = np.clip(ids, 0, self.NUM_NAV_CLASSES - 1)
        onehot = np.zeros((len(ids), self.NUM_NAV_CLASSES), dtype=np.float32)
        onehot[np.arange(len(ids)), ids] = 1.0
        return onehot

    # -------- 对外接口：state / traj / infer_traj --------
    def prepare_state(self, all_data, nav_onehot: bool = False):
        """根据当前模式对状态进行归一化

        Args:
            all_data: pkl 数据字典
            nav_onehot: 是否将 navigation_id 编码为 one-hot 向量（5 维）。
                        False 时使用原始的 minmax 归一化标量（1 维），保持向后兼容。
        """
        his_speed = self._normalize_by_mode(
            all_data["his_speed"], self.speed_01, self.speed_99, self.speed_min, self.speed_max
        )
        his_acceleration = self._normalize_by_mode(
            all_data["his_acceleration"],
            self.acceleration_01,
            self.acceleration_99,
            self.acceleration_min,
            self.acceleration_max,
        )
        his_angular_velocity = self._normalize_by_mode(
            all_data["his_angular_velocity"],
            self.angular_velocity_01,
            self.angular_velocity_99,
            self.angular_velocity_min,
            self.angular_velocity_max,
        )
        his_theta = self._normalize_by_mode(
            all_data["his_theta"], self.theta_01, self.theta_99, self.theta_min, self.theta_max
        )

        parts = [
            his_speed[:, np.newaxis],
            his_acceleration,
            his_angular_velocity,
            his_theta[:, np.newaxis],
        ]

        if nav_onehot:
            # 5 维 one-hot，不做连续归一化
            nav_oh = self._nav_id_to_onehot(all_data["navigation_id"])
            parts.append(nav_oh)
        else:
            # 向后兼容：1 维 minmax 归一化标量
            navigation_id = self._normalize_by_mode(
                all_data["navigation_id"],
                self.nav_id_01,
                self.nav_id_99,
                self.nav_id_min,
                self.nav_id_max,
            )
            parts.append(navigation_id[:, np.newaxis])

        state = torch.tensor(np.concatenate(parts, axis=1))
        return state

    def prepare_traj(self, all_data):
        """根据当前模式对 fur_x / fur_y 进行归一化"""
        fur_x = self._normalize_by_mode(
            all_data["fur_x"], self.fur_x_01, self.fur_x_99, self.fur_x_min, self.fur_x_max
        )
        fur_y = self._normalize_by_mode(
            all_data["fur_y"], self.fur_y_01, self.fur_y_99, self.fur_y_min, self.fur_y_max
        )
        trajectory = torch.tensor(
            np.concatenate((fur_x[:, :, np.newaxis], fur_y[:, :, np.newaxis]), axis=2)
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