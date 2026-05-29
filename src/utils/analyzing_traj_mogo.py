"""
MOGO 数据集轨迹数据分析工具

用于加载、展示和分析 MOGO 数据集的轨迹 npy 文件
支持 MOGO 数据集的目录结构：

1. 嵌套结构：
   - 20-bags/
     - 20251210141637/
       - drivepi/
         - 00001.pkl
         - 00002.pkl
         - ...
     - 20251210144033/
       - drivepi/
         - 00001.pkl
         - ...

2. 扁平化结构（新增支持）：
   - datasets/mogo-test/val/
     - 20251210141637_step1.pkl
     - 20251210141637_step2.pkl
     - ...
     - 20251210144033_step1.pkl
     - ...

轨迹 npy 文件命名格式：
- {timestamp}_gt_trajectories.npy
- {timestamp}_pred_trajectories.npy

支持基本统计信息展示、可视化轨迹、计算误差指标等功能

"""

import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from typing import Optional, Tuple, List, Dict, Set
import cv2
from tqdm import tqdm
import os
import pickle
import re
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.metric import get_action_accuracy
from src.utils.bev_axes import configure_bev_axes
from src.utils.collision_time_config import get_default_collision_interp_dt
from src.data.utils.normalization import Normalize

DEFAULT_HDMAP_POINT_CLOUD_RANGE = (-48.0, -81.6, -1.0, 48.0, 97.6, 3.0)
DEFAULT_HDMAP_VOXEL_SIZE = (0.1, 0.1, 0.2)
DEFAULT_BEV_XLIM = (-30.0, 60.0)
DEFAULT_BEV_YLIM = (-40.0, 40.0)
DEFAULT_HDMAP_ALPHA = 1.0
DUAL_VIEW_FIGSIZE = (18, 7.2)
DUAL_VIEW_DPI = 150

# 元组结构: (channel_idx, label, rgb_color, channel_alpha)
# channel_idx 对应 BEV npz 中的物理通道下标 (0=lanes, 1=lane_links,
# 2=lane_boundaries, 3=junctions)，与列表顺序解耦。
# 列表内的顺序决定**绘制先后**：靠前的先画、被靠后的覆盖在上面。
# 想让 junctions 不挡住车道线，把它放在前面 + 保持低 alpha 即可。
HDMAP_CHANNEL_STYLES = (
    (3, "junctions", (1.00, 0.78, 0.20), 0.12),
    (0, "lanes", (0.02, 0.02, 0.02), 0.35),
    (1, "lane_links", (0.10, 0.55, 0.25), 0.45),
    (2, "lane_boundaries", (0.05, 0.30, 0.85), 0.70),
)

OBJECT_BOX_ALPHA = 0.28
OBJECT_BOX_LINEWIDTH = 1.4
OBJECT_HEADING_LENGTH = 2.5
OBJECT_CLASS_STYLES = {
    0: ("car", "#ff7f0e"),
    1: ("pedestrian", "#9467bd"),
    2: ("bus/truck", "#8c564b"),
    3: ("rider", "#17becf"),
    4: ("tricycle", "#bcbd22"),
    5: ("unknown", "#7f7f7f"),
}

# navigation_id -> 映射前的导航指令文本（与 json_to_pkl_py3.py 中 NAVIGATION_COMMAND_MAP 反向对应）
NAVIGATION_ID_TO_TEXT = {
    0: "straight",
    1: "turn left",
    2: "turn right",
    3: "u turn",
    4: "park",
}
# 反向映射：文本 -> id（用于从 navigation_command 推断 nav_id）
NAVIGATION_COMMAND_TO_ID = {v: k for k, v in NAVIGATION_ID_TO_TEXT.items()}

HDMAP_BEV_EXTENT = [48.0, -48.0, -89.6, 89.6]
HDMAP_OVERLAY_CONFIG = {
    "lanes": {
        "channel_idx": 0,
        "color": "#74c476",
        "alpha": 0.18,
        "label": "Lane Area",
        "zorder": 0,
    },
    "junctions": {
        "channel_idx": 3,
        "color": "#f6b04d",
        "alpha": 0.22,
        "label": "Junction",
        "zorder": 1,
    },
    "lane_boundaries": {
        "channel_idx": 2,
        "color": "#33d1ff",
        "alpha": 0.28,
        "label": "Lane Boundary",
        "zorder": 2,
    },
}


class MOGOTrajectoryAnalyzer:
    """MOGO 数据集轨迹分析器"""
    
    def __init__(self, trajectory_dir: str, pkl_root_dir: Optional[str] = None):
        """
        初始化轨迹分析器
        
        Args:
            trajectory_dir: 轨迹 npy 文件目录路径（包含所有时间戳的 npy 文件）
                - 期望格式：{timestamp}_gt_trajectories.npy 和 {timestamp}_pred_trajectories.npy
            pkl_root_dir: pkl 文件根目录，用于 dual 视图。支持两种目录结构：
                - 嵌套结构：{pkl_root_dir}/{timestamp}/drivepi/00001.pkl
                - 扁平化结构：{pkl_root_dir}/{timestamp}_step1.pkl, {timestamp}_step2.pkl, ...
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.pkl_root_dir = Path(pkl_root_dir) if pkl_root_dir else None
        self.gt_trajectories = None
        self.pred_trajectories = None
        
        # 场景相关数据（时间戳作为场景名）
        self.scene_data: Dict[str, Dict] = {}  # {timestamp: {'gt': array, 'pred': array, 'start_idx': int}}
        self.scene_names: List[str] = []  # 时间戳列表
        
        # pkl文件映射：用于dual视图时根据sample_idx找到对应的pkl文件
        # 格式：{sample_idx: (timestamp, pkl_path)}
        self.pkl_file_map: Dict[int, Tuple[str, Path]] = {}
        self._warned_hdmap_paths = set()
        if self.pkl_root_dir and self.pkl_root_dir.exists():
            self._build_pkl_file_map()
    
    def _build_pkl_file_map(self):
        """构建pkl文件映射：sample_idx -> (timestamp, pkl_path)"""
        if self.pkl_root_dir is None:
            return
        
        # 按时间戳组织pkl文件
        timestamp_pkl_map: Dict[str, List[Tuple[int, Path]]] = {}
        
        # 首先检查是否是扁平化结构（目录下直接有 pkl 文件）
        pkl_files_in_root = list(self.pkl_root_dir.glob("*.pkl"))
        
        if pkl_files_in_root:
            # 扁平化结构：直接包含 pkl 文件，格式如 20251210141637_step1.pkl
            timestamp_groups: Dict[str, List[Tuple[int, Path]]] = {}
            
            for pkl_file in pkl_files_in_root:
                # 尝试匹配时间戳_stepN格式（支持纯数字或带字母/连字符的前缀，如 RBS-242768_step1.pkl）
                match_timestamp = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
                if match_timestamp:
                    timestamp = match_timestamp.group(1)
                    step_num = int(match_timestamp.group(2))
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = []
                    timestamp_groups[timestamp].append((step_num, pkl_file))
                else:
                    # 如果不是时间戳格式，尝试匹配纯数字格式：00001.pkl
                    match_num = re.match(r'(\d+)\.pkl$', pkl_file.name)
                    if match_num:
                        # 对于纯数字格式，使用目录名作为时间戳
                        timestamp = self.pkl_root_dir.name
                        file_num = int(match_num.group(1))
                        if timestamp not in timestamp_groups:
                            timestamp_groups[timestamp] = []
                        timestamp_groups[timestamp].append((file_num, pkl_file))
            
            # 按时间戳分组后，对每个时间戳内的文件按编号排序
            for timestamp, pkl_files in timestamp_groups.items():
                timestamp_pkl_map[timestamp] = sorted(pkl_files, key=lambda x: x[0])
        else:
            # 嵌套结构：遍历所有时间戳目录
            for timestamp_dir in sorted(self.pkl_root_dir.iterdir()):
                if not timestamp_dir.is_dir():
                    continue
                
                timestamp = timestamp_dir.name
                drivepi_dir = timestamp_dir / "drivepi"
                
                if not drivepi_dir.exists() or not drivepi_dir.is_dir():
                    continue
                
                # 查找所有 pkl 文件，支持两种格式：
                # 1. 纯数字格式：00001.pkl
                # 2. 时间戳_step格式：20251210141637_step1.pkl
                pkl_files = []
                for pkl_file in drivepi_dir.glob("*.pkl"):
                    # 首先尝试匹配时间戳_step格式（支持纯数字或带字母/连字符的前缀）
                    match_timestamp = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
                    if match_timestamp:
                        # 提取时间戳和step编号
                        file_timestamp = match_timestamp.group(1)
                        step_num = int(match_timestamp.group(2))
                        # 验证时间戳是否匹配目录名（可选，但更安全）
                        if file_timestamp == timestamp:
                            pkl_files.append((step_num, pkl_file))
                    else:
                        # 尝试匹配纯数字格式
                        match_num = re.match(r'(\d+)\.pkl$', pkl_file.name)
                        if match_num:
                            file_num = int(match_num.group(1))
                            pkl_files.append((file_num, pkl_file))
                
                if pkl_files:
                    # 按文件编号排序
                    pkl_files = sorted(pkl_files, key=lambda x: x[0])
                    timestamp_pkl_map[timestamp] = pkl_files
        
        # 按时间戳和文件编号排序，建立全局索引
        sample_idx = 0
        for timestamp in sorted(timestamp_pkl_map.keys()):
            pkl_files = timestamp_pkl_map[timestamp]
            for file_num, pkl_file in pkl_files:
                self.pkl_file_map[sample_idx] = (timestamp, pkl_file)
                sample_idx += 1
    
    def load_trajectories(self, scene_names: Optional[List[str]] = None):
        """
        加载轨迹数据
        
        Args:
            scene_names: 要加载的时间戳列表（None表示加载所有时间戳）
        """
        print(f"检测到 MOGO 数据集格式，正在加载...")
        
        # 找到所有时间戳的轨迹文件
        all_timestamps = set()
        for file in self.trajectory_dir.glob("*_gt_trajectories.npy"):
            # 提取时间戳：从文件名中提取
            match = re.match(r'(.+?)_gt_trajectories\.npy$', file.name)
            if match:
                timestamp = match.group(1)
                all_timestamps.add(timestamp)
        
        if not all_timestamps:
            raise FileNotFoundError(f"在 {self.trajectory_dir} 中未找到任何轨迹文件（格式：*_gt_trajectories.npy）")
        
        # 确定要加载的时间戳
        if scene_names is None:
            scene_names = sorted(all_timestamps)
        else:
            # 验证时间戳是否存在
            missing_scenes = [s for s in scene_names if s not in all_timestamps]
            if missing_scenes:
                raise FileNotFoundError(f"以下时间戳文件不存在: {missing_scenes}")
        
        self.scene_names = scene_names
        
        # 加载每个时间戳的数据
        all_gt_trajectories = []
        all_pred_trajectories = []
        current_idx = 0
        
        for timestamp in tqdm(scene_names, desc="Loading timestamps"):
            gt_path = self.trajectory_dir / f"{timestamp}_gt_trajectories.npy"
            pred_path = self.trajectory_dir / f"{timestamp}_pred_trajectories.npy"
            
            if not gt_path.exists():
                print(f"警告: GT文件不存在，跳过时间戳 {timestamp}: {gt_path}")
                continue
            if not pred_path.exists():
                print(f"警告: 预测文件不存在，跳过时间戳 {timestamp}: {pred_path}")
                continue
            
            scene_gt = np.load(gt_path)  # [N, horizon, 2]
            scene_pred = np.load(pred_path)  # [N, horizon, 2]
            
            # 检查一致性
            if len(scene_gt) != len(scene_pred):
                print(f"警告: 时间戳 {timestamp} 的GT和预测数量不匹配，跳过")
                continue
            
            if scene_gt.shape[1:] != scene_pred.shape[1:]:
                print(f"警告: 时间戳 {timestamp} 的GT和预测形状不匹配，跳过")
                continue
            
            # 存储场景数据
            self.scene_data[timestamp] = {
                'gt': scene_gt,
                'pred': scene_pred,
                'start_idx': current_idx,
                'num_samples': len(scene_gt)
            }
            
            all_gt_trajectories.append(scene_gt)
            all_pred_trajectories.append(scene_pred)
            current_idx += len(scene_gt)
        
        if len(all_gt_trajectories) == 0:
            raise ValueError("没有找到有效的轨迹文件")

        # 合并所有场景
        self.gt_trajectories = np.concatenate(all_gt_trajectories, axis=0)
        self.pred_trajectories = np.concatenate(all_pred_trajectories, axis=0)

        # ----------------------------------------------------------------
        # 关键：重建 pkl_file_map，使其与当前已加载场景的索引严格对齐。
        # _build_pkl_file_map 按全量时间戳排序编号，与 scene_names 过滤后
        # 的 current_idx 空间不同，必须在 load_trajectories 结束后重建。
        # ----------------------------------------------------------------
        if self.pkl_root_dir is not None:
            self.pkl_file_map = {}
            # 重新收集各时间戳的 pkl 文件（复用 _build_pkl_file_map 的解析逻辑）
            timestamp_pkl_map: Dict[str, List] = {}
            pkl_files_in_root = list(self.pkl_root_dir.glob("*.pkl"))
            if pkl_files_in_root:
                for pkl_file in pkl_files_in_root:
                    m = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
                    if m:
                        ts, step = m.group(1), int(m.group(2))
                        timestamp_pkl_map.setdefault(ts, []).append((step, pkl_file))
            else:
                for ts_dir in sorted(self.pkl_root_dir.iterdir()):
                    if not ts_dir.is_dir():
                        continue
                    ts = ts_dir.name
                    drivepi_dir = ts_dir / "drivepi"
                    search_dir = drivepi_dir if drivepi_dir.exists() else ts_dir
                    for pkl_file in search_dir.glob("*.pkl"):
                        m = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
                        if m and m.group(1) == ts:
                            timestamp_pkl_map.setdefault(ts, []).append((int(m.group(2)), pkl_file))
                            continue
                        m2 = re.match(r'(\d+)\.pkl$', pkl_file.name)
                        if m2:
                            timestamp_pkl_map.setdefault(ts, []).append((int(m2.group(1)), pkl_file))
            # 排序
            for ts in timestamp_pkl_map:
                timestamp_pkl_map[ts] = sorted(timestamp_pkl_map[ts], key=lambda x: x[0])

            # 只为已加载的场景建立映射，start_idx 严格取自 scene_data
            for timestamp in self.scene_names:
                if timestamp not in self.scene_data:
                    continue
                if timestamp not in timestamp_pkl_map:
                    continue
                start_idx = self.scene_data[timestamp]['start_idx']
                for offset, (_, pkl_file) in enumerate(timestamp_pkl_map[timestamp]):
                    self.pkl_file_map[start_idx + offset] = (timestamp, pkl_file)
        
        print(f"✓ 加载完成")
        print(f"  加载了 {len(self.scene_names)} 个时间戳")
        print(f"  总样本数: {len(self.gt_trajectories)}")
        print(f"  轨迹形状: {self.gt_trajectories.shape}")
        
        # 打印每个时间戳的信息
        for timestamp in self.scene_names:
            if timestamp in self.scene_data:
                info = self.scene_data[timestamp]
                print(f"    - {timestamp}: {info['num_samples']} 个样本 (索引 {info['start_idx']}-{info['start_idx']+info['num_samples']-1})")
    
    def print_basic_info(self):
        """打印基本信息"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        
        print("\n" + "="*60)
        print("轨迹数据基本信息 (MOGO 数据集)")
        print("="*60)
        
        num_samples, horizon, dim = self.gt_trajectories.shape
        
        print(f"\n时间戳数量: {len(self.scene_names)}")
        print(f"时间戳列表: {', '.join(self.scene_names[:5])}{'...' if len(self.scene_names) > 5 else ''}")
        
        print(f"\n数据维度:")
        print(f"  样本数量: {num_samples}")
        print(f"  时间步数 (horizon): {horizon}")
        print(f"  坐标维度: {dim}")
        
        print(f"\n数据类型:")
        print(f"  GT轨迹: {self.gt_trajectories.dtype}")
        print(f"  预测轨迹: {self.pred_trajectories.dtype}")
        
        print(f"\nGT轨迹统计信息:")
        print(f"  X坐标范围: [{self.gt_trajectories[:, :, 0].min():.3f}, {self.gt_trajectories[:, :, 0].max():.3f}]")
        print(f"  Y坐标范围: [{self.gt_trajectories[:, :, 1].min():.3f}, {self.gt_trajectories[:, :, 1].max():.3f}]")
        print(f"  X坐标均值: {self.gt_trajectories[:, :, 0].mean():.3f}")
        print(f"  Y坐标均值: {self.gt_trajectories[:, :, 1].mean():.3f}")
        print(f"  X坐标标准差: {self.gt_trajectories[:, :, 0].std():.3f}")
        print(f"  Y坐标标准差: {self.gt_trajectories[:, :, 1].std():.3f}")
        
        print(f"\n预测轨迹统计信息:")
        print(f"  X坐标范围: [{self.pred_trajectories[:, :, 0].min():.3f}, {self.pred_trajectories[:, :, 0].max():.3f}]")
        print(f"  Y坐标范围: [{self.pred_trajectories[:, :, 1].min():.3f}, {self.pred_trajectories[:, :, 1].max():.3f}]")
        print(f"  X坐标均值: {self.pred_trajectories[:, :, 0].mean():.3f}")
        print(f"  Y坐标均值: {self.pred_trajectories[:, :, 1].mean():.3f}")
        print(f"  X坐标标准差: {self.pred_trajectories[:, :, 0].std():.3f}")
        print(f"  Y坐标标准差: {self.pred_trajectories[:, :, 1].std():.3f}")
        
        # 显示前几个样本的轨迹
        print(f"\n前3个样本的GT轨迹 (前5个时间步):")
        for i in range(min(3, num_samples)):
            scene_info = self.get_scene_info(i)
            if scene_info:
                timestamp, local_idx = scene_info
                print(f"  样本 {i} ({timestamp}, step {local_idx}):")
            else:
                print(f"  样本 {i}:")
            for t in range(min(5, horizon)):
                x, y = self.gt_trajectories[i, t]
                print(f"    t={t}: ({x:.3f}, {y:.3f})")
        
        print("="*60)
    
    def compute_errors(self):
        """计算基于真实坐标的 L2 误差统计"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return None
        
        print("\n" + "="*60)
        print("误差分析")
        print("="*60)
        
        # 计算L2距离（欧氏距离）
        diff = self.pred_trajectories - self.gt_trajectories
        l2_errors = np.sqrt(np.sum(diff ** 2, axis=2))  # [N, horizon]
        
        # 计算每个时间步的平均误差
        mean_errors_per_step = l2_errors.mean(axis=0)
        std_errors_per_step = l2_errors.std(axis=0)
        
        print(f"\n各时间步的平均L2误差:")
        for t in range(len(mean_errors_per_step)):
            print(f"  t={t}: {mean_errors_per_step[t]:.4f} ± {std_errors_per_step[t]:.4f}")
        
        # 计算整体统计
        overall_mean = l2_errors.mean()
        overall_std = l2_errors.std()
        overall_median = np.median(l2_errors)
        
        print(f"\n整体误差统计:")
        print(f"  平均L2误差: {overall_mean:.4f}")
        print(f"  标准差: {overall_std:.4f}")
        print(f"  中位数: {overall_median:.4f}")
        print(f"  最小误差: {l2_errors.min():.4f}")
        print(f"  最大误差: {l2_errors.max():.4f}")
        
        # 计算最终时间步的误差
        final_errors = l2_errors[:, -1]
        print(f"\n最终时间步 (t={len(mean_errors_per_step)-1}) 误差:")
        print(f"  平均: {final_errors.mean():.4f}")
        print(f"  中位数: {np.median(final_errors):.4f}")
        print(f"  标准差: {final_errors.std():.4f}")
        
        print("="*60)
        
        return {
            'l2_errors': l2_errors,
            'mean_errors_per_step': mean_errors_per_step,
            'std_errors_per_step': std_errors_per_step,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_median': overall_median,
            'final_errors': final_errors
        }
    
    def compute_eval_style_metrics(
        self,
        statistic_path: str = "statistics.json",
        eval_thresholds: Optional[List[float]] = None,
        norm_mode: str = "percentile",
    ) -> Dict[str, object]:
        """按照 eval_old.py 的风格计算评估指标

        ``norm_mode`` 必须与训练时 ``cfg.data.val.norm_mode`` 完全一致，否则 acc
        会落到错误的归一化空间，造成与训练 eval 不可比。直接构造新的 Normalize
        而不是用单例，避免 ``Normalize._instance`` 被同进程其他调用提前固化。
        """
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return {}

        if eval_thresholds is None:
            eval_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

        print("\n" + "=" * 60)
        print("Eval-style 指标")
        print("=" * 60)

        # 真实坐标下的 L1 / L2
        diff = self.pred_trajectories - self.gt_trajectories
        l1_error = np.mean(np.abs(diff))
        l2_error = np.mean(diff ** 2)

        print(f"真实坐标 L1 Loss: {l1_error:.4f}")
        print(f"真实坐标 L2 Loss (MSE): {l2_error:.4f}")

        # 归一化空间下的多阈值准确率
        normalize = Normalize(statistic_path, norm_mode=norm_mode)
        print(f"acc 评测使用 norm_mode = {normalize.norm_mode}")

        gt_x = self.gt_trajectories[:, :, 0]
        gt_y = self.gt_trajectories[:, :, 1]
        pred_x = self.pred_trajectories[:, :, 0]
        pred_y = self.pred_trajectories[:, :, 1]

        gt_x_norm = normalize._normalize_by_mode(
            gt_x, normalize.fur_x_01, normalize.fur_x_99,
            normalize.fur_x_min, normalize.fur_x_max,
        )
        gt_y_norm = normalize._normalize_by_mode(
            gt_y, normalize.fur_y_01, normalize.fur_y_99,
            normalize.fur_y_min, normalize.fur_y_max,
        )
        pred_x_norm = normalize._normalize_by_mode(
            pred_x, normalize.fur_x_01, normalize.fur_x_99,
            normalize.fur_x_min, normalize.fur_x_max,
        )
        pred_y_norm = normalize._normalize_by_mode(
            pred_y, normalize.fur_y_01, normalize.fur_y_99,
            normalize.fur_y_min, normalize.fur_y_max,
        )

        gt_norm = np.stack([gt_x_norm, gt_y_norm], axis=-1)
        pred_norm = np.stack([pred_x_norm, pred_y_norm], axis=-1)

        gt_tensor = torch.from_numpy(gt_norm).float()
        pred_tensor = torch.from_numpy(pred_norm).float()

        acc_tensor = get_action_accuracy(gt_tensor, pred_tensor, thresholds=eval_thresholds)
        accuracies = acc_tensor.cpu().numpy()

        print("\n归一化空间多阈值准确率（与训练 eval 一致）：")
        for t, acc in zip(eval_thresholds, accuracies):
            print(f"  acc thres {t}: {acc:.4f}")

        print("=" * 60)

        return {
            "l1_loss": l1_error,
            "l2_loss": l2_error,
            "thresholds": eval_thresholds,
            "accuracies": accuracies,
            "norm_mode": normalize.norm_mode,
        }

    @staticmethod
    def _normalize_angle(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _estimate_yaw_from_traj(
        traj_xy: np.ndarray,
        fallback_yaw: float,
        valid_mask: Optional[np.ndarray] = None,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """根据轨迹切线估计每个时刻的朝向，无法估计时回退到 fallback_yaw。"""
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        num_steps = traj_xy.shape[0]
        yaws = np.full((num_steps,), float(fallback_yaw), dtype=np.float32)

        if valid_mask is None:
            valid_mask = np.ones((num_steps,), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

        for t in range(num_steps):
            if not valid_mask[t]:
                continue
            prev_idx = t - 1
            next_idx = t + 1
            direction = None
            if next_idx < num_steps and valid_mask[next_idx]:
                delta = traj_xy[next_idx] - traj_xy[t]
                if float(np.linalg.norm(delta)) > eps:
                    direction = delta
            if direction is None and prev_idx >= 0 and valid_mask[prev_idx]:
                delta = traj_xy[t] - traj_xy[prev_idx]
                if float(np.linalg.norm(delta)) > eps:
                    direction = delta
            if direction is not None:
                yaws[t] = np.arctan2(direction[1], direction[0]).astype(np.float32)
        return yaws

    @staticmethod
    def _interpolate_polyline(
        points_xy: np.ndarray,
        src_times: np.ndarray,
        target_times: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对轨迹做线性插值，返回插值结果和有效 mask。"""
        points_xy = np.asarray(points_xy, dtype=np.float32)
        src_times = np.asarray(src_times, dtype=np.float32)
        target_times = np.asarray(target_times, dtype=np.float32)
        if valid_mask is None:
            valid_mask = np.ones((len(points_xy),), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

        valid_points = points_xy[valid_mask]
        valid_times = src_times[valid_mask]
        out = np.zeros((len(target_times), 2), dtype=np.float32)
        out_mask = np.zeros((len(target_times),), dtype=bool)

        if len(valid_times) == 0:
            return out, out_mask

        if len(valid_times) == 1:
            out[target_times == valid_times[0]] = valid_points[0]
            out_mask[target_times == valid_times[0]] = True
            return out, out_mask

        for i, t in enumerate(target_times):
            if t < valid_times[0] or t > valid_times[-1]:
                continue
            if np.any(np.isclose(valid_times, t, atol=1e-6)):
                idx = int(np.argmin(np.abs(valid_times - t)))
                out[i] = valid_points[idx]
                out_mask[i] = True
                continue
            right = int(np.searchsorted(valid_times, t, side="right"))
            left = right - 1
            if left < 0 or right >= len(valid_times):
                continue
            t0, t1 = valid_times[left], valid_times[right]
            if float(t1 - t0) <= 1e-6:
                out[i] = valid_points[left]
                out_mask[i] = True
                continue
            ratio = float((t - t0) / (t1 - t0))
            out[i] = valid_points[left] + ratio * (valid_points[right] - valid_points[left])
            out_mask[i] = True
        return out, out_mask

    @staticmethod
    def _box_corners(center_xy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
        """构造旋转框四角点。length 对应前后方向，width 对应左右方向。"""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        hl = max(float(length), 1e-3) * 0.5
        hw = max(float(width), 1e-3) * 0.5
        local = np.array(
            [
                [hl, hw],
                [hl, -hw],
                [-hl, -hw],
                [-hl, hw],
            ],
            dtype=np.float32,
        )
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        rot = np.array(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
            dtype=np.float32,
        )
        return local @ rot.T + np.array([cx, cy], dtype=np.float32)

    @staticmethod
    def _polygons_overlap_sat(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1e-6) -> bool:
        """使用 SAT 判定两个凸四边形是否相交。"""
        for poly in (poly_a, poly_b):
            for i in range(len(poly)):
                p0 = poly[i]
                p1 = poly[(i + 1) % len(poly)]
                edge = p1 - p0
                axis = np.array([-edge[1], edge[0]], dtype=np.float32)
                norm = float(np.linalg.norm(axis))
                if norm <= eps:
                    continue
                axis = axis / norm
                proj_a = poly_a @ axis
                proj_b = poly_b @ axis
                if float(np.max(proj_a)) < float(np.min(proj_b)) - eps:
                    return False
                if float(np.max(proj_b)) < float(np.min(proj_a)) - eps:
                    return False
        return True

    def _load_raw_pkl(self, sample_idx: int) -> Optional[Dict[str, object]]:
        if self.pkl_root_dir is None or sample_idx not in self.pkl_file_map:
            return None
        _, pkl_path = self.pkl_file_map[sample_idx]
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f, encoding="latin-1")
        except Exception as e:
            print(f"加载 pkl 失败 {pkl_path}: {e}")
            return None

    @staticmethod
    def _load_hdmap_overlays(
        raw_pkl: Dict[str, object],
        channel_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """从 pkl 的 hdmap_bev 中读取指定通道，返回转置后的 BEV 叠加图。"""
        hdmap_bev_path = raw_pkl.get("hdmap_bev")
        if hdmap_bev_path is None:
            return {}
        if isinstance(hdmap_bev_path, bytes):
            hdmap_bev_path = hdmap_bev_path.decode("utf-8")
        if not hdmap_bev_path or not os.path.exists(hdmap_bev_path):
            return {}
        try:
            bev = np.load(hdmap_bev_path)["hdmap_bev"].astype(np.float32)
        except Exception:
            return {}
        if bev.ndim != 3:
            return {}

        selected_names = channel_names or ["lanes", "junctions", "lane_boundaries"]
        overlays = {}
        for name in selected_names:
            config = HDMAP_OVERLAY_CONFIG.get(name)
            if config is None:
                continue
            channel_idx = int(config["channel_idx"])
            if channel_idx >= bev.shape[0]:
                continue
            # 转为绘图坐标: horizontal=Y(left-right), vertical=X(rear-front)
            overlays[name] = bev[channel_idx].T
        return overlays

    @staticmethod
    def _draw_hdmap_overlays(ax, overlays: Dict[str, np.ndarray]) -> List[str]:
        """在 BEV 坐标系中叠加 HDMap 图层，返回实际绘制的图层名。"""
        drawn_layers: List[str] = []
        for name in ("lanes", "junctions", "lane_boundaries"):
            overlay = overlays.get(name)
            config = HDMAP_OVERLAY_CONFIG.get(name)
            if overlay is None or config is None:
                continue
            mask = np.ma.masked_where(overlay <= 0.0, overlay)
            if mask.count() == 0:
                continue
            ax.imshow(
                mask,
                extent=HDMAP_BEV_EXTENT,
                origin="lower",
                cmap=ListedColormap([config["color"]]),
                alpha=float(config["alpha"]),
                interpolation="nearest",
                zorder=int(config["zorder"]),
            )
            drawn_layers.append(name)
        return drawn_layers

    @staticmethod
    def _build_bev_legend_handles(
        hdmap_layers: List[str],
        include_history: bool,
        include_collision_objects: bool = False,
        include_collision_box: bool = False,
    ) -> List[object]:
        handles: List[object] = []
        for layer_name in ("lanes", "junctions", "lane_boundaries"):
            if layer_name not in hdmap_layers:
                continue
            config = HDMAP_OVERLAY_CONFIG[layer_name]
            handles.append(
                Patch(
                    facecolor=config["color"],
                    edgecolor="none",
                    alpha=float(config["alpha"]),
                    label=str(config["label"]),
                )
            )
        handles.append(Line2D([0], [0], color="green", linewidth=2, label="GT"))
        handles.append(Line2D([0], [0], color="red", linewidth=2, label="Pred"))
        if include_history:
            handles.append(Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=2, label="History"))
        handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color="blue",
                markersize=8,
                linestyle="None",
                label="Ego Start",
            )
        )
        if include_collision_objects:
            handles.append(Line2D([0], [0], color="gray", linewidth=1.5, alpha=0.7, label="Other Agents"))
            handles.append(Line2D([0], [0], color="orange", linewidth=1.8, label="Colliding Agent"))
        if include_collision_box:
            handles.append(Line2D([0], [0], color="crimson", linewidth=2.0, label="Collision Box"))
        return handles

    def compute_collision_metrics(
        self,
        interpolation_dt: float = get_default_collision_interp_dt(),
        class_id_to_name: Optional[Dict[int, str]] = None,
    ) -> Dict[str, object]:
        """统计当前和未来预测轨迹上的碰撞率。"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return {}
        if self.pkl_root_dir is None:
            print("错误: 计算碰撞率需要提供 pkl_root_dir")
            return {}

        if class_id_to_name is None:
            class_id_to_name = {
                0: "CAR",
                1: "PEDESTRIAN",
                2: "BUS",
                3: "RIDER",
                4: "TRICYCLE",
                5: "UNKNOWN",
            }

        future_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
        dense_future_times = np.arange(0.0, future_times[-1] + 1e-6, interpolation_dt, dtype=np.float32)
        num_samples = len(self.pred_trajectories)

        current_collisions = np.zeros((num_samples,), dtype=bool)
        future_collisions = np.zeros((num_samples,), dtype=bool)
        overall_collisions = np.zeros((num_samples,), dtype=bool)
        future_collision_per_step = np.zeros((len(dense_future_times),), dtype=np.int64)
        gt_future_collision_per_step = np.zeros((len(dense_future_times),), dtype=np.int64)
        gt_future_collisions = np.zeros((num_samples,), dtype=bool)
        first_collision_time = np.full((num_samples,), np.nan, dtype=np.float32)
        collided_sample_infos: List[Dict[str, object]] = []
        gt_collided_sample_infos: List[Dict[str, object]] = []
        class_collision_counter: Dict[str, int] = {}
        gt_class_collision_counter: Dict[str, int] = {}
        valid_samples = 0

        print("\n" + "=" * 60)
        print("碰撞率分析")
        print("=" * 60)

        for sample_idx in tqdm(range(num_samples), desc="collision", unit="sample"):
            raw = self._load_raw_pkl(sample_idx)
            if raw is None:
                continue

            pred_future_xy = np.asarray(self.pred_trajectories[sample_idx], dtype=np.float32)
            gt_future_xy = np.asarray(self.gt_trajectories[sample_idx], dtype=np.float32)
            av_idx = int(np.asarray(raw.get("av_idx", 0)).item())

            boxes_wh = np.asarray(raw["boxes_wh"], dtype=np.float32)
            classes_type = np.asarray(raw["classes_type"], dtype=np.int32)
            agent_centers = np.asarray(raw["agent_centers"], dtype=np.float32)
            agent_yaws = np.asarray(raw["agent_rotate_angles"], dtype=np.float32)
            obj_future_xy = np.asarray(raw["y"], dtype=np.float32)
            obj_future_invalid = np.asarray(raw["y_padding_mask"], dtype=bool)

            if av_idx < 0 or av_idx >= len(boxes_wh):
                continue

            valid_samples += 1
            av_length, av_width = boxes_wh[av_idx]
            av_center0 = agent_centers[av_idx]
            av_current_yaw = 0.0
            av_pred_full = np.concatenate([av_center0.reshape(1, 2), pred_future_xy], axis=0)
            av_gt_full = np.concatenate([av_center0.reshape(1, 2), gt_future_xy], axis=0)
            av_src_times = np.concatenate([np.array([0.0], dtype=np.float32), future_times], axis=0)
            av_dense_pred_xy, av_dense_pred_mask = self._interpolate_polyline(
                av_pred_full,
                av_src_times,
                dense_future_times,
            )
            av_dense_gt_xy, av_dense_gt_mask = self._interpolate_polyline(
                av_gt_full,
                av_src_times,
                dense_future_times,
            )
            av_dense_pred_yaw = self._estimate_yaw_from_traj(av_dense_pred_xy, fallback_yaw=av_current_yaw, valid_mask=av_dense_pred_mask)
            av_dense_gt_yaw = self._estimate_yaw_from_traj(av_dense_gt_xy, fallback_yaw=av_current_yaw, valid_mask=av_dense_gt_mask)

            sample_current_collision = False
            sample_future_collision = False
            sample_gt_future_collision = False
            sample_first_collision_time = np.nan
            sample_gt_first_collision_time = np.nan
            sample_collided_classes = set()
            sample_gt_collided_classes = set()
            sample_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)
            sample_gt_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)

            av_poly_current = self._box_corners(av_center0, av_current_yaw, av_length, av_width)

            for obj_idx in range(len(boxes_wh)):
                if obj_idx == av_idx:
                    continue

                obj_length, obj_width = boxes_wh[obj_idx]
                obj_class_name = class_id_to_name.get(int(classes_type[obj_idx]), f"CLASS_{int(classes_type[obj_idx])}")
                obj_center0 = agent_centers[obj_idx]
                obj_yaw0 = float(agent_yaws[obj_idx])

                obj_poly_current = self._box_corners(obj_center0, obj_yaw0, obj_length, obj_width)
                if self._polygons_overlap_sat(av_poly_current, obj_poly_current):
                    sample_current_collision = True
                    sample_collided_classes.add(obj_class_name)
                    if np.isnan(sample_first_collision_time):
                        sample_first_collision_time = 0.0

                obj_future_invalid_i = obj_future_invalid[obj_idx]
                obj_future_valid_i = ~obj_future_invalid_i
                obj_future_full = np.concatenate([obj_center0.reshape(1, 2), obj_future_xy[obj_idx]], axis=0)
                obj_src_times = np.concatenate([np.array([0.0], dtype=np.float32), future_times], axis=0)
                obj_src_valid = np.concatenate([np.array([True], dtype=bool), obj_future_valid_i], axis=0)
                obj_dense_xy, obj_dense_mask = self._interpolate_polyline(
                    obj_future_full,
                    obj_src_times,
                    dense_future_times,
                    valid_mask=obj_src_valid,
                )
                obj_dense_yaw = self._estimate_yaw_from_traj(obj_dense_xy, fallback_yaw=obj_yaw0, valid_mask=obj_dense_mask)

                for t_idx, t_sec in enumerate(dense_future_times):
                    if t_idx == 0:
                        continue
                    if not (av_dense_pred_mask[t_idx] and obj_dense_mask[t_idx]):
                        continue
                    av_poly = self._box_corners(
                        av_dense_pred_xy[t_idx],
                        float(av_dense_pred_yaw[t_idx]),
                        av_length,
                        av_width,
                    )
                    obj_poly = self._box_corners(
                        obj_dense_xy[t_idx],
                        float(obj_dense_yaw[t_idx]),
                        obj_length,
                        obj_width,
                    )
                    if self._polygons_overlap_sat(av_poly, obj_poly):
                        sample_future_collision = True
                        sample_collision_steps[t_idx] = True
                        sample_collided_classes.add(obj_class_name)
                        if np.isnan(sample_first_collision_time):
                            sample_first_collision_time = float(t_sec)
                        break

                for t_idx, _ in enumerate(dense_future_times):
                    if t_idx == 0:
                        continue
                    if not (av_dense_gt_mask[t_idx] and obj_dense_mask[t_idx]):
                        continue
                    av_poly_gt = self._box_corners(
                        av_dense_gt_xy[t_idx],
                        float(av_dense_gt_yaw[t_idx]),
                        av_length,
                        av_width,
                    )
                    obj_poly = self._box_corners(
                        obj_dense_xy[t_idx],
                        float(obj_dense_yaw[t_idx]),
                        obj_length,
                        obj_width,
                    )
                    if self._polygons_overlap_sat(av_poly_gt, obj_poly):
                        sample_gt_future_collision = True
                        sample_gt_collision_steps[t_idx] = True
                        sample_gt_collided_classes.add(obj_class_name)
                        if np.isnan(sample_gt_first_collision_time):
                            sample_gt_first_collision_time = float(dense_future_times[t_idx])
                        break

            current_collisions[sample_idx] = sample_current_collision
            future_collisions[sample_idx] = sample_future_collision
            overall_collisions[sample_idx] = sample_current_collision or sample_future_collision
            gt_future_collisions[sample_idx] = sample_gt_future_collision
            first_collision_time[sample_idx] = sample_first_collision_time
            future_collision_per_step += sample_collision_steps.astype(np.int64)
            gt_future_collision_per_step += sample_gt_collision_steps.astype(np.int64)

            if overall_collisions[sample_idx]:
                scene_info = self.get_scene_info(sample_idx)
                timestamp, local_idx = scene_info if scene_info else ("unknown", sample_idx)
                collided_sample_infos.append(
                    {
                        "sample_idx": int(sample_idx),
                        "timestamp": str(timestamp),
                        "local_idx": int(local_idx),
                        "current_collision": bool(sample_current_collision),
                        "future_collision": bool(sample_future_collision),
                        "first_collision_time": None if np.isnan(sample_first_collision_time) else float(sample_first_collision_time),
                        "collided_classes": sorted(sample_collided_classes),
                    }
                )
                for cls_name in sample_collided_classes:
                    class_collision_counter[cls_name] = class_collision_counter.get(cls_name, 0) + 1
            if sample_gt_future_collision:
                scene_info = self.get_scene_info(sample_idx)
                timestamp, local_idx = scene_info if scene_info else ("unknown", sample_idx)
                gt_collided_sample_infos.append(
                    {
                        "sample_idx": int(sample_idx),
                        "timestamp": str(timestamp),
                        "local_idx": int(local_idx),
                        "first_collision_time": None if np.isnan(sample_gt_first_collision_time) else float(sample_gt_first_collision_time),
                        "collided_classes": sorted(sample_gt_collided_classes),
                    }
                )
                for cls_name in sample_gt_collided_classes:
                    gt_class_collision_counter[cls_name] = gt_class_collision_counter.get(cls_name, 0) + 1

        denom = max(valid_samples, 1)
        current_collision_rate = float(current_collisions.sum() / denom)
        future_collision_rate = float(future_collisions.sum() / denom)
        overall_collision_rate = float(overall_collisions.sum() / denom)
        gt_future_collision_rate = float(gt_future_collisions.sum() / denom)
        collision_rate_per_step = (future_collision_per_step / denom).astype(np.float32)
        gt_collision_rate_per_step = (gt_future_collision_per_step / denom).astype(np.float32)

        print(f"有效样本数: {valid_samples}")
        print(f"当前碰撞率 current_collision_rate: {current_collision_rate:.4f}")
        print(f"未来碰撞率 future_collision_rate: {future_collision_rate:.4f}")
        print(f"总体碰撞率 overall_collision_rate: {overall_collision_rate:.4f}")
        print(f"GT未来碰撞率 gt_future_collision_rate: {gt_future_collision_rate:.4f}")
        print(f"未来逐时刻碰撞率 ({interpolation_dt:.1f}s):")
        for t_sec, rate in zip(dense_future_times[1:], collision_rate_per_step[1:]):
            print(f"  t={t_sec:.1f}s: {rate:.4f}")
        print(f"GT未来逐时刻碰撞率 ({interpolation_dt:.1f}s):")
        for t_sec, rate in zip(dense_future_times[1:], gt_collision_rate_per_step[1:]):
            print(f"  t={t_sec:.1f}s: {rate:.4f}")
        if class_collision_counter:
            print("按类别统计的碰撞样本数:")
            for cls_name, count in sorted(class_collision_counter.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {cls_name}: {count}")
        if gt_class_collision_counter:
            print("按类别统计的 GT 碰撞样本数:")
            for cls_name, count in sorted(gt_class_collision_counter.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {cls_name}: {count}")
        print("=" * 60)

        return {
            "num_valid_samples": int(valid_samples),
            "current_collision_rate": current_collision_rate,
            "future_collision_rate": future_collision_rate,
            "overall_collision_rate": overall_collision_rate,
            "gt_future_collision_rate": gt_future_collision_rate,
            "dense_future_times_sec": dense_future_times.tolist(),
            "collision_rate_per_step": collision_rate_per_step.tolist(),
            "gt_collision_rate_per_step": gt_collision_rate_per_step.tolist(),
            "num_current_collisions": int(current_collisions.sum()),
            "num_future_collisions": int(future_collisions.sum()),
            "num_overall_collisions": int(overall_collisions.sum()),
            "num_gt_future_collisions": int(gt_future_collisions.sum()),
            "class_collision_counter": class_collision_counter,
            "gt_class_collision_counter": gt_class_collision_counter,
            "collided_samples": collided_sample_infos,
            "gt_collided_samples": gt_collided_sample_infos,
        }

    def _build_collision_snapshot(
        self,
        sample_idx: int,
        interpolation_dt: float = get_default_collision_interp_dt(),
    ) -> Optional[Dict[str, object]]:
        """为单个样本构建碰撞可视化所需的几何快照。"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            return None
        raw = self._load_raw_pkl(sample_idx)
        if raw is None:
            return None

        future_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
        dense_future_times = np.arange(0.0, future_times[-1] + 1e-6, interpolation_dt, dtype=np.float32)

        pred_future_xy = np.asarray(self.pred_trajectories[sample_idx], dtype=np.float32)
        gt_future_xy = np.asarray(self.gt_trajectories[sample_idx], dtype=np.float32)
        av_idx = int(np.asarray(raw.get("av_idx", 0)).item())
        boxes_wh = np.asarray(raw["boxes_wh"], dtype=np.float32)
        classes_type = np.asarray(raw["classes_type"], dtype=np.int32)
        agent_centers = np.asarray(raw["agent_centers"], dtype=np.float32)
        agent_yaws = np.asarray(raw["agent_rotate_angles"], dtype=np.float32)
        obj_future_xy = np.asarray(raw["y"], dtype=np.float32)
        obj_future_invalid = np.asarray(raw["y_padding_mask"], dtype=bool)

        if av_idx < 0 or av_idx >= len(boxes_wh):
            return None

        av_length, av_width = boxes_wh[av_idx]
        av_center0 = agent_centers[av_idx]
        av_pred_full = np.concatenate([av_center0.reshape(1, 2), pred_future_xy], axis=0)
        av_gt_full = np.concatenate([av_center0.reshape(1, 2), gt_future_xy], axis=0)
        av_src_times = np.concatenate([np.array([0.0], dtype=np.float32), future_times], axis=0)
        av_dense_pred_xy, av_dense_pred_mask = self._interpolate_polyline(
            av_pred_full, av_src_times, dense_future_times
        )
        av_dense_gt_xy, av_dense_gt_mask = self._interpolate_polyline(
            av_gt_full, av_src_times, dense_future_times
        )
        av_dense_pred_yaw = self._estimate_yaw_from_traj(
            av_dense_pred_xy, fallback_yaw=0.0, valid_mask=av_dense_pred_mask
        )
        av_dense_gt_yaw = self._estimate_yaw_from_traj(
            av_dense_gt_xy, fallback_yaw=0.0, valid_mask=av_dense_gt_mask
        )

        obj_snapshots: List[Dict[str, object]] = []
        pred_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)
        gt_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)
        pred_collided_obj_indices = set()
        gt_collided_obj_indices = set()
        first_pred_collision_time = np.nan
        first_gt_collision_time = np.nan

        for obj_idx in range(len(boxes_wh)):
            if obj_idx == av_idx:
                continue
            obj_length, obj_width = boxes_wh[obj_idx]
            obj_center0 = agent_centers[obj_idx]
            obj_yaw0 = float(agent_yaws[obj_idx])
            obj_future_valid_i = ~obj_future_invalid[obj_idx]
            obj_future_full = np.concatenate([obj_center0.reshape(1, 2), obj_future_xy[obj_idx]], axis=0)
            obj_src_times = np.concatenate([np.array([0.0], dtype=np.float32), future_times], axis=0)
            obj_src_valid = np.concatenate([np.array([True], dtype=bool), obj_future_valid_i], axis=0)
            obj_dense_xy, obj_dense_mask = self._interpolate_polyline(
                obj_future_full, obj_src_times, dense_future_times, valid_mask=obj_src_valid
            )
            obj_dense_yaw = self._estimate_yaw_from_traj(
                obj_dense_xy, fallback_yaw=obj_yaw0, valid_mask=obj_dense_mask
            )

            obj_pred_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)
            obj_gt_collision_steps = np.zeros((len(dense_future_times),), dtype=bool)
            for t_idx, t_sec in enumerate(dense_future_times):
                if not (av_dense_pred_mask[t_idx] and obj_dense_mask[t_idx]):
                    continue
                av_poly = self._box_corners(
                    av_dense_pred_xy[t_idx], float(av_dense_pred_yaw[t_idx]), av_length, av_width
                )
                obj_poly = self._box_corners(
                    obj_dense_xy[t_idx], float(obj_dense_yaw[t_idx]), obj_length, obj_width
                )
                if self._polygons_overlap_sat(av_poly, obj_poly):
                    obj_pred_collision_steps[t_idx] = True
                    pred_collision_steps[t_idx] = True
                    pred_collided_obj_indices.add(obj_idx)
                    if np.isnan(first_pred_collision_time):
                        first_pred_collision_time = float(t_sec)

            for t_idx, t_sec in enumerate(dense_future_times):
                if not (av_dense_gt_mask[t_idx] and obj_dense_mask[t_idx]):
                    continue
                av_poly_gt = self._box_corners(
                    av_dense_gt_xy[t_idx], float(av_dense_gt_yaw[t_idx]), av_length, av_width
                )
                obj_poly = self._box_corners(
                    obj_dense_xy[t_idx], float(obj_dense_yaw[t_idx]), obj_length, obj_width
                )
                if self._polygons_overlap_sat(av_poly_gt, obj_poly):
                    obj_gt_collision_steps[t_idx] = True
                    gt_collision_steps[t_idx] = True
                    gt_collided_obj_indices.add(obj_idx)
                    if np.isnan(first_gt_collision_time):
                        first_gt_collision_time = float(t_sec)

            obj_snapshots.append(
                {
                    "obj_idx": int(obj_idx),
                    "class_id": int(classes_type[obj_idx]),
                    "center0": obj_center0.tolist(),
                    "yaw0": obj_yaw0,
                    "length": float(obj_length),
                    "width": float(obj_width),
                    "dense_xy": obj_dense_xy.tolist(),
                    "dense_mask": obj_dense_mask.tolist(),
                    "dense_yaw": obj_dense_yaw.tolist(),
                    "pred_collision_steps": obj_pred_collision_steps.tolist(),
                    "gt_collision_steps": obj_gt_collision_steps.tolist(),
                }
            )

        return {
            "sample_idx": int(sample_idx),
            "dense_future_times_sec": dense_future_times.tolist(),
            "av_idx": int(av_idx),
            "av_length": float(av_length),
            "av_width": float(av_width),
            "av_dense_pred_xy": av_dense_pred_xy.tolist(),
            "av_dense_pred_mask": av_dense_pred_mask.tolist(),
            "av_dense_pred_yaw": av_dense_pred_yaw.tolist(),
            "av_dense_gt_xy": av_dense_gt_xy.tolist(),
            "av_dense_gt_mask": av_dense_gt_mask.tolist(),
            "av_dense_gt_yaw": av_dense_gt_yaw.tolist(),
            "pred_collision_steps": pred_collision_steps.tolist(),
            "gt_collision_steps": gt_collision_steps.tolist(),
            "pred_collided_obj_indices": sorted(int(v) for v in pred_collided_obj_indices),
            "gt_collided_obj_indices": sorted(int(v) for v in gt_collided_obj_indices),
            "first_pred_collision_time": None if np.isnan(first_pred_collision_time) else float(first_pred_collision_time),
            "first_gt_collision_time": None if np.isnan(first_gt_collision_time) else float(first_gt_collision_time),
            "objects": obj_snapshots,
        }

    def export_collision_visualizations(
        self,
        collision_metrics: Dict[str, object],
        output_dir: str,
        interpolation_dt: float = get_default_collision_interp_dt(),
        max_samples: int = 20,
        collision_type: str = "pred",
    ) -> None:
        """导出碰撞样本可视化图。"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        collided_samples = (
            collision_metrics.get("gt_collided_samples", [])
            if collision_type == "gt"
            else collision_metrics.get("collided_samples", [])
        )
        if not collided_samples:
            print(f"没有 {collision_type} 碰撞样本，跳过碰撞可视化导出")
            return

        class_name_map = {
            0: "CAR",
            1: "PEDESTRIAN",
            2: "BUS",
            3: "RIDER",
            4: "TRICYCLE",
            5: "UNKNOWN",
        }

        selected = collided_samples[:max_samples]
        for item in tqdm(selected, desc="collision vis", unit="img"):
            sample_idx = int(item["sample_idx"])
            result = self._load_images_from_pkl(sample_idx)
            if result is None:
                continue
            imgs, nav_text, his_traj = result
            raw_pkl = self._load_raw_pkl(sample_idx)
            snapshot = self._build_collision_snapshot(sample_idx, interpolation_dt=interpolation_dt)
            if snapshot is None:
                continue

            gt_traj = self.gt_trajectories[sample_idx]
            pred_traj = self.pred_trajectories[sample_idx]
            scene_info = self.get_scene_info(sample_idx)
            if scene_info:
                timestamp, local_idx = scene_info
                title_info = f"collision sample {sample_idx}, {timestamp}, step {local_idx}"
            else:
                title_info = f"collision sample {sample_idx}"
            if nav_text:
                title_info += f" | Nav: {nav_text}"

            fig = plt.figure(figsize=(20, 9))
            gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], hspace=0.25, wspace=0.12)

            cam_grid = [('front', 0, 0), ('back', 0, 1), ('left', 1, 0), ('right', 1, 1)]
            cam_labels = {'front': 'Front', 'right': 'Right', 'left': 'Left', 'back': 'Back'}
            for cam, row, col in cam_grid:
                ax = fig.add_subplot(gs[row, col])
                img = imgs.get(cam)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.set_facecolor('#cccccc')
                    ax.text(0.5, 0.5, f'{cam_labels[cam]}\n(N/A)', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12, color='#555555')
                ax.set_title(cam_labels[cam], fontsize=11)
                ax.axis("off")

            ax_bev = fig.add_subplot(gs[:, 2])
            hdmap_overlays = self._load_hdmap_overlays(raw_pkl) if raw_pkl is not None else {}
            drawn_hdmap_layers = self._draw_hdmap_overlays(ax_bev, hdmap_overlays)

            origin = np.array([[0.0, 0.0]], dtype=np.float32)
            gt_traj_with_origin = np.concatenate([origin, gt_traj], axis=0)
            pred_traj_with_origin = np.concatenate([origin, pred_traj], axis=0)
            ax_bev.plot(gt_traj_with_origin[:, 1], gt_traj_with_origin[:, 0], "g-", linewidth=2, zorder=4)
            ax_bev.plot(pred_traj_with_origin[:, 1], pred_traj_with_origin[:, 0], "r-", linewidth=2, zorder=4)
            if his_traj is not None and len(his_traj) > 0:
                his_with_origin = np.concatenate([his_traj, origin], axis=0)
                ax_bev.plot(his_with_origin[:, 1], his_with_origin[:, 0], color="tab:blue",
                            linestyle="--", linewidth=2, zorder=4)
            ax_bev.scatter(0, 0, color="blue", s=100, marker="s", zorder=6)

            dense_times = np.asarray(snapshot["dense_future_times_sec"], dtype=np.float32)
            if collision_type == "gt":
                av_xy = np.asarray(snapshot["av_dense_gt_xy"], dtype=np.float32)
                av_mask = np.asarray(snapshot["av_dense_gt_mask"], dtype=bool)
                av_yaw = np.asarray(snapshot["av_dense_gt_yaw"], dtype=np.float32)
                collision_steps = np.asarray(snapshot["gt_collision_steps"], dtype=bool)
                collided_obj_indices = set(snapshot["gt_collided_obj_indices"])
                first_collision_time = snapshot["first_gt_collision_time"]
            else:
                av_xy = np.asarray(snapshot["av_dense_pred_xy"], dtype=np.float32)
                av_mask = np.asarray(snapshot["av_dense_pred_mask"], dtype=bool)
                av_yaw = np.asarray(snapshot["av_dense_pred_yaw"], dtype=np.float32)
                collision_steps = np.asarray(snapshot["pred_collision_steps"], dtype=bool)
                collided_obj_indices = set(snapshot["pred_collided_obj_indices"])
                first_collision_time = snapshot["first_pred_collision_time"]
            av_length = float(snapshot["av_length"])
            av_width = float(snapshot["av_width"])

            all_pts = [gt_traj, pred_traj, origin, av_xy[av_mask]]
            if his_traj is not None and len(his_traj) > 0:
                all_pts.append(his_traj)

            for obj_item in snapshot["objects"]:
                obj_xy = np.asarray(obj_item["dense_xy"], dtype=np.float32)
                obj_mask = np.asarray(obj_item["dense_mask"], dtype=bool)
                obj_yaw = np.asarray(obj_item["dense_yaw"], dtype=np.float32)
                obj_collision_steps = np.asarray(
                    obj_item["gt_collision_steps"] if collision_type == "gt" else obj_item["pred_collision_steps"],
                    dtype=bool,
                )
                cls_name = class_name_map.get(int(obj_item["class_id"]), f"CLASS_{int(obj_item['class_id'])}")
                line_color = "orange" if np.any(obj_collision_steps) else "gray"
                alpha = 0.95 if np.any(obj_collision_steps) else 0.45
                if np.any(obj_mask):
                    ax_bev.plot(
                        obj_xy[obj_mask, 1],
                        obj_xy[obj_mask, 0],
                        color=line_color,
                        alpha=alpha,
                        linewidth=1.5,
                        zorder=4,
                    )
                    ax_bev.scatter(obj_xy[0, 1], obj_xy[0, 0], color=line_color, s=18, alpha=alpha, zorder=5)
                    all_pts.append(obj_xy[obj_mask])
                for t_idx, t_sec in enumerate(dense_times):
                    if not obj_mask[t_idx]:
                        continue
                    if t_idx == 0 or np.isclose(t_sec % 0.5, 0.0, atol=1e-6) or obj_collision_steps[t_idx]:
                        corners = self._box_corners(
                            obj_xy[t_idx],
                            float(obj_yaw[t_idx]),
                            float(obj_item["length"]),
                            float(obj_item["width"]),
                        )
                        poly = np.vstack([corners, corners[0]])
                        ax_bev.plot(poly[:, 1], poly[:, 0], color=line_color, alpha=alpha, linewidth=1.0, zorder=5)
                        if obj_collision_steps[t_idx]:
                            ax_bev.text(obj_xy[t_idx, 1], obj_xy[t_idx, 0], f"{cls_name}[{int(obj_item['obj_idx'])}]",
                                        color="darkred", fontsize=7, zorder=6)

            for t_idx, t_sec in enumerate(dense_times):
                if not av_mask[t_idx]:
                    continue
                if t_idx == 0 or np.isclose(t_sec % 0.5, 0.0, atol=1e-6) or collision_steps[t_idx]:
                    av_corners = self._box_corners(av_xy[t_idx], float(av_yaw[t_idx]), av_length, av_width)
                    av_poly = np.vstack([av_corners, av_corners[0]])
                    color = "crimson" if collision_steps[t_idx] else "tab:red"
                    width = 2.0 if collision_steps[t_idx] else 1.0
                    ax_bev.plot(av_poly[:, 1], av_poly[:, 0], color=color, linewidth=width, alpha=0.95, zorder=6)
                    if collision_steps[t_idx]:
                        ax_bev.text(av_xy[t_idx, 1], av_xy[t_idx, 0], f"{collision_type.upper()} COL@{t_sec:.1f}s",
                                    color="crimson", fontsize=8, fontweight="bold", zorder=7)

            all_pts = np.concatenate([pts for pts in all_pts if len(pts) > 0], axis=0)
            x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
            y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
            margin_x = max(2.0, 0.12 * (x_max - x_min + 1e-6))
            margin_y = max(2.0, 0.12 * (y_max - y_min + 1e-6))
            ax_bev.set_xlim((y_min - margin_y, y_max + margin_y))
            ax_bev.set_ylim((x_min - margin_x, x_max + margin_x))
            ax_bev.set_aspect("equal")
            ax_bev.grid(True, alpha=0.3)
            configure_bev_axes(ax_bev)
            ax_bev.set_title(f"Collision BEV ({collision_type.upper()})")
            legend_handles = self._build_bev_legend_handles(
                hdmap_layers=drawn_hdmap_layers,
                include_history=his_traj is not None and len(his_traj) > 0,
                include_collision_objects=True,
                include_collision_box=True,
            )
            ax_bev.legend(handles=legend_handles, loc="upper left", fontsize=8)

            summary_text = (
                f"type: {collision_type}\n"
                f"first_collision_time: {first_collision_time}\n"
                f"current_collision: {item.get('current_collision')}\n"
                f"future_collision: {item.get('future_collision')}\n"
                f"collided_classes: {', '.join(item.get('collided_classes', [])) or 'None'}\n"
                f"collided_obj_indices: {sorted(collided_obj_indices)}"
            )
            ax_bev.text(
                1.02, 1.0, summary_text, ha="left", va="top", transform=ax_bev.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

            fig.suptitle(title_info, fontsize=11, y=0.98)
            scene_dir = output_path / collision_type / str(item.get("timestamp", "unknown"))
            scene_dir.mkdir(parents=True, exist_ok=True)
            local_idx = int(item.get("local_idx", sample_idx))
            fig.savefig(scene_dir / f"collision_{local_idx:06d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def export_collision_class_table(
        collision_metrics: Dict[str, object],
        output_dir: str,
    ) -> None:
        """导出按类别统计的碰撞率表。"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        class_counter = collision_metrics.get("class_collision_counter", {})
        gt_class_counter = collision_metrics.get("gt_class_collision_counter", {})
        num_valid_samples = int(collision_metrics.get("num_valid_samples", 0))
        rows = []
        for cls_name, count in sorted(class_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            rate = float(count / num_valid_samples) if num_valid_samples > 0 else 0.0
            rows.append(("pred", cls_name, int(count), rate))
        for cls_name, count in sorted(gt_class_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            rate = float(count / num_valid_samples) if num_valid_samples > 0 else 0.0
            rows.append(("gt", cls_name, int(count), rate))

        txt_path = output_path / "collision_class_table.txt"
        csv_path = output_path / "collision_class_table.csv"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("collision_type,class_name,count,sample_level_rate\n")
            for collision_type, cls_name, count, rate in rows:
                f.write(f"{collision_type},{cls_name},{count},{rate:.6f}\n")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("collision_type,class_name,count,sample_level_rate\n")
            for collision_type, cls_name, count, rate in rows:
                f.write(f"{collision_type},{cls_name},{count},{rate:.6f}\n")
    
    @staticmethod
    def _extract_object_annotations(data: Dict) -> Optional[Dict[str, object]]:
        """从 pkl 字典中提取当前帧障碍物框绘制所需字段。"""
        required_keys = ("boxes_wh", "classes_type", "agent_centers", "agent_rotate_angles")
        if any(key not in data for key in required_keys):
            return None

        boxes_wh = np.asarray(data["boxes_wh"], dtype=np.float32)
        classes_type = np.asarray(data["classes_type"], dtype=np.int32).reshape(-1)
        agent_centers = np.asarray(data["agent_centers"], dtype=np.float32)
        agent_rotate_angles = np.asarray(data["agent_rotate_angles"], dtype=np.float32).reshape(-1)
        ids = np.asarray(data.get("ids", np.arange(len(classes_type))), dtype=np.int64).reshape(-1)

        if boxes_wh.ndim != 2 or boxes_wh.shape[1] != 2:
            return None
        if agent_centers.ndim != 2 or agent_centers.shape[1] != 2:
            return None

        num_agents = min(
            len(ids),
            len(boxes_wh),
            len(classes_type),
            len(agent_centers),
            len(agent_rotate_angles),
        )
        if num_agents == 0:
            return None

        def _scalar_int(value, default: int = -1) -> int:
            if value is None:
                return default
            try:
                arr = np.asarray(value)
                if arr.size == 0:
                    return default
                return int(arr.reshape(-1)[0])
            except (TypeError, ValueError):
                return default

        return {
            "ids": ids[:num_agents],
            "boxes_wh": boxes_wh[:num_agents],
            "classes_type": classes_type[:num_agents],
            "agent_centers": agent_centers[:num_agents],
            "agent_rotate_angles": agent_rotate_angles[:num_agents],
            "agent_idx": _scalar_int(data.get("agent_idx")),
            "av_idx": _scalar_int(data.get("av_idx")),
        }

    def _load_images_from_pkl(
        self, sample_idx: int
    ) -> Optional[Tuple[dict, Optional[str], Optional[np.ndarray], Optional[str], Optional[Dict[str, object]]]]:
        """从pkl加载4路摄像头图像（当前帧）、导航指令文本和历史轨迹

        Returns:
            (imgs, nav_text, his_traj, hdmap_bev_path, object_annotations)，其中 imgs 是
            {'front': ndarray, 'right': ndarray, 'left': ndarray, 'back': ndarray}
            （RGB格式，缺失的摄像头为 None），nav_text 为映射前的导航指令文本
            （如 "straight"），his_traj 为历史轨迹，坐标格式与 gt/pred 保持一致，
            均为 [x, y]，hdmap_bev_path 是高精地图 BEV npz 路径，
            object_annotations 为当前帧障碍物框信息。加载完全失败时返回 None。
        """
        if self.pkl_root_dir is None:
            print("错误: 未指定pkl文件根目录")
            return None

        if sample_idx not in self.pkl_file_map:
            print(f"错误: 样本索引 {sample_idx} 没有对应的pkl文件")
            return None

        timestamp, pkl_path = self.pkl_file_map[sample_idx]
        try:
            import pickle
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding='latin-1')

            def _decode_img_ref(img_ref):
                """将各种格式的图像路径引用解码为字符串路径"""
                if isinstance(img_ref, dict) and "_string_wrapper_marker" in img_ref:
                    img_ref = img_ref.get("_bytes_value", b"")
                if hasattr(img_ref, "numpy"):
                    img_ref = img_ref.numpy()
                if isinstance(img_ref, np.ndarray):
                    if img_ref.size == 0:
                        return None
                    img_ref = img_ref.reshape(-1)[0]
                if isinstance(img_ref, (list, tuple)):
                    if len(img_ref) == 0:
                        return None
                    img_ref = img_ref[0]
                if isinstance(img_ref, bytes):
                    return img_ref.decode("utf-8")
                elif isinstance(img_ref, str):
                    return img_ref
                return None

            def _load_cam(key):
                """读取单路摄像头图像，失败返回 None"""
                ref = data.get(key)
                if ref is None:
                    return None
                path = _decode_img_ref(ref)
                if not path or not Path(path).exists():
                    return None
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    return None
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 4路摄像头的当前帧字段（his_cam_* 为当前时刻图像）
            imgs = {
                'front': _load_cam('cam_front'),
                'right': _load_cam('cam_right'),
                'left':  _load_cam('cam_left'),
                'back':  _load_cam('cam_back'),
                # 'front': _load_cam('his_cam_front'),
                # 'right': _load_cam('his_cam_right'),
                # 'left':  _load_cam('his_cam_left'),
                # 'back':  _load_cam('his_cam_back'),
            }

            # 至少需要前视图才有意义
            if imgs['front'] is None:
                print(f"错误: pkl缺少 his_cam_front 字段或文件不存在: {pkl_path}")
                return None

            # 读取 navigation_id 并反向映射为文本
            nav_text = None
            if 'navigation_id' not in data:
                # 从 navigation_command 推断 nav_id，不依赖外部 import
                nav_id = 0
                cmd = data.get('navigation_command')
                if cmd is not None:
                    cmd_arr = np.asarray(cmd)
                    if cmd_arr.size > 0:
                        token = str(
                            cmd_arr.reshape(-1, cmd_arr.shape[-1])[0][0]
                            if cmd_arr.ndim >= 2 else cmd_arr.reshape(-1)[0]
                        ).strip().lower()
                        nav_id = NAVIGATION_COMMAND_TO_ID.get(token, 0)
                nav_id_val = nav_id
            else:
                nav_id_arr = data.get("navigation_id")
                if nav_id_arr is not None:
                    nav_id_val = int(nav_id_arr[0]) if hasattr(nav_id_arr, '__len__') and len(nav_id_arr) > 0 else int(nav_id_arr)
                else:
                    nav_id_val = 0
            nav_text = NAVIGATION_ID_TO_TEXT.get(nav_id_val, f"unknown({nav_id_val})")

            his_traj = None
            his_traj_raw = data.get("his_trajectory")
            if his_traj_raw is not None:
                his_traj_arr = np.asarray(his_traj_raw, dtype=np.float32)
                if his_traj_arr.ndim == 2 and his_traj_arr.shape[-1] == 2:
                    # 与当前训练 / 统计脚本保持一致：将存储中的 [y, x] 转为 [x, y]
                    his_traj = np.stack([his_traj_arr[:, 1], his_traj_arr[:, 0]], axis=-1)

            hdmap_bev_path = _decode_img_ref(data.get("hdmap_bev"))
            object_annotations = self._extract_object_annotations(data)

            return imgs, nav_text, his_traj, hdmap_bev_path, object_annotations
        except Exception as e:
            print(f"加载pkl失败 {pkl_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 保留旧接口名以兼容外部调用
    def _load_front_image_from_pkl(self, sample_idx: int) -> Optional[Tuple[np.ndarray, Optional[str]]]:
        result = self._load_images_from_pkl(sample_idx)
        if result is None:
            return None
        imgs, nav_text, _, _, _ = result
        return imgs['front'], nav_text

    def _warn_hdmap_once(self, hdmap_path: str, message: str):
        """避免批量可视化时对同一个 HDMap 问题重复刷屏。"""
        warn_key = (hdmap_path, message)
        if warn_key in self._warned_hdmap_paths:
            return
        self._warned_hdmap_paths.add(warn_key)
        print(f"警告: {message}: {hdmap_path}")

    @staticmethod
    def _load_hdmap_bev_npz(hdmap_path: Optional[str]) -> Optional[np.ndarray]:
        """读取 pkl 中 hdmap_bev 指向的 4 通道 BEV npz。"""
        if not hdmap_path:
            return None

        npz_path = Path(hdmap_path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"HDMap npz 不存在: {npz_path}")

        with np.load(npz_path) as npz_data:
            if "hdmap_bev" in npz_data:
                bev = np.array(npz_data["hdmap_bev"], dtype=np.float32)
            elif len(npz_data.files) == 1:
                bev = np.array(npz_data[npz_data.files[0]], dtype=np.float32)
            else:
                raise KeyError(f"未找到 hdmap_bev，可用 keys: {npz_data.files}")

        if bev.ndim != 3 or bev.shape[0] != 4:
            raise ValueError(f"期望 HDMap BEV shape 为 (4, H, W)，实际得到 {bev.shape}")
        return bev

    @staticmethod
    def _build_hdmap_rgba_overlay(bev: np.ndarray, alpha_scale: float) -> np.ndarray:
        """生成透明背景 HDMap 叠加层，使用 alpha over 合成保证多层互不抹除。"""
        rgba = np.zeros((bev.shape[1], bev.shape[2], 4), dtype=np.float32)
        alpha_scale = float(np.clip(alpha_scale, 0.0, 1.0))

        for channel_idx, _label, rgb_color, channel_alpha in HDMAP_CHANNEL_STYLES:
            mask = np.clip(bev[channel_idx], 0.0, 1.0)
            src_a = mask * channel_alpha * alpha_scale
            update_mask = src_a > 0.0
            if not np.any(update_mask):
                continue

            sa = src_a[update_mask].astype(np.float32)
            da = rgba[update_mask, 3].astype(np.float32)
            out_a = sa + da * (1.0 - sa)
            safe_oa = np.where(out_a > 0.0, out_a, 1.0)

            src_rgb = np.asarray(rgb_color, dtype=np.float32)
            dst_rgb = rgba[update_mask, :3]
            new_rgb = (
                src_rgb[None, :] * sa[:, None]
                + dst_rgb * da[:, None] * (1.0 - sa[:, None])
            ) / safe_oa[:, None]

            rgba[update_mask, :3] = new_rgb
            rgba[update_mask, 3] = out_a

        return rgba

    @staticmethod
    def _hdmap_legend_handles() -> List[Patch]:
        """构造 HDMap 图例句柄。"""
        return [
            Patch(facecolor=rgb_color, edgecolor="none", alpha=channel_alpha, label=label)
            for _channel_idx, label, rgb_color, channel_alpha in HDMAP_CHANNEL_STYLES
        ]

    @staticmethod
    def _object_legend_handles(class_ids: Set[int]) -> List[Patch]:
        """构造障碍物类别图例句柄。"""
        handles = []
        for class_id in sorted(class_ids):
            label, color = OBJECT_CLASS_STYLES.get(class_id, OBJECT_CLASS_STYLES[5])
            handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    alpha=OBJECT_BOX_ALPHA,
                    label=f"obj:{label}",
                )
            )
        return handles

    @staticmethod
    def _oriented_box_corners(
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """根据中心点、长宽和航向角生成 BEV 框角点，坐标格式为 [x_forward, y_left]。"""
        length, width = float(size[0]), float(size[1])
        local_corners = np.array(
            [
                [-length / 2.0, -width / 2.0],
                [ length / 2.0, -width / 2.0],
                [ length / 2.0,  width / 2.0],
                [-length / 2.0,  width / 2.0],
            ],
            dtype=np.float32,
        )
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw,  cos_yaw],
            ],
            dtype=np.float32,
        )
        return local_corners @ rotation.T + center.astype(np.float32)

    def _draw_object_boxes(
        self,
        ax,
        object_annotations: Optional[Dict[str, object]],
    ) -> Set[int]:
        """在 BEV 坐标系中绘制当前帧障碍物框，返回实际绘制到的类别。"""
        if object_annotations is None:
            return set()

        boxes_wh = object_annotations["boxes_wh"]
        classes_type = object_annotations["classes_type"]
        agent_centers = object_annotations["agent_centers"]
        agent_rotate_angles = object_annotations["agent_rotate_angles"]
        av_idx = int(object_annotations.get("av_idx", -1))
        drawn_classes: Set[int] = set()

        for idx, (center, size, yaw, class_id) in enumerate(
            zip(agent_centers, boxes_wh, agent_rotate_angles, classes_type)
        ):
            if idx == av_idx:
                continue
            if not (
                np.all(np.isfinite(center))
                and np.all(np.isfinite(size))
                and np.isfinite(yaw)
            ):
                continue
            if size[0] <= 0.0 or size[1] <= 0.0:
                continue

            class_id = int(class_id)
            _label, color = OBJECT_CLASS_STYLES.get(class_id, OBJECT_CLASS_STYLES[5])
            corners = self._oriented_box_corners(center, size, float(yaw))
            # 障碍物字段横向坐标为左正；当前 BEV 横轴为右正，绘制前需取反。
            plot_corners = np.column_stack([-corners[:, 1], corners[:, 0]])
            patch = Polygon(
                plot_corners,
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=OBJECT_BOX_LINEWIDTH,
                alpha=OBJECT_BOX_ALPHA,
                zorder=2,
            )
            ax.add_patch(patch)

            heading_len = min(OBJECT_HEADING_LENGTH, max(float(size[0]) * 0.5, 0.8))
            ax.arrow(
                float(-center[1]),
                float(center[0]),
                float(-np.sin(yaw) * heading_len),
                float(np.cos(yaw) * heading_len),
                color=color,
                width=0.04,
                head_width=0.45,
                head_length=0.6,
                alpha=0.85,
                length_includes_head=True,
                zorder=2.2,
            )
            drawn_classes.add(class_id if class_id in OBJECT_CLASS_STYLES else 5)

        return drawn_classes

    @staticmethod
    def _infer_hdmap_bev_layout(
        bev: np.ndarray,
        point_cloud_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
    ) -> str:
        """推断 BEV 空间维度是 (right, forward) 还是 (forward, right)。"""
        right_bins = int(round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]))
        forward_bins = int(round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]))

        if bev.shape[1:] == (right_bins, forward_bins):
            return "xy"
        if bev.shape[1:] == (forward_bins, right_bins):
            return "yx"

        raise ValueError(
            "无法根据 point_cloud_range/voxel_size 推断 HDMap BEV 布局，"
            f"bev.shape={bev.shape}, 期望 spatial 为 "
            f"({right_bins}, {forward_bins}) 或 ({forward_bins}, {right_bins})"
        )

    @staticmethod
    def _to_hdmap_display_image(image: np.ndarray, bev_layout: str) -> np.ndarray:
        """转换为 imshow 使用的 (forward, right, channel) 显示布局。"""
        if bev_layout == "xy":
            return image.transpose(1, 0, 2)
        if bev_layout == "yx":
            return image
        raise ValueError(f"未知 HDMap BEV 布局: {bev_layout}")

    def _draw_hdmap_bev_background(
        self,
        ax,
        hdmap_path: Optional[str],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        alpha: float,
    ) -> bool:
        """在当前 BEV 轨迹坐标系下绘制 HDMap 底图。"""
        if not hdmap_path:
            return False

        try:
            bev = self._load_hdmap_bev_npz(hdmap_path)
            if bev is None:
                return False
            bev_layout = self._infer_hdmap_bev_layout(bev, point_cloud_range, voxel_size)
            display_image = self._to_hdmap_display_image(
                self._build_hdmap_rgba_overlay(bev, alpha),
                bev_layout,
            )
        except Exception as exc:
            self._warn_hdmap_once(str(hdmap_path), f"无法绘制 HDMap BEV ({exc})")
            return False

        right_min, forward_min = point_cloud_range[0], point_cloud_range[1]
        right_max, forward_max = point_cloud_range[3], point_cloud_range[4]
        ax.imshow(
            display_image,
            origin="lower",
            extent=(right_min, right_max, forward_min, forward_max),
            interpolation="antialiased",
            interpolation_stage="rgba",
            resample=True,
            zorder=0,
        )
        return True
    
    def get_scene_info(self, sample_idx: int) -> Optional[Tuple[str, int]]:
        """根据样本索引获取场景信息（时间戳和本地索引）"""
        for timestamp in self.scene_names:
            if timestamp in self.scene_data:
                info = self.scene_data[timestamp]
                start_idx = info['start_idx']
                num_samples = info['num_samples']
                if start_idx <= sample_idx < start_idx + num_samples:
                    local_idx = sample_idx - start_idx
                    return (timestamp, local_idx)
        return None
    
    def visualize_dual_view(
        self,
        sample_idx: int,
        save_path: Optional[str],
        bev_xlim: Optional[Tuple[float, float]] = None,
        bev_ylim: Optional[Tuple[float, float]] = None,
        show_hdmap: bool = True,
        show_objects: bool = False,
        hdmap_point_cloud_range: Tuple[float, float, float, float, float, float] = DEFAULT_HDMAP_POINT_CLOUD_RANGE,
        hdmap_voxel_size: Tuple[float, float, float] = DEFAULT_HDMAP_VOXEL_SIZE,
        hdmap_alpha: float = DEFAULT_HDMAP_ALPHA,
    ):
        """上：4路摄像头图像（front/right/left/back）；下：BEV 轨迹"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        if self.pkl_root_dir is None:
            print("错误: 未提供 pkl_root_dir，无法加载图像")
            return
        if sample_idx >= len(self.gt_trajectories):
            print(f"错误: 样本索引 {sample_idx} 超出范围")
            return

        gt_traj = self.gt_trajectories[sample_idx]
        pred_traj = self.pred_trajectories[sample_idx]
        result = self._load_images_from_pkl(sample_idx)
        if result is None:
            return
        imgs, nav_text, his_traj, hdmap_bev_path, object_annotations = result

        # 获取场景信息
        scene_info = self.get_scene_info(sample_idx)
        if scene_info:
            timestamp, local_idx = scene_info
            title_info = f"sample {sample_idx}, {timestamp}, step {local_idx}"
        else:
            title_info = f"sample {sample_idx}"
        if nav_text:
            title_info += f" | Nav: {nav_text}"

        # 布局：左侧 4 路摄像头，右侧固定物理视野 BEV；保持紧凑固定尺寸。
        fig = plt.figure(figsize=DUAL_VIEW_FIGSIZE)
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.0, 1.0, 1.95],
            hspace=0.25,
            wspace=0.12,
            # left=0.018,
            # right=0.995,
            # bottom=0.052,
            # top=0.915,
        )

        # 摄像头布局：第一排 front/back，第二排 left/right
        cam_grid = [
            ('front', 0, 0),  # row, col in left 2x2 grid
            ('back',  0, 1),
            ('left',  1, 0),
            ('right', 1, 1),
        ]
        cam_labels = {
            'front': 'Front',
            'right': 'Right',
            'left':  'Left',
            'back':  'Back',
        }

        font_cv = cv2.FONT_HERSHEY_SIMPLEX

        def _draw_nav_label(img, text):
            """在图像左上角绘制带半透明黑底的白色导航文字"""
            img = img.copy()
            label = f"Nav: {text}"
            font_scale, thickness = 0.9, 2
            (tw, th), baseline = cv2.getTextSize(label, font_cv, font_scale, thickness)
            overlay = img.copy()
            cv2.rectangle(overlay, (6, 6), (6 + tw + 8, 6 + th + baseline + 8), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, label, (10, 6 + th + 4), font_cv, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            return img

        # 左侧2x2摄像头
        for cam, row, col in cam_grid:
            ax = fig.add_subplot(gs[row, col])
            img = imgs.get(cam)
            if img is not None:
                # 仅在 front 图上叠加导航指令文字
                if cam == 'front' and nav_text:
                    img = _draw_nav_label(img, nav_text)
                ax.imshow(img)
            else:
                # 缺失摄像头：显示灰色占位
                ax.set_facecolor('#cccccc')
                ax.text(0.5, 0.5, f'{cam_labels[cam]}\n(N/A)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='#555555')
            ax.set_title(cam_labels[cam], fontsize=9, pad=2)
            ax.axis("off")

        # 右侧：BEV 轨迹（跨2行）
        ax_bev = fig.add_subplot(gs[:, 2])
        hdmap_drawn = False
        if show_hdmap:
            hdmap_drawn = self._draw_hdmap_bev_background(
                ax=ax_bev,
                hdmap_path=hdmap_bev_path,
                point_cloud_range=hdmap_point_cloud_range,
                voxel_size=hdmap_voxel_size,
                alpha=hdmap_alpha,
            )

        object_classes_drawn: Set[int] = set()
        if show_objects:
            object_classes_drawn = self._draw_object_boxes(ax_bev, object_annotations)

        # 从原点 (0,0) 开始连线到轨迹各点
        origin = np.array([[0.0, 0.0]])
        gt_traj_with_origin = np.concatenate([origin, gt_traj], axis=0)
        pred_traj_with_origin = np.concatenate([origin, pred_traj], axis=0)
        history_traj_with_origin = None
        if his_traj is not None and len(his_traj) > 0:
            his_dist = np.linalg.norm(his_traj, axis=1)
            if his_dist[0] < his_dist[-1]:
                his_traj_for_plot = his_traj[::-1]
            else:
                his_traj_for_plot = his_traj
            history_traj_with_origin = np.concatenate([his_traj_for_plot, origin], axis=0)
        else:
            his_traj_for_plot = None

        ax_bev.plot(gt_traj_with_origin[:, 1], gt_traj_with_origin[:, 0], "g-", linewidth=2, label="GT", zorder=3)
        ax_bev.plot(pred_traj_with_origin[:, 1], pred_traj_with_origin[:, 0], "r-", linewidth=2, label="Pred", zorder=3)
        if history_traj_with_origin is not None:
            ax_bev.plot(
                history_traj_with_origin[:, 1],
                history_traj_with_origin[:, 0],
                color="tab:blue",
                linestyle="--",
                linewidth=2,
                label="History",
                zorder=3,
            )
        ax_bev.scatter(gt_traj[:, 1], gt_traj[:, 0], c="green", s=30, alpha=0.7, zorder=4)
        ax_bev.scatter(pred_traj[:, 1], pred_traj[:, 0], c="red", s=30, alpha=0.7, marker="x", zorder=4)
        if his_traj is not None and len(his_traj) > 0:
            ax_bev.scatter(
                his_traj_for_plot[:, 1],
                his_traj_for_plot[:, 0],
                c="tab:blue",
                s=24,
                alpha=0.75,
                zorder=4,
            )
        ax_bev.scatter(0, 0, color="blue", s=100, marker="s", label="Ego Start", zorder=6)

        configure_bev_axes(ax_bev)

        # 默认使用固定物理视野，保证批量输出的 BEV 宽高和比例一致。
        # 如果用户传参，则只使用用户指定范围，不再按轨迹动态缩放。
        if bev_ylim:
            ax_bev.set_xlim(bev_ylim)
        else:
            ax_bev.set_xlim(DEFAULT_BEV_YLIM)

        if bev_xlim:
            ax_bev.set_ylim(bev_xlim)
        else:
            ax_bev.set_ylim(DEFAULT_BEV_XLIM)

        ax_bev.set_aspect("equal", adjustable="box")
        ax_bev.set_anchor("W")
        ax_bev.grid(not hdmap_drawn, alpha=0.3)
        legend_handles, legend_labels = ax_bev.get_legend_handles_labels()
        if hdmap_drawn:
            hdmap_handles = self._hdmap_legend_handles()
            legend_handles.extend(hdmap_handles)
            legend_labels.extend([h.get_label() for h in hdmap_handles])
        if object_classes_drawn:
            object_handles = self._object_legend_handles(object_classes_drawn)
            legend_handles.extend(object_handles)
            legend_labels.extend([h.get_label() for h in object_handles])
        if legend_handles:
            dedup_handles = []
            dedup_labels = []
            for handle, label in zip(legend_handles, legend_labels):
                if label in dedup_labels:
                    continue
                dedup_handles.append(handle)
                dedup_labels.append(label)
            ax_bev.legend(
                dedup_handles,
                dedup_labels,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                fontsize=7,
            )
        title_parts = ["BEV Trajectory"]
        if hdmap_drawn:
            title_parts.append("HDMap")
        if object_classes_drawn:
            title_parts.append("Objects")
        ax_bev.set_title(" + ".join(title_parts), fontsize=10, pad=3)

        # 在 BEV 内部展示坐标，避免额外文本列造成模块松散。
        history_lines = []
        if his_traj_for_plot is not None and len(his_traj_for_plot) > 0:
            history_lines = [
                f"{i}: ({his_traj_for_plot[i, 0]:.2f}, {his_traj_for_plot[i, 1]:.2f})"
                for i in range(len(his_traj_for_plot))
            ]
        gt_lines = [f"{i}: ({gt_traj[i, 0]:.2f}, {gt_traj[i, 1]:.2f})" for i in range(len(gt_traj))]
        pred_lines = [f"{i}: ({pred_traj[i, 0]:.2f}, {pred_traj[i, 1]:.2f})" for i in range(len(pred_traj))]
        coords_sections = []
        if history_lines:
            coords_sections.append("History:\n" + "\n".join(history_lines))
        coords_sections.append("GT:\n" + "\n".join(gt_lines))
        coords_sections.append("Pred:\n" + "\n".join(pred_lines))
        coords_text = "\n\n".join(coords_sections)
        ax_bev.text(1.05, 1.0, coords_text, ha='left', va='top',
                    transform=ax_bev.transAxes, fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(title_info, fontsize=11, y=0.98)

        if save_path:
            fig.savefig(save_path, dpi=DUAL_VIEW_DPI, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

    def visualize_dual_view_range(
        self,
        start_idx: int,
        num_samples: int,
        save_dir: Optional[str],
        bev_xlim: Optional[Tuple[float, float]],
        bev_ylim: Optional[Tuple[float, float]],
        max_scenes: Optional[int] = None,
        show_hdmap: bool = True,
        show_objects: bool = False,
        hdmap_point_cloud_range: Tuple[float, float, float, float, float, float] = DEFAULT_HDMAP_POINT_CLOUD_RANGE,
        hdmap_voxel_size: Tuple[float, float, float] = DEFAULT_HDMAP_VOXEL_SIZE,
        hdmap_alpha: float = DEFAULT_HDMAP_ALPHA,
    ):
        """可视化指定范围的样本"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        total = len(self.gt_trajectories)
        end_idx = min(start_idx + num_samples, total)
        indices = list(range(start_idx, end_idx))
        if not indices:
            print("警告: 范围为空")
            return

        save_dir_path = Path(save_dir) if save_dir else None
        if save_dir_path:
            save_dir_path.mkdir(parents=True, exist_ok=True)

        # 如果指定了最大场景数量，需要跟踪已处理的时间戳
        processed_timestamps = set()
        filtered_indices = []
        
        if max_scenes is not None:
            for idx in indices:
                scene_info = self.get_scene_info(idx)
                if scene_info:
                    timestamp, _ = scene_info
                    if timestamp not in processed_timestamps:
                        if len(processed_timestamps) >= max_scenes:
                            break
                        processed_timestamps.add(timestamp)
                    filtered_indices.append(idx)
                else:
                    filtered_indices.append(idx)
            
            if len(processed_timestamps) > 0:
                print(f"限制可视化时间戳数量为 {max_scenes}，实际处理了 {len(processed_timestamps)} 个时间戳: {sorted(list(processed_timestamps))[:10]}{'...' if len(processed_timestamps) > 10 else ''}")
        else:
            filtered_indices = indices

        if not filtered_indices:
            print("警告: 过滤后范围为空")
            return

        for idx in tqdm(filtered_indices, desc="dual visualize", unit="img"):
            # 获取场景信息，用于组织文件结构
            scene_info = self.get_scene_info(idx)
            if scene_info and save_dir_path:
                # 按时间戳组织保存路径
                timestamp, local_idx = scene_info
                scene_dir = save_dir_path / timestamp
                scene_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(scene_dir / f"dual_view_{local_idx:06d}.png")
            else:
                save_path = str(save_dir_path / f"dual_view_{idx:06d}.png") if save_dir_path else None
            
            self.visualize_dual_view(
                sample_idx=idx,
                save_path=save_path,
                bev_xlim=bev_xlim,
                bev_ylim=bev_ylim,
                show_hdmap=show_hdmap,
                show_objects=show_objects,
                hdmap_point_cloud_range=hdmap_point_cloud_range,
                hdmap_voxel_size=hdmap_voxel_size,
                hdmap_alpha=hdmap_alpha,
            )
    
    def plot_error_statistics(self, save_path: Optional[str] = None):
        """绘制误差统计图"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        
        errors = self.compute_errors()
        if errors is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各时间步的平均误差
        ax = axes[0, 0]
        time_steps = np.arange(len(errors['mean_errors_per_step']))
        ax.plot(time_steps, errors['mean_errors_per_step'], 'b-o', linewidth=2, markersize=8)
        ax.fill_between(
            time_steps,
            errors['mean_errors_per_step'] - errors['std_errors_per_step'],
            errors['mean_errors_per_step'] + errors['std_errors_per_step'],
            alpha=0.3
        )
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('L2 Error (m)', fontsize=12)
        ax.set_title('Average L2 Error per Time Step', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        ax = axes[0, 1]
        ax.hist(errors['l2_errors'].flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(errors['overall_mean'], color='r', linestyle='--', linewidth=2, label=f'Mean: {errors["overall_mean"]:.4f}')
        ax.axvline(errors['overall_median'], color='g', linestyle='--', linewidth=2, label=f'Median: {errors["overall_median"]:.4f}')
        ax.set_xlabel('L2 Error (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 最终时间步误差分布
        ax = axes[1, 0]
        ax.hist(errors['final_errors'], bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(errors['final_errors'].mean(), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {errors["final_errors"].mean():.4f}')
        ax.axvline(np.median(errors['final_errors']), color='g', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(errors["final_errors"]):.4f}')
        ax.set_xlabel('L2 Error at Final Time Step (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Final Time Step Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 误差热力图
        ax = axes[1, 1]
        max_samples = min(1000, len(errors['l2_errors']))
        im = ax.imshow(errors['l2_errors'][:max_samples].T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Time Step', fontsize=12)
        ax.set_title(f'Error Heatmap (first {max_samples} samples)', fontsize=14)
        plt.colorbar(im, ax=ax, label='L2 Error (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"误差统计图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="MOGO 数据集轨迹数据分析工具")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="轨迹 npy 文件目录路径（包含所有时间戳的 npy 文件，格式：{timestamp}_gt_trajectories.npy）"
    )
    parser.add_argument(
        "--pkl_root_dir",
        type=str,
        default=None,
        help="pkl 文件根目录，用于 dual 视图。支持两种结构：1) 嵌套结构：{pkl_root_dir}/{timestamp}/drivepi/00001.pkl 2) 扁平化结构：{pkl_root_dir}/{timestamp}_step1.pkl"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="info",
        choices=["info", "error", "dual"],
        help="运行模式: info(基本信息), error(误差分析+碰撞率), dual(左图pkl+右图BEV)"
    )
    parser.add_argument(
        "--scene_names",
        type=str,
        nargs="+",
        default=None,
        help="要加载的时间戳列表（None表示加载所有时间戳）"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="要可视化的样本索引（dual模式）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="要可视化的样本数量（dual模式，从sample_idx开始；默认全部）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（用于保存可视化结果；未指定则直接显示）"
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="statistics.json",
        help="用于归一化/反归一化的统计文件路径"
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="percentile",
        choices=["percentile", "percentile_clip", "minmax"],
        help="acc 评测使用的归一化模式，必须与训练 cfg.data.val.norm_mode 一致",
    )
    parser.add_argument(
        "--eval_thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5],
        help="多阈值准确率的阈值列表"
    )
    parser.add_argument(
        "--bev_xlim",
        type=float,
        nargs=2,
        default=None,
        help="BEV可视化的X轴范围（前后方向），默认固定为 -30 60"
    )
    parser.add_argument(
        "--bev_ylim",
        type=float,
        nargs=2,
        default=None,
        help="BEV可视化的Y轴范围（左右方向），默认固定为 -40 40"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="限制可视化的时间戳数量（仅用于dual模式，None表示不限制）"
    )
    parser.add_argument(
        "--collision_interp_dt",
        type=float,
        default=get_default_collision_interp_dt(),
        help="碰撞检测的时间插值分辨率（秒）"
    )
    parser.add_argument(
        "--no_hdmap",
        action="store_true",
        help="dual 模式下不在 BEV 轨迹图中叠加 HDMap"
    )
    parser.add_argument(
        "--show_objects",
        action="store_true",
        help="dual 模式下根据 pkl 障碍物字段在 BEV 轨迹图中绘制目标框"
    )
    parser.add_argument(
        "--hdmap_point_cloud_range",
        type=float,
        nargs=6,
        default=list(DEFAULT_HDMAP_POINT_CLOUD_RANGE),
        metavar=("RIGHT_MIN", "FORWARD_MIN", "Z_MIN", "RIGHT_MAX", "FORWARD_MAX", "Z_MAX"),
        help="HDMap BEV 对应的 point_cloud_range，默认与 tool/visualize_hdmap_npz.py 一致"
    )
    parser.add_argument(
        "--hdmap_voxel_size",
        type=float,
        nargs=3,
        default=list(DEFAULT_HDMAP_VOXEL_SIZE),
        metavar=("RIGHT_VOXEL", "FORWARD_VOXEL", "Z_VOXEL"),
        help="HDMap BEV 对应的 voxel_size，默认与 tool/visualize_hdmap_npz.py 一致"
    )
    parser.add_argument(
        "--hdmap_alpha",
        type=float,
        default=DEFAULT_HDMAP_ALPHA,
        help="HDMap 底图整体透明度缩放，取值范围 [0, 1]"
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = MOGOTrajectoryAnalyzer(
        args.trajectory_dir,
        pkl_root_dir=args.pkl_root_dir
    )
    
    # 加载数据
    try:
        analyzer.load_trajectories(scene_names=args.scene_names)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 默认 num_samples 为剩余全部
    if args.num_samples is None:
        args.num_samples = len(analyzer.gt_trajectories) - args.sample_idx
        args.num_samples = max(args.num_samples, 0)
    
    # 根据模式执行相应操作
    if args.mode == "info":
        analyzer.print_basic_info()
    
    if args.mode == "error":
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            analyzer.plot_error_statistics(save_path=str(output_dir / "error_statistics.png"))
        else:
            analyzer.plot_error_statistics()

        analyzer.compute_eval_style_metrics(
            statistic_path=args.statistic_path,
            eval_thresholds=args.eval_thresholds,
            norm_mode=args.norm_mode,
        )
        if args.pkl_root_dir is None:
            print("警告: 未提供 pkl_root_dir，跳过碰撞率分析")
        else:
            collision_metrics = analyzer.compute_collision_metrics(
                interpolation_dt=args.collision_interp_dt,
            )
            if args.output_dir and collision_metrics:
                collision_json_path = output_dir / "collision_metrics.json"
                collision_txt_path = output_dir / "collision_report.txt"
                collision_vis_dir = output_dir / "collision_visualizations"
                with open(collision_json_path, "w", encoding="utf-8") as f:
                    json.dump(collision_metrics, f, ensure_ascii=False, indent=2)
                with open(collision_txt_path, "w", encoding="utf-8") as f:
                    f.write("MOGO Collision Metrics\n")
                    f.write(f"num_valid_samples: {collision_metrics['num_valid_samples']}\n")
                    f.write(f"current_collision_rate: {collision_metrics['current_collision_rate']:.6f}\n")
                    f.write(f"future_collision_rate: {collision_metrics['future_collision_rate']:.6f}\n")
                    f.write(f"overall_collision_rate: {collision_metrics['overall_collision_rate']:.6f}\n")
                    f.write(f"gt_future_collision_rate: {collision_metrics['gt_future_collision_rate']:.6f}\n")
                    f.write("collision_rate_per_step:\n")
                    for t_sec, rate in zip(
                        collision_metrics["dense_future_times_sec"][1:],
                        collision_metrics["collision_rate_per_step"][1:],
                    ):
                        f.write(f"  t={t_sec:.1f}s: {rate:.6f}\n")
                    f.write("class_collision_counter:\n")
                    for cls_name, count in sorted(
                        collision_metrics["class_collision_counter"].items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    ):
                        f.write(f"  {cls_name}: {count}\n")
                analyzer.export_collision_class_table(
                    collision_metrics=collision_metrics,
                    output_dir=str(output_dir),
                )
                analyzer.export_collision_visualizations(
                    collision_metrics=collision_metrics,
                    output_dir=str(collision_vis_dir),
                    interpolation_dt=args.collision_interp_dt,
                    max_samples=20,
                    collision_type="pred",
                )
                analyzer.export_collision_visualizations(
                    collision_metrics=collision_metrics,
                    output_dir=str(collision_vis_dir),
                    interpolation_dt=args.collision_interp_dt,
                    max_samples=20,
                    collision_type="gt",
                )
                print(f"碰撞指标 JSON 已保存到: {collision_json_path}")
                print(f"碰撞指标文本报告已保存到: {collision_txt_path}")

    if args.mode == "dual":
        if args.pkl_root_dir is None:
            print("错误: dual 模式需要提供 --pkl_root_dir")
            return
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.trajectory_dir) / "visualizations_dual"
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer.visualize_dual_view_range(
            start_idx=args.sample_idx,
            num_samples=args.num_samples,
            save_dir=str(output_dir),
            bev_xlim=tuple(args.bev_xlim) if args.bev_xlim else None,
            bev_ylim=tuple(args.bev_ylim) if args.bev_ylim else None,
            max_scenes=args.num_scenes,
            show_hdmap=not args.no_hdmap,
            show_objects=args.show_objects,
            hdmap_point_cloud_range=tuple(args.hdmap_point_cloud_range),
            hdmap_voxel_size=tuple(args.hdmap_voxel_size),
            hdmap_alpha=args.hdmap_alpha,
        )


if __name__ == "__main__":
    main()
