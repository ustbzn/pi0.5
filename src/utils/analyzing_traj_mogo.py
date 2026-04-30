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
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import cv2
from tqdm import tqdm
import os
import re
import torch

from src.utils.metric import get_action_accuracy
from src.data.utils.normalization import Normalize

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
                # 尝试匹配时间戳_stepN格式：20251210141637_step1.pkl
                match_timestamp = re.match(r'(\d+)_step(\d+)\.pkl$', pkl_file.name)
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
                    # 首先尝试匹配时间戳_step格式
                    match_timestamp = re.match(r'(\d+)_step(\d+)\.pkl$', pkl_file.name)
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
                    m = re.match(r'(\d+)_step(\d+)\.pkl$', pkl_file.name)
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
                        m = re.match(r'(\d+)_step(\d+)\.pkl$', pkl_file.name)
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
    ) -> Dict[str, object]:
        """按照 eval_old.py 的风格计算评估指标"""
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
        normalize = Normalize.get_instance(statistic_path)
        
        gt_x = self.gt_trajectories[:, :, 0]
        gt_y = self.gt_trajectories[:, :, 1]
        pred_x = self.pred_trajectories[:, :, 0]
        pred_y = self.pred_trajectories[:, :, 1]
        
        gt_x_norm = normalize.standard_normalize(gt_x, normalize.fur_x_01, normalize.fur_x_99)
        gt_y_norm = normalize.standard_normalize(gt_y, normalize.fur_y_01, normalize.fur_y_99)
        pred_x_norm = normalize.standard_normalize(pred_x, normalize.fur_x_01, normalize.fur_x_99)
        pred_y_norm = normalize.standard_normalize(pred_y, normalize.fur_y_01, normalize.fur_y_99)
        
        gt_norm = np.stack([gt_x_norm, gt_y_norm], axis=-1)
        pred_norm = np.stack([pred_x_norm, pred_y_norm], axis=-1)
        
        gt_tensor = torch.from_numpy(gt_norm).float()
        pred_tensor = torch.from_numpy(pred_norm).float()
        
        acc_tensor = get_action_accuracy(gt_tensor, pred_tensor, thresholds=eval_thresholds)
        accuracies = acc_tensor.cpu().numpy()
        
        print("\n归一化空间多阈值准确率（与 eval_old.py 一致）：")
        for t, acc in zip(eval_thresholds, accuracies):
            print(f"  acc thres {t}: {acc:.4f}")
        
        print("=" * 60)
        
        return {
            "l1_loss": l1_error,
            "l2_loss": l2_error,
            "thresholds": eval_thresholds,
            "accuracies": accuracies,
        }
    
    def _load_images_from_pkl(
        self, sample_idx: int
    ) -> Optional[Tuple[dict, Optional[str], Optional[np.ndarray]]]:
        """从pkl加载4路摄像头图像（当前帧）、导航指令文本和历史轨迹

        Returns:
            (imgs, nav_text)，其中 imgs 是 {'front': ndarray, 'right': ndarray,
            'left': ndarray, 'back': ndarray}（RGB格式，缺失的摄像头为 None），
            nav_text 为映射前的导航指令文本（如 "straight"），his_traj 为
            历史轨迹，坐标格式与 gt/pred 保持一致，均为 [x, y]。
            加载完全失败时返回 None。
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

            return imgs, nav_text, his_traj
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
        imgs, nav_text, _ = result
        return imgs['front'], nav_text
    
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
        imgs, nav_text, his_traj = result

        # 获取场景信息
        scene_info = self.get_scene_info(sample_idx)
        if scene_info:
            timestamp, local_idx = scene_info
            title_info = f"sample {sample_idx}, {timestamp}, step {local_idx}"
        else:
            title_info = f"sample {sample_idx}"
        if nav_text:
            title_info += f" | Nav: {nav_text}"

        # ----------------------------------------------------------------
        # 布局：左边4个摄像头（2行2列），右边BEV轨迹
        #   左侧 (col 0-1): 2行2列摄像头
        #     第0行: front(col0), back(col1)
        #     第1行: left(col2), right(col3)  <- 注意col索引在gs中
        #   右侧 (col 2): BEV轨迹（跨2行）
        # ----------------------------------------------------------------
        # GridSpec: 2行 x 3列，左2列放摄像头，右1列放BEV
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.0, 1.0, 1.2],
            hspace=0.25,
            wspace=0.12,
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
            ax.set_title(cam_labels[cam], fontsize=11)
            ax.axis("off")

        # 右侧：BEV 轨迹（跨2行）
        ax_bev = fig.add_subplot(gs[:, 2])

        # 从原点 (0,0) 开始连线到轨迹各点
        origin = np.array([[0.0, 0.0]])
        gt_traj_with_origin = np.concatenate([origin, gt_traj], axis=0)
        pred_traj_with_origin = np.concatenate([origin, pred_traj], axis=0)
        history_traj_with_origin = None
        if his_traj is not None and len(his_traj) > 0:
            history_traj_with_origin = np.concatenate([his_traj, origin], axis=0)

        ax_bev.plot(gt_traj_with_origin[:, 1], gt_traj_with_origin[:, 0], "g-", linewidth=2, label="GT")
        ax_bev.plot(pred_traj_with_origin[:, 1], pred_traj_with_origin[:, 0], "r-", linewidth=2, label="Pred")
        if history_traj_with_origin is not None:
            ax_bev.plot(
                history_traj_with_origin[:, 1],
                history_traj_with_origin[:, 0],
                color="tab:blue",
                linestyle="--",
                linewidth=2,
                label="History",
            )
        ax_bev.scatter(gt_traj[:, 1], gt_traj[:, 0], c='green', s=30, alpha=0.7, label='GT Points')
        ax_bev.scatter(pred_traj[:, 1], pred_traj[:, 0], c='red', s=30, alpha=0.7, marker='x', label='Pred Points')
        if his_traj is not None and len(his_traj) > 0:
            ax_bev.scatter(
                his_traj[:, 1],
                his_traj[:, 0],
                c='tab:blue',
                s=24,
                alpha=0.75,
                label='History Points',
            )
        ax_bev.scatter(0, 0, color='blue', s=100, marker='s', label='Start Point', zorder=5)

        ax_bev.set_xlabel("Y (right positive)")
        ax_bev.set_ylabel("X (up positive)")

        all_pts_list = [gt_traj, pred_traj, origin]
        if his_traj is not None and len(his_traj) > 0:
            all_pts_list.append(his_traj)
        all_pts = np.concatenate(all_pts_list, axis=0)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        margin_x = max(1.0, 0.1 * (x_max - x_min + 1e-6))
        margin_y = max(1.0, 0.1 * (y_max - y_min + 1e-6))

        auto_xlim = (y_min - margin_y, y_max + margin_y)
        auto_ylim = (x_min - margin_x, x_max + margin_x)

        # 如果用户传了可视化范围，优先保留该视野，但必要时自动向外扩展，
        # 避免历史轨迹落在原点后方时被固定窗口裁掉。
        if bev_ylim:
            xlim = (min(bev_ylim[0], auto_xlim[0]), max(bev_ylim[1], auto_xlim[1]))
            ax_bev.set_xlim(xlim)
        else:
            ax_bev.set_xlim(auto_xlim)

        if bev_xlim:
            ylim = (min(bev_xlim[0], auto_ylim[0]), max(bev_xlim[1], auto_ylim[1]))
            ax_bev.set_ylim(ylim)
        else:
            ax_bev.set_ylim(auto_ylim)

        ax_bev.set_aspect("equal")
        ax_bev.grid(True, alpha=0.3)
        ax_bev.legend(loc='upper left', fontsize=9)
        ax_bev.set_title("BEV Trajectory")

        # 在 BEV 图右侧展示轨迹坐标数值（文本注释）
        history_lines = []
        if his_traj is not None and len(his_traj) > 0:
            history_lines = [
                f"{i}: ({his_traj[i, 0]:.2f}, {his_traj[i, 1]:.2f})"
                for i in range(len(his_traj))
            ]
        gt_lines = [f"{i}: ({gt_traj[i, 0]:.2f}, {gt_traj[i, 1]:.2f})" for i in range(len(gt_traj))]
        pred_lines = [f"{i}: ({pred_traj[i, 0]:.2f}, {pred_traj[i, 1]:.2f})" for i in range(len(pred_traj))]
        coords_sections = []
        if history_lines:
            coords_sections.append("History:\n" + "\n".join(history_lines))
        coords_sections.append("GT:\n" + "\n".join(gt_lines))
        coords_sections.append("Pred:\n" + "\n".join(pred_lines))
        coords_text = "\n\n".join(coords_sections)
        ax_bev.text(1.02, 1.0, coords_text, ha='left', va='top',
                    transform=ax_bev.transAxes, fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(title_info, fontsize=11, y=0.98)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
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
        help="运行模式: info(基本信息), error(误差分析), dual(左图pkl+右图BEV)"
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
        help="BEV可视化的X轴范围，如: --bev_xlim -10 60"
    )
    parser.add_argument(
        "--bev_ylim",
        type=float,
        nargs=2,
        default=None,
        help="BEV可视化的Y轴范围，如: --bev_ylim -20 20"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="限制可视化的时间戳数量（仅用于dual模式，None表示不限制）"
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
        )

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
        )


if __name__ == "__main__":
    main()
