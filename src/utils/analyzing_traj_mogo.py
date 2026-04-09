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
    
    def _load_front_image_from_pkl(self, sample_idx: int) -> Optional[Tuple[np.ndarray, Optional[str]]]:
        """从pkl加载前视图像和导航指令文本

        Returns:
            (img_rgb, nav_text) 元组，img_rgb 为前视图像，nav_text 为映射前的导航指令文本（如 "straight"）。
            加载失败时返回 None。
        """
        if self.pkl_root_dir is None:
            print("错误: 未指定pkl文件根目录")
            return None

        # 从映射中找到对应的pkl文件
        if sample_idx not in self.pkl_file_map:
            print(f"错误: 样本索引 {sample_idx} 没有对应的pkl文件")
            return None

        timestamp, pkl_path = self.pkl_file_map[sample_idx]
        try:
            import pickle
            # 兼容 Python 2.7 生成的 pickle 文件
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding='latin-1')

            img_ref = data.get("his_cam_front", None)
            if img_ref is None:
                print(f"错误: pkl缺少 his_cam_front 字段: {pkl_path}")
                return None

            # 兼容不同的数据格式（与 inference_mogo.py 保持一致）
            # 1) 处理 dict 类型（StringWrapper 序列化为 dict 的情况）
            if isinstance(img_ref, dict) and "_string_wrapper_marker" in img_ref:
                img_ref = img_ref.get("_bytes_value", b"")
            # 2) 处理有 .numpy() 方法的对象
            if hasattr(img_ref, "numpy"):
                img_ref = img_ref.numpy()
            # 3) 处理 bytes 或 str
            if isinstance(img_ref, bytes):
                img_path = img_ref.decode("utf-8")
            elif isinstance(img_ref, str):
                img_path = img_ref
            else:
                print(f"错误: his_cam_front 类型不支持: {type(img_ref)}")
                return None

            if not Path(img_path).exists():
                print(f"错误: 图像文件不存在: {img_path}")
                return None

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"错误: 无法读取图像: {img_path}")
                return None

            # 读取 navigation_id 并反向映射为文本
            nav_text = None
            nav_id_arr = data.get("navigation_id", None)
            if nav_id_arr is not None:
                if hasattr(nav_id_arr, '__len__') and len(nav_id_arr) > 0:
                    nav_id = int(nav_id_arr[0])
                else:
                    nav_id = int(nav_id_arr)
                nav_text = NAVIGATION_ID_TO_TEXT.get(nav_id, f"unknown({nav_id})")

            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), nav_text
        except Exception as e:
            print(f"加载pkl失败 {pkl_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
        """左：pkl前视图像；右：BEV 轨迹"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        if self.pkl_root_dir is None:
            print("错误: 未提供 pkl_root_dir，无法加载前视图像")
            return
        if sample_idx >= len(self.gt_trajectories):
            print(f"错误: 样本索引 {sample_idx} 超出范围")
            return

        gt_traj = self.gt_trajectories[sample_idx]
        pred_traj = self.pred_trajectories[sample_idx]
        result = self._load_front_image_from_pkl(sample_idx)
        if result is None:
            return
        img_rgb, nav_text = result

        # 获取场景信息
        scene_info = self.get_scene_info(sample_idx)
        if scene_info:
            timestamp, local_idx = scene_info
            front_title = f"Front image (sample {sample_idx}, {timestamp}, step {local_idx})"
        else:
            front_title = f"Front image (sample {sample_idx})"

        # 将导航指令文字叠加到前视图像上（左上角，醒目显示）
        if nav_text:
            front_title += f" | Nav: {nav_text}"
            img_with_nav = img_rgb.copy()
            # 使用 OpenCV 在图像上绘制带背景的导航指令文字
            label = f"Nav: {nav_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            # 绘制半透明背景矩形
            overlay = img_with_nav.copy()
            cv2.rectangle(overlay, (10, 10), (10 + tw + 10, 10 + th + baseline + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_with_nav, 0.4, 0, img_with_nav)
            # 绘制白色文字
            cv2.putText(img_with_nav, label, (15, 10 + th + 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            img_rgb = img_with_nav

        fig, axes = plt.subplots(
            1, 2,
            figsize=(16, 6),
            gridspec_kw={'width_ratios': [1.6, 1.0]}
        )

        # 左：前视图
        axes[0].imshow(img_rgb)
        axes[0].set_title(front_title)
        axes[0].axis("off")

        # 右：BEV 轨迹
        # 数值上：gt_traj[:, 0] / pred_traj[:, 0] 是原始 fur_x，gt_traj[:, 1] / pred_traj[:, 1] 是原始 fur_y
        # 可视化约定：
        #   - 纵轴：x，向上为正（前方）
        #   - 横轴：y，向右为正（右侧，往左 y 变小）
        # 因此在图中使用 (y, x) 作为 (横, 纵)
        axes[1].plot(gt_traj[:, 1], gt_traj[:, 0], "g-", linewidth=2, label="GT")
        axes[1].plot(pred_traj[:, 1], pred_traj[:, 0], "r-", linewidth=2, label="Pred")
        axes[1].scatter(gt_traj[:, 1], gt_traj[:, 0], c='green', s=30, alpha=0.7, label='GT Points')
        axes[1].scatter(pred_traj[:, 1], pred_traj[:, 0], c='red', s=30, alpha=0.7, marker='x', label='Pred Points')
        axes[1].scatter(gt_traj[0, 1], gt_traj[0, 0], color='green', s=100, marker='s', label='GT Start', zorder=5)
        axes[1].scatter(pred_traj[0, 1], pred_traj[0, 0], color='red', s=100, marker='s', label='Pred Start', zorder=5)
        
        # 坐标轴语义：
        #   - 横轴：Y（右正左负，往左 y 变小）
        #   - 纵轴：X（前正后负）
        axes[1].set_xlabel("Y (right positive)")
        axes[1].set_ylabel("X (up positive)")

        # 计算轨迹范围用于自动范围或兜底（仍基于原始 (x, y) 数值）
        all_pts = np.concatenate([gt_traj, pred_traj], axis=0)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()  # 用于纵轴
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()  # 用于横轴
        margin_x = max(1.0, 0.1 * (x_max - x_min + 1e-6))
        margin_y = max(1.0, 0.1 * (y_max - y_min + 1e-6))

        def _in_range(lim, data_min, data_max):
            return not ((data_max < lim[0]) or (data_min > lim[1]))

        # bev_xlim / bev_ylim 依然使用 (x, y) 的物理含义：
        #   bev_xlim 作用在纵轴 X，bev_ylim 作用在横轴 Y
        if bev_ylim and _in_range(bev_ylim, y_min, y_max):
            # 横轴范围：Y
            axes[1].set_xlim(bev_ylim)
        else:
            axes[1].set_xlim(y_min - margin_y, y_max + margin_y)

        if bev_xlim and _in_range(bev_xlim, x_min, x_max):
            # 纵轴范围：X
            axes[1].set_ylim(bev_xlim)
        else:
            axes[1].set_ylim(x_min - margin_x, x_max + margin_x)

        # 横轴方向：向右 y 变大，向左 y 变小（标准方向，无需反转）
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_title("BEV Trajectory")

        # 在整张画布右侧展示轨迹坐标
        gt_lines = [f"{i}: ({gt_traj[i, 0]:.2f}, {gt_traj[i, 1]:.2f})" for i in range(len(gt_traj))]
        pred_lines = [f"{i}: ({pred_traj[i, 0]:.2f}, {pred_traj[i, 1]:.2f})" for i in range(len(pred_traj))]
        gt_text = "GT points:\n" + "\n".join(gt_lines)
        pred_text = "Pred points:\n" + "\n".join(pred_lines)
        fig.text(
            0.93, 0.35,
            f"{gt_text}\n\n{pred_text}",
            ha="left", va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
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

