"""
轨迹数据分析工具

用于加载、展示和分析保存的轨迹npy文件
支持两种格式：
1. 单个文件格式：gt_trajectories.npy 和 pred_trajectories.npy（向后兼容）
2. 按场景格式：{scene_name}_gt_trajectories.npy 和 {scene_name}_pred_trajectories.npy

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


class TrajectoryAnalyzer:
    """轨迹分析器"""
    
    def __init__(self, trajectory_dir: str, pkl_dir: Optional[str] = None):
        """
        初始化轨迹分析器
        
        Args:
            trajectory_dir: 轨迹文件目录路径
                - 单个文件格式：包含 gt_trajectories.npy 和 pred_trajectories.npy
                - 按场景格式：包含多个 {scene_name}_gt_trajectories.npy 和 {scene_name}_pred_trajectories.npy
            pkl_dir: pkl文件目录路径（用于dual视图）
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.gt_trajectories = None
        self.pred_trajectories = None
        self.pkl_dir = Path(pkl_dir) if pkl_dir else None
        
        # 场景相关数据
        self.scene_data: Dict[str, Dict] = {}  # {scene_name: {'gt': array, 'pred': array, 'start_idx': int}}
        self.scene_names: List[str] = []
        self.load_mode: str = "unknown"  # "single" 或 "multi_scene"
        
        # pkl文件映射：用于dual视图时根据sample_idx找到对应的pkl文件
        self.pkl_file_map: Dict[int, Path] = {}
        if self.pkl_dir and self.pkl_dir.exists():
            self._build_pkl_file_map()
    
    def _build_pkl_file_map(self):
        """构建pkl文件映射：sample_idx -> pkl_path"""
        if self.pkl_dir is None:
            return
        
        # 按场景组织pkl文件
        scene_pkl_map: Dict[str, List[Tuple[int, Path]]] = {}
        for pkl_file in self.pkl_dir.glob("*.pkl"):
            # 提取场景名和step编号
            match = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
            if match:
                scene_name = match.group(1)
                step_num = int(match.group(2))
                if scene_name not in scene_pkl_map:
                    scene_pkl_map[scene_name] = []
                scene_pkl_map[scene_name].append((step_num, pkl_file))
        
        # 按场景和step排序，建立全局索引
        sample_idx = 0
        for scene_name in sorted(scene_pkl_map.keys()):
            scene_files = sorted(scene_pkl_map[scene_name], key=lambda x: x[0])
            for step_num, pkl_file in scene_files:
                self.pkl_file_map[sample_idx] = pkl_file
                sample_idx += 1
    
    def load_trajectories(self, scene_names: Optional[List[str]] = None):
        """
        加载轨迹数据
        
        Args:
            scene_names: 要加载的场景列表（None表示加载所有场景或单个文件）
        """
        # 检查是单个文件格式还是按场景格式
        single_gt_path = self.trajectory_dir / "gt_trajectories.npy"
        single_pred_path = self.trajectory_dir / "pred_trajectories.npy"
        
        if single_gt_path.exists() and single_pred_path.exists():
            # 单个文件格式（向后兼容）
            print(f"检测到单个文件格式，正在加载...")
            self.load_mode = "single"
            self.gt_trajectories = np.load(single_gt_path)
            self.pred_trajectories = np.load(single_pred_path)
            
            print(f"✓ 加载完成")
            print(f"  GT轨迹形状: {self.gt_trajectories.shape}")
            print(f"  预测轨迹形状: {self.pred_trajectories.shape}")
            
            # 检查数据一致性
            if len(self.gt_trajectories) != len(self.pred_trajectories):
                raise ValueError(
                    f"GT轨迹数量 ({len(self.gt_trajectories)}) 与预测轨迹数量 "
                    f"({len(self.pred_trajectories)}) 不匹配"
                )
            
            if self.gt_trajectories.shape[1:] != self.pred_trajectories.shape[1:]:
                raise ValueError(
                    f"GT轨迹形状 {self.gt_trajectories.shape[1:]} 与预测轨迹形状 "
                    f"{self.pred_trajectories.shape[1:]} 不匹配"
                )
        else:
            # 按场景格式
            print(f"检测到按场景格式，正在加载...")
            self.load_mode = "multi_scene"
            
            # 找到所有场景文件
            all_scene_names = set()
            for file in self.trajectory_dir.glob("*_gt_trajectories.npy"):
                scene_name = file.name.replace("_gt_trajectories.npy", "")
                all_scene_names.add(scene_name)
            
            # 确定要加载的场景
            if scene_names is None:
                scene_names = sorted(all_scene_names)
            else:
                # 验证场景是否存在
                missing_scenes = [s for s in scene_names if s not in all_scene_names]
                if missing_scenes:
                    raise FileNotFoundError(f"以下场景文件不存在: {missing_scenes}")
            
            self.scene_names = scene_names
            
            # 加载每个场景的数据
            all_gt_trajectories = []
            all_pred_trajectories = []
            current_idx = 0
            
            for scene_name in tqdm(scene_names, desc="Loading scenes"):
                gt_path = self.trajectory_dir / f"{scene_name}_gt_trajectories.npy"
                pred_path = self.trajectory_dir / f"{scene_name}_pred_trajectories.npy"
                
                if not gt_path.exists():
                    print(f"警告: GT文件不存在，跳过场景 {scene_name}: {gt_path}")
                    continue
                if not pred_path.exists():
                    print(f"警告: 预测文件不存在，跳过场景 {scene_name}: {pred_path}")
                    continue
                
                scene_gt = np.load(gt_path)  # [N, horizon, 2]
                scene_pred = np.load(pred_path)  # [N, horizon, 2]
                
                # 检查一致性
                if len(scene_gt) != len(scene_pred):
                    print(f"警告: 场景 {scene_name} 的GT和预测数量不匹配，跳过")
                    continue
                
                if scene_gt.shape[1:] != scene_pred.shape[1:]:
                    print(f"警告: 场景 {scene_name} 的GT和预测形状不匹配，跳过")
                    continue
                
                # 存储场景数据
                self.scene_data[scene_name] = {
                    'gt': scene_gt,
                    'pred': scene_pred,
                    'start_idx': current_idx,
                    'num_samples': len(scene_gt)
                }
                
                all_gt_trajectories.append(scene_gt)
                all_pred_trajectories.append(scene_pred)
                current_idx += len(scene_gt)
            
            if len(all_gt_trajectories) == 0:
                raise ValueError("没有找到有效的场景轨迹文件")
            
            # 合并所有场景
            self.gt_trajectories = np.concatenate(all_gt_trajectories, axis=0)
            self.pred_trajectories = np.concatenate(all_pred_trajectories, axis=0)
            
            print(f"✓ 加载完成")
            print(f"  加载了 {len(self.scene_names)} 个场景")
            print(f"  总样本数: {len(self.gt_trajectories)}")
            print(f"  轨迹形状: {self.gt_trajectories.shape}")
            
            # 打印每个场景的信息
            for scene_name in self.scene_names:
                if scene_name in self.scene_data:
                    info = self.scene_data[scene_name]
                    print(f"    - {scene_name}: {info['num_samples']} 个样本 (索引 {info['start_idx']}-{info['start_idx']+info['num_samples']-1})")
    
    def print_basic_info(self):
        """打印基本信息"""
        if self.gt_trajectories is None or self.pred_trajectories is None:
            print("错误: 请先加载轨迹数据")
            return
        
        print("\n" + "="*60)
        print("轨迹数据基本信息")
        print("="*60)
        
        num_samples, horizon, dim = self.gt_trajectories.shape
        
        print(f"\n加载模式: {self.load_mode}")
        if self.load_mode == "multi_scene":
            print(f"场景数量: {len(self.scene_names)}")
            print(f"场景列表: {', '.join(self.scene_names[:5])}{'...' if len(self.scene_names) > 5 else ''}")
        
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
                scene_name, local_idx = scene_info
                print(f"  样本 {i} ({scene_name}, step {local_idx}):")
            else:
                print(f"  样本 {i}:")
            for t in range(min(5, horizon)):
                x, y = self.gt_trajectories[i, t]
                print(f"    t={t}: ({x:.3f}, {y:.3f})")
        
        print("="*60)
    
    def compute_errors(self):
        """计算误差指标"""
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
        
        # 计算最终时间步的误差（通常更重要）
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
    

    # --------- 从pkl取图 + BEV 组合可视化 ----------
    def _load_front_image_from_pkl(self, sample_idx: int) -> Optional[np.ndarray]:
        """从pkl加载前视图像"""
        if self.pkl_dir is None:
            print("错误: 未指定pkl文件目录")
            return None
        
        # 从映射中找到对应的pkl文件
        if sample_idx not in self.pkl_file_map:
            print(f"错误: 样本索引 {sample_idx} 没有对应的pkl文件")
            return None
        
        pkl_path = self.pkl_file_map[sample_idx]
        try:
            import pickle
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            img_ref = data.get("his_cam_front", None)
            if img_ref is None:
                print(f"错误: pkl缺少 his_cam_front 字段: {pkl_path}")
                return None
            if hasattr(img_ref, "numpy"):
                img_ref = img_ref.numpy()
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
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"加载pkl失败 {pkl_path}: {e}")
            return None
    
    def get_scene_info(self, sample_idx: int) -> Optional[Tuple[str, int]]:
        """
        根据样本索引获取场景信息
        
        Returns:
            (scene_name, local_idx) 或 None
        """
        if self.load_mode == "single":
            return None
        
        for scene_name in self.scene_names:
            if scene_name in self.scene_data:
                info = self.scene_data[scene_name]
                start_idx = info['start_idx']
                num_samples = info['num_samples']
                if start_idx <= sample_idx < start_idx + num_samples:
                    local_idx = sample_idx - start_idx
                    return (scene_name, local_idx)
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
        if self.pkl_dir is None:
            print("错误: 未提供 pkl_dir，无法加载前视图像")
            return
        if sample_idx >= len(self.gt_trajectories):
            print(f"错误: 样本索引 {sample_idx} 超出范围")
            return

        gt_traj = self.gt_trajectories[sample_idx]
        pred_traj = self.pred_trajectories[sample_idx]
        # 逆时针旋转90度 (x, y) -> (-y, x)
        rotation = np.array([[0, -1], [1, 0]])
        gt_traj = gt_traj @ rotation.T
        pred_traj = pred_traj @ rotation.T
        img_rgb = self._load_front_image_from_pkl(sample_idx)
        if img_rgb is None:
            return

        # 获取场景信息
        scene_info = self.get_scene_info(sample_idx)
        if scene_info:
            scene_name, local_idx = scene_info
            front_title = f"Front image (sample {sample_idx}, {scene_name}, step {local_idx})"
        else:
            front_title = f"Front image (sample {sample_idx})"

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig, axes = plt.subplots(
            1, 2,
            figsize=(16, 6),              # 整体画布放大
            gridspec_kw={'width_ratios': [1.6, 1.0]}  # 左图更宽
        )

        # 左：前视图（不再叠加轨迹）
        axes[0].imshow(img_rgb)
        axes[0].set_title(front_title)
        axes[0].axis("off")

        # 右：BEV 轨迹（matplotlib）
        axes[1].plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2, label="GT")
        axes[1].plot(pred_traj[:, 0], pred_traj[:, 1], "r-", linewidth=2, label="Pred")
        
        # 添加轨迹点标记
        axes[1].scatter(gt_traj[:, 0], gt_traj[:, 1], c='green', s=30, alpha=0.7, label='GT Points')
        axes[1].scatter(pred_traj[:, 0], pred_traj[:, 1], c='red', s=30, alpha=0.7, marker='x', label='Pred Points')
        
        # 标记起始点
        axes[1].scatter(gt_traj[0, 0], gt_traj[0, 1], color='green', s=100, marker='s', label='GT Start', zorder=5)
        axes[1].scatter(pred_traj[0, 0], pred_traj[0, 1], color='red', s=100, marker='s', label='Pred Start', zorder=5)
        
        axes[1].set_xlabel("Y (left positive)")
        axes[1].set_ylabel("X (up positive)")
        # axes[1].set_xlabel("X")
        # axes[1].set_ylabel("Y")

        # 计算轨迹范围用于自动范围或兜底
        all_pts = np.concatenate([gt_traj, pred_traj], axis=0)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        margin_x = max(1.0, 0.1 * (x_max - x_min + 1e-6))
        margin_y = max(1.0, 0.1 * (y_max - y_min + 1e-6))

        # 如果用户指定范围，先尝试使用；若所有点都在范围外，则改用数据自动范围
        def _in_range(lim, data_min, data_max):
            return not ((data_max < lim[0]) or (data_min > lim[1]))

        if bev_xlim and _in_range(bev_xlim, x_min, x_max):
            axes[1].set_xlim(bev_xlim)
        else:
            axes[1].set_xlim(x_min - margin_x, x_max + margin_x)

        # 反转 X 轴方向：左正右负
        axes[1].invert_xaxis()

        if bev_ylim and _in_range(bev_ylim, y_min, y_max):
            axes[1].set_ylim(bev_ylim)
        else:
            axes[1].set_ylim(y_min - margin_y, y_max + margin_y)
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
    ):
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

        for idx in tqdm(indices, desc="dual visualize", unit="img"):
            # 获取场景信息，用于组织文件结构
            scene_info = self.get_scene_info(idx)
            if scene_info and save_dir_path:
                # 多场景模式：保存到场景文件夹中
                scene_name, local_idx = scene_info
                scene_dir = save_dir_path / scene_name
                scene_dir.mkdir(parents=True, exist_ok=True)
                # 使用本地索引而不是全局索引
                save_path = str(scene_dir / f"dual_view_{local_idx:06d}.png")
            else:
                # 单文件模式或未指定保存目录：使用全局索引
                save_path = str(save_dir_path / f"dual_view_{idx:06d}.png") if save_dir_path else None
            
            self.visualize_dual_view(
                sample_idx=idx,
                save_path=save_path,
                bev_xlim=bev_xlim,
                bev_ylim=bev_ylim,
            )
    
    
    def plot_error_statistics(self, save_path: Optional[str] = None):
        """
        绘制误差统计图
        
        Args:
            save_path: 保存路径（如果为None则显示）
        """
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
        
        # 4. 误差热力图（样本 x 时间步）
        ax = axes[1, 1]
        # 为了可视化，只显示前1000个样本（如果太多）
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
    parser = argparse.ArgumentParser(description="轨迹数据分析工具")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="轨迹文件目录路径（支持单个文件格式或按场景格式）"
    )
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default=None,
        help="pkl文件目录（dual视图左侧使用）"
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
        help="要加载的场景列表（仅用于按场景格式，None表示加载所有场景）"
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
        help="输出目录（用于保存可视化结果）"
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="是否保存图表（而不是显示）"
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
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = TrajectoryAnalyzer(
        args.trajectory_dir,
        pkl_dir=args.pkl_dir
    )
    
    # 加载数据
    try:
        analyzer.load_trajectories(scene_names=args.scene_names)
    except Exception as e:
        print(f"错误: {e}")
        return

    # 默认 num_samples 为剩余全部
    if args.num_samples is None:
        args.num_samples = len(analyzer.gt_trajectories) - args.sample_idx
        args.num_samples = max(args.num_samples, 0)
    
    # 根据模式执行相应操作
    if args.mode == "info":
        analyzer.print_basic_info()
    
    if args.mode == "error":
        analyzer.compute_errors()
        if args.save_plots or args.output_dir:
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.trajectory_dir) / "analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            analyzer.plot_error_statistics(save_path=str(output_dir / "error_statistics.png"))
        else:
            analyzer.plot_error_statistics()

    if args.mode == "dual":
        if args.pkl_dir is None:
            print("错误: dual 模式需要提供 --pkl_dir")
            return
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.trajectory_dir) / "visualizations_dual"
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer.visualize_dual_view_range(
            start_idx=args.sample_idx,
            num_samples=args.num_samples,
            save_dir=str(output_dir),
            bev_xlim=tuple(args.bev_xlim) if args.bev_xlim else None,
            bev_ylim=tuple(args.bev_ylim) if args.bev_ylim else None,
        )


if __name__ == "__main__":
    main()

