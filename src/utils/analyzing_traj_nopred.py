"""
轨迹数据可视化工具

用于从 pkl 文件加载并可视化原始轨迹数据
支持：
1. 可视化指定场景的所有轨迹（带前视图像）
2. 可视化所有场景的轨迹

# 可视化指定场景的所有轨迹（带前视图像）
python src/utils/analyzing_traj_nopred.py \
    --pkl_dir /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-1230/train \
    --scene_name scene-0001 \
    --output_dir ./visualizations/dual_nopred

# 可视化所有场景的轨迹
python src/utils/analyzing_traj_nopred.py \
    --pkl_dir /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-1230/train \
    --output_dir ./visualizations/dual_nopred

# 设置 BEV 坐标轴范围
python src/utils/analyzing_traj_nopred.py \
    --pkl_dir /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-1230/val \
    --scene_name scene-0016 \
    --bev_xlim -10 60 \
    --bev_ylim -20 20 \
    --output_dir ./visualizations/dual_nopred

"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import cv2
from tqdm import tqdm
import re
import pickle

from src.utils.bev_axes import configure_bev_axes


class TrajectoryAnalyzer:
    """轨迹分析器"""
    
    def __init__(self):
        """初始化轨迹分析器"""
        pass
    
    def _extract_image_path(self, img_ref) -> Optional[str]:
        """从不同格式的图像引用中提取路径字符串"""
        if img_ref is None:
            return None
        
        # 处理 EagerTensor 类型（TensorFlow）
        if hasattr(img_ref, "numpy"):
            img_ref = img_ref.numpy()
        
        # 处理 dict 类型（StringWrapper 序列化为 dict 的情况）
        if isinstance(img_ref, dict) and '_string_wrapper_marker' in img_ref:
            img_ref = img_ref.get('_bytes_value', b'')
        
        # 处理 bytes 或 str
        if isinstance(img_ref, bytes):
            return img_ref.decode("utf-8")
        elif isinstance(img_ref, str):
            return img_ref
        else:
            return None
    
    def load_original_trajectories_from_pkl(
        self,
        pkl_dir: str,
        scene_name: Optional[str] = None
    ):
        """
        从 pkl 文件加载原始轨迹数据
        
        Args:
            pkl_dir: pkl 文件目录路径
            scene_name: 场景名称（如 "scene-0001"），None 表示加载所有场景
        
        Returns:
            Dict: {
                'trajectories': List[np.ndarray],  # 每个样本的轨迹 [horizon, 2]
                'pkl_files': List[Path],           # 对应的 pkl 文件路径
                'scene_names': List[str],         # 对应的场景名称
                'step_nums': List[int]            # 对应的 step 编号
            }
        """
        pkl_dir_path = Path(pkl_dir)
        if not pkl_dir_path.exists():
            raise FileNotFoundError(f"pkl 目录不存在: {pkl_dir}")
        
        trajectories = []
        pkl_files = []
        scene_names_list = []
        step_nums = []
        
        # 收集所有 pkl 文件
        all_pkl_files = list(pkl_dir_path.glob("*.pkl"))
        
        # 按场景名称过滤
        if scene_name:
            all_pkl_files = [f for f in all_pkl_files if f.name.startswith(f"{scene_name}_step")]
        
        # 按文件名排序（确保场景和 step 顺序正确）
        all_pkl_files.sort(key=lambda x: x.name)
        
        print(f"找到 {len(all_pkl_files)} 个 pkl 文件")
        if scene_name:
            print(f"过滤场景: {scene_name}")
        
        for pkl_file in tqdm(all_pkl_files, desc="加载 pkl 文件"):
            try:
                # 提取场景名和 step 编号
                match = re.match(r'(.+)_step(\d+)\.pkl$', pkl_file.name)
                if not match:
                    continue
                
                file_scene_name = match.group(1)
                step_num = int(match.group(2))
                
                # 如果指定了场景名，再次确认
                if scene_name and file_scene_name != scene_name:
                    continue
                
                # 加载 pkl 文件
                with open(pkl_file, "rb") as f:
                    try:
                        data = pickle.load(f, encoding='latin-1')
                    except TypeError:
                        data = pickle.load(f)
                
                # 提取轨迹数据
                fur_x = data.get('fur_x', None)
                fur_y = data.get('fur_y', None)
                
                if fur_x is None or fur_y is None:
                    print(f"警告: {pkl_file.name} 缺少 fur_x 或 fur_y 字段，跳过")
                    continue
                
                # 转换为 numpy 数组
                if isinstance(fur_x, np.ndarray):
                    x_coords = fur_x
                else:
                    x_coords = np.array(fur_x)
                
                if isinstance(fur_y, np.ndarray):
                    y_coords = fur_y
                else:
                    y_coords = np.array(fur_y)
                
                # 处理形状：fur_x 和 fur_y 可能是 (1, horizon) 或 (horizon,)
                if len(x_coords.shape) == 2:
                    x_coords = x_coords.squeeze(0)
                if len(y_coords.shape) == 2:
                    y_coords = y_coords.squeeze(0)
                
                # 组合成轨迹 [horizon, 2]
                if len(x_coords) != len(y_coords):
                    print(f"警告: {pkl_file.name} 的 fur_x 和 fur_y 长度不匹配，跳过")
                    continue
                
                trajectory = np.stack([x_coords, y_coords], axis=1)  # [horizon, 2]
                
                trajectories.append(trajectory)
                pkl_files.append(pkl_file)
                scene_names_list.append(file_scene_name)
                step_nums.append(step_num)
                
            except Exception as e:
                print(f"加载 {pkl_file.name} 失败: {e}")
                continue
        
        print(f"✓ 成功加载 {len(trajectories)} 个轨迹")
        
        return {
            'trajectories': trajectories,
            'pkl_files': pkl_files,
            'scene_names': scene_names_list,
            'step_nums': step_nums
        }
    
    def visualize_original_trajectories(
        self,
        pkl_dir: str,
        scene_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        show_image: bool = True,
        bev_xlim: Optional[Tuple[float, float]] = None,
        bev_ylim: Optional[Tuple[float, float]] = None,
    ):
        """
        可视化从 pkl 文件加载的原始轨迹
        
        Args:
            pkl_dir: pkl 文件目录路径
            scene_name: 场景名称（如 "scene-0001"），None 表示加载所有场景
            output_dir: 输出目录（如果为 None，则显示图像）
            show_image: 是否显示前视图像（dual 视图）
            bev_xlim: BEV X 轴范围
            bev_ylim: BEV Y 轴范围
        """
        # 加载轨迹数据
        data = self.load_original_trajectories_from_pkl(pkl_dir, scene_name)
        trajectories = data['trajectories']
        pkl_files = data['pkl_files']
        scene_names_list = data['scene_names']
        step_nums = data['step_nums']
        
        if len(trajectories) == 0:
            print("错误: 没有加载到任何轨迹数据")
            return
        
        # 设置输出目录
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            output_dir_path = None
        
        # 可视化每个轨迹
        for idx, (traj, pkl_file, file_scene_name, step_num) in enumerate(
            tqdm(zip(trajectories, pkl_files, scene_names_list, step_nums), 
                 desc="可视化轨迹", total=len(trajectories))
        ):
            # 创建图形
            if show_image:
                # Dual 视图：左图前视图像，右图 BEV 轨迹
                fig, axes = plt.subplots(
                    1, 2,
                    figsize=(16, 6),
                    gridspec_kw={'width_ratios': [1.6, 1.0]}
                )
                
                # 加载前视图像
                try:
                    with open(pkl_file, "rb") as f:
                        try:
                            pkl_data = pickle.load(f, encoding='latin-1')
                        except TypeError:
                            pkl_data = pickle.load(f)
                    
                    img_ref = pkl_data.get("his_cam_front", None)
                    img_path = self._extract_image_path(img_ref)
                    
                    if img_path and Path(img_path).exists():
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            axes[0].imshow(img_rgb)
                            axes[0].set_title(f"Front image ({file_scene_name}, step {step_num})")
                        else:
                            axes[0].text(0.5, 0.5, "无法加载图像", ha='center', va='center')
                            axes[0].set_title(f"Front image ({file_scene_name}, step {step_num})")
                    else:
                        axes[0].text(0.5, 0.5, "图像文件不存在", ha='center', va='center')
                        axes[0].set_title(f"Front image ({file_scene_name}, step {step_num})")
                except Exception as e:
                    axes[0].text(0.5, 0.5, f"加载图像失败: {e}", ha='center', va='center')
                    axes[0].set_title(f"Front image ({file_scene_name}, step {step_num})")
                
                axes[0].axis("off")
                ax_bev = axes[1]
            else:
                # 仅 BEV 视图
                fig, ax_bev = plt.subplots(1, 1, figsize=(10, 8))
            
            # 绘制 BEV 轨迹
            # 数值上：traj[:, 0] / traj[:, 1] 分别是原始 fur_x / fur_y
            # 可视化约定：
            #   - 纵轴：x，向上为正
            #   - 横轴：y，向左为正
            ax_bev.plot(traj[:, 1], traj[:, 0], "b-", linewidth=2, label="Trajectory", marker='o', markersize=4)
            
            # 标记起始点
            ax_bev.scatter(traj[0, 1], traj[0, 0], color='green', s=100, marker='s', 
                          label='Start', zorder=5)
            # 标记终点
            ax_bev.scatter(traj[-1, 1], traj[-1, 0], color='red', s=100, marker='s', 
                          label='End', zorder=5)
            
            configure_bev_axes(ax_bev)
            ax_bev.set_title(f"BEV Trajectory ({file_scene_name}, step {step_num})")
            
            # 设置坐标轴范围（基于原始 (x, y) 数值）
            x_min, x_max = traj[:, 0].min(), traj[:, 0].max()  # 纵轴
            y_min, y_max = traj[:, 1].min(), traj[:, 1].max()  # 横轴
            margin_x = max(1.0, 0.1 * (x_max - x_min + 1e-6))
            margin_y = max(1.0, 0.1 * (y_max - y_min + 1e-6))
            
            if bev_ylim:
                # 横轴范围使用 y
                ax_bev.set_xlim(bev_ylim)
            else:
                ax_bev.set_xlim(y_min - margin_y, y_max + margin_y)
            
            if bev_xlim:
                # 纵轴范围使用 x
                ax_bev.set_ylim(bev_xlim)
            else:
                ax_bev.set_ylim(x_min - margin_x, x_max + margin_x)
            
            # 横轴方向：向右 y 变大，向左 y 变小（标准方向，无需反转）
            
            ax_bev.set_aspect("equal")
            ax_bev.grid(True, alpha=0.3)
            ax_bev.legend()
            
            # 在图上显示轨迹坐标
            traj_text = "\n".join([f"t={i}: ({traj[i, 0]:.2f}, {traj[i, 1]:.2f})" 
                                  for i in range(len(traj))])
            fig.text(
                0.93, 0.5 if show_image else 0.3,
                f"Trajectory points:\n{traj_text}",
                ha="left", va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )
            
            plt.tight_layout()
            
            # 保存或显示
            if output_dir_path:
                save_path = output_dir_path / f"{file_scene_name}_step{step_num:03d}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="轨迹数据可视化工具（从 pkl 文件加载原始轨迹）")
    parser.add_argument(
        "--pkl_dir",
        type=str,
        required=True,
        help="pkl 文件目录路径"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="指定场景名称（如 scene-0001），None 表示加载所有场景"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（用于保存可视化结果；未指定则直接显示）"
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
    analyzer = TrajectoryAnalyzer()
    
    # 可视化原始轨迹（总是显示前视图像）
    analyzer.visualize_original_trajectories(
        pkl_dir=args.pkl_dir,
        scene_name=args.scene_name,
        output_dir=args.output_dir,
        show_image=True,  # 总是显示前视图像
        bev_xlim=tuple(args.bev_xlim) if args.bev_xlim else None,
        bev_ylim=tuple(args.bev_ylim) if args.bev_ylim else None,
    )


if __name__ == "__main__":
    main()
