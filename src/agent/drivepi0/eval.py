import logging
import os

import hydra
import einops
import imageio
from tqdm import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time
from src.data.utils.normalization import Normalize

log = logging.getLogger(__name__)


class DrivePiZeroEvalAgent:
    """
    DrivePi-0 的评估 Agent。
    评估流程总体分为几步：
      1. 按照 Hydra 配置加载已经训练好的 DrivePiZeroInference 模型，并加载指定 checkpoint。
      2. 用 Bench2Drive 的 val 划分构建 DataLoader。
      3. 对每个 batch：
         - 预处理图像（前视当前帧 + 历史帧）、状态、语言指令；
         - 调用 self.model.infer_action 得到预测轨迹 preds；
         - 在归一化空间下，用 get_action_accuracy 计算多阈值准确率；
         - 再用 Normalize 把 preds / gt 从 [-1, 1] 反归一化到真实坐标系，计算 L1 / L2 损失。
      4. 对所有 batch 的指标取平均，打印到日志。
    """
    def __init__(self, cfg):
        """
        初始化评估 Agent。

        参数:
          cfg: Hydra 解析后的配置对象，对应 config/eval/DrivePi0/base.yaml。
        """
        # 日志目录（由 Hydra 配置指定，一般是 logs/eval/...）
        self.log_dir = cfg.log_dir
        self.config = cfg
        # 是否保存轨迹数据（用于后续可视化）
        self.save_trajectories = cfg.get("save_trajectories", False)
        if self.save_trajectories:
            # 创建轨迹数据保存目录
            self.trajectory_save_dir = os.path.join(self.log_dir, "trajectories")
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            log.info(f"Trajectory saving enabled. Data will be saved to: {self.trajectory_save_dir}")
            # 用于存储所有轨迹数据
            self.all_pred_trajectories = []
            self.all_gt_trajectories = []
        
        # ------------------------------
        # 1) 加载推理模型 DrivePiZeroInference
        # ------------------------------
        # 选择 GPU 设备
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        # 评估精度：优先使用 bf16，若未开启则使用 float32
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        # 创建推理模型（注意这里是 DrivePiZeroInference，而非训练用 DrivePiZero）
        # use_ddp=False：评估默认单机单卡，不需要 DDP 包裹
        self.model = DrivePiZeroInference(cfg, use_ddp=False)

        # 从指定的 checkpoint 恢复参数（cfg.checkpoint_path 在 eval 配置里配置）
        self.load_checkpoint(cfg.checkpoint_path)

        # 评估阶段不需要训练，冻结全部参数
        self.model.freeze_all_weights()
        # 把模型权重搬到指定 dtype & GPU 上
        self.model.to(self.dtype)
        self.model.to(self.device)

        # 可选：使用 torch.compile 对模型进行编译，加速推理（首个 batch 会有编译开销）
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # 设置为 eval 模式（关闭 dropout 等）
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        # 打印当前已分配的 GPU 显存，方便监控
        log_allocated_gpu_memory(log, "loading model")
        
        # ------------------------------
        # 2) 构建验证 DataLoader
        # ------------------------------
        # Bench2DriveDataset(cfg.data).dataset 返回 torch Dataset，
        # cfg.data 在 config/eval/DrivePi0/base.yaml 里定义（包括 work_dir / split=val 等）
        self.val_dataloader = DataLoader(
            Bench2DriveDataset(cfg.data).dataset, 
            batch_size=cfg.device_batch_size,
            pin_memory=True,
            num_workers=cfg.num_workers
        )
        # 多阈值列表，比如 [0.05, 0.1, 0.2, 0.3, 0.5]
        self.eval_thresholds = cfg.eval_thresholds
        
        # ------------------------------
        # 3) 准备 tokenizer / VLAProcessor，将图像+文本打包成模型输入
        # ------------------------------
        # 文本 tokenizer（与预训练模型保持一致）
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        # VLAProcessor：封装了图像 + 文本 -> token 序列 的预处理逻辑，
        # 会输出 input_ids / pixel_values / attention_mask 等
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        print('Agent done')

    def run(self):
        """
        评估入口函数。核心逻辑：

          - 遍历 val_dataloader
          - 对每个 batch：
              1) 预处理 batch => inputs
              2) 推理动作 => preds
              3) 计算多阈值准确率（归一化空间）
              4) 反归一化到真实坐标，计算 L1/L2
          - 汇总所有 batch 的平均指标并打印日志。
        """
        # 当前版本中未使用的变量，保留做接口兼容或后续扩展（如 episode 级别指标）
        cnt_episode = 0
        successes = []
        # 累计的多阈值准确率张量，长度 = len(eval_thresholds)
        eval_accuracy = torch.zeros(len(self.eval_thresholds), device=self.device)
        # 累计 L1 / L2 误差（最终会除以 batch 数）
        eval_l1_loss = torch.tensor(0.0, device=self.device)
        eval_l2_loss = torch.tensor(0.0, device=self.device)
        
        # ============================================================
        # 内部函数：对一个 batch 做预处理，转成模型期望的输入格式
        # ============================================================
        def preprocess_batch(batch, split_mask: bool, sample_fm_time: bool):
            """
            预处理 DataLoader 给出的一个 batch。

            batch 的结构（来自 Bench2DriveDataset）：
              - batch["image_front"]:        (B, H, W, 3) 当前前视图
              - batch["image_front_time"]:   (B, H, W, 3) 历史前视图（一个时间点）
              - batch["image_back"]:         (B, H, W, 3) 后视图（当前版本未拼入）
              - batch["state"]:              (B, window, state_dim) 车辆状态
              - batch["trajectory"]:         (B, 1, horizon, action_dim) 轨迹（已归一化）
              - batch["language_instruction"]: 长度 B 的文本指令列表

            参数:
              split_mask: 是否将 full causal mask 分拆成 image_text_proprio_mask / action_mask
              sample_fm_time: 是否采样 flow matching 时间 t（只在训练用，这里一般为 False）
            """
            images_front = batch["image_front"]                 # 前视当前帧
            images_front_history = batch["image_front_time"]    # 前视历史帧
            images_back = batch["image_back"]                   # 后视图（暂未拼接入模型，但保留以便未来修改）
            # 当前 batch 大小
            batch_size = images_front.shape[0]
            # 状态（proprio）和未来轨迹（actions）
            state = batch["state"]
            # 原始 shape 为 (B, 1, horizon, action_dim)，这里去掉冗余的时间维度
            trajectory = batch["trajectory"].squeeze(1)  # remove the time dimension
            # 文本描述/指令
            texts = batch["language_instruction"]
            # =============================
            # 图像预处理：BHWC -> BCHW，再加上“帧维度”
            # =============================
            # 前视当前帧: (B, H, W, C) -> (B, C, H, W)
            images_front = einops.rearrange(images_front, "B H W C -> B C H W")  # remove cond_steps dimension
            # 在第 1 维增加帧维度 => (B, 1, C, H, W)
            images_front = images_front.unsqueeze(1)
            
            # 前视历史帧: (B, H, W, C) -> (B, C, H, W) -> (B, 1, C, H, W)
            images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")  # remove cond_steps dimension
            images_front_history = images_front_history.unsqueeze(1)

            # 后视图: (B, H, W, C) -> (B, C, H, W) -> (B, 1, C, H, W)
            images_back = einops.rearrange(images_back, "B H W C -> B C H W")  # remove cond_steps dimension
            images_back = images_back.unsqueeze(1)
            
            # 如需三路图像一起送入模型，可以用下面这行（目前被注释掉）：
            # images = torch.cat((images_front, images_front_history, images_back), dim=1)
            # 当前只使用前视当前帧 + 历史帧，共 2 帧
            images = torch.cat((images_front, images_front_history), dim=1)

            # VLAProcessor 负责把 (文本, 多帧图像) 转成模型的多模态输入：
            #   - input_ids
            #   - pixel_values
            #   - attention_mask
            model_inputs = self.processor(text=texts, images=images)

            # =======================================
            # 构造轨迹的 causal_mask 和 position_ids
            # =======================================
            # 根据 attention_mask 构造：
            #   - causal_mask: 完整自回归 mask
            #   - vlm_position_ids: 图像+文本 token 的 position id
            #   - state_position_ids: 状态 token 的 position id
            #   - action_position_ids: 动作 token 的 position id
            causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
                self.model.build_causal_mask_and_position_ids(
                    model_inputs["attention_mask"], self.dtype
                )
            )

            # 把所有需要的输入打包为一个 dict（后续直接 **inputs 传给模型）
            inputs = {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": model_inputs["pixel_values"].to(self.dtype),
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": state_position_ids,
                "action_position_ids": action_position_ids,
                "proprios": state.to(self.dtype),
                "actions": trajectory.to(self.dtype),
            }
            # 是否拆分 mask
            if split_mask:
                # 将 full causal mask 拆成：
                #   - image_text_proprio_mask：图像+文本+状态部分的 mask
                #   - action_mask：动作 token 部分的 mask
                image_text_proprio_mask, action_mask = (
                    self.model.split_full_mask_into_submasks(causal_mask)
                )
                inputs["image_text_proprio_mask"] = image_text_proprio_mask
                inputs["action_mask"] = action_mask
            else:
                # 直接使用完整的 causal_mask
                inputs["causal_mask"] = causal_mask

            # 采样 flow-matching 时间 t（评估默认不启用）
            if sample_fm_time:
                # 注意：DrivePiZeroEvalAgent 本身并没有定义 sample_fm_time，
                # 这一分支是为了接口对齐 DrivePiZeroTrainAgent ，一般 eval 时不会走到这里。
                inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

            # 把所有 tensor 移动到 GPU 上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        
        # 用于统计一共处理了多少个 batch，后面做平均
        cnt = 0
        print('Running evaluation ... ')
        
        # =====================================
        # 主循环：遍历整个验证集 DataLoader
        # =====================================
        for batch in tqdm(self.val_dataloader):
            # 1) 预处理当前 batch
            inputs = preprocess_batch(batch, split_mask=True, sample_fm_time=False)
            # ground truth 轨迹（仍在归一化空间，形状 [B, horizon, 2]）
            gt_trajectory = inputs.pop("actions")
            # 2) 调用模型进行轨迹推理（只前向，不反向）
            preds = self.model.infer_action(**inputs)
            # 3) 在归一化空间计算多阈值准确率
            #    get_action_accuracy 会对每个阈值 t 计算：
            #       acc_t = ( |pred - gt| < t 的样本/点 比例 )
            eval_accuracy += get_action_accuracy(gt_trajectory, preds, self.eval_thresholds)
            # 4) 使用 Normalize 单例，将预测/真实轨迹从 [-1,1] 反归一化回真实坐标（米等物理单位）
            normalize = Normalize.get_instance('statistics.json')
            pred_trajectory = []
            label_trajectory = []

            # 4.1 反归一化预测轨迹
            for pred in preds:
                # pred 原 shape: [horizon, 2]，先加 batch 维变成 [1, horizon, 2]
                pred = pred.unsqueeze(0)
                # infer_traj 内部使用 fur_x_01 / fur_x_99, fur_y_01 / fur_y_99
                # 对 x / y 分量做 standard_normalize 的逆变换：
                # (norm + 1) * (p_99 - p_01) / 2 + p_01
                x_pred, y_pred = normalize.infer_traj(pred) # use true trajectory to calculate l1 loss and l2 loss
                # 拼回 [horizon, 2]
                pred = torch.stack([x_pred, y_pred], dim=1)
                # 再加 batch 维 -> [1, horizon, 2]
                pred = pred.unsqueeze(0)
                pred_trajectory.append(pred)

            # 4.2 反归一化真实轨迹
            for label in gt_trajectory:
                label = label.unsqueeze(0)
                x_label, y_label = normalize.infer_traj(label)
                label = torch.stack([x_label, y_label], dim=1)
                label = label.unsqueeze(0)
                label_trajectory.append(label)
            
            # 拼成 [B, horizon, 2]
            pred_trajectory = torch.cat(pred_trajectory, dim=0)
            label_trajectory = torch.cat(label_trajectory, dim=0)
            cnt += 1
            # 5) 在真实坐标系下计算 L1 / L2 损失（每个 batch 得到一个平均误差，再累加）
            eval_l1_loss += torch.nn.functional.l1_loss(pred_trajectory, label_trajectory)
            eval_l2_loss += torch.nn.functional.mse_loss(pred_trajectory, label_trajectory)
            
            # 6) 保存轨迹数据（用于后续可视化）
            if self.save_trajectories:
                # 转换为numpy数组（CPU上）
                pred_traj_np = pred_trajectory.cpu().numpy()  # [B, horizon, 2]
                gt_traj_np = label_trajectory.cpu().numpy()    # [B, horizon, 2]
                
                # 存储轨迹数据
                self.all_pred_trajectories.append(pred_traj_np)
                self.all_gt_trajectories.append(gt_traj_np)
        
        # =====================================
        # 7) 保存轨迹数据（如果启用）
        # =====================================
        if self.save_trajectories:
            # 合并所有batch的轨迹数据
            if len(self.all_pred_trajectories) > 0:
                all_pred = np.concatenate(self.all_pred_trajectories, axis=0)  # [N, horizon, 2]
                all_gt = np.concatenate(self.all_gt_trajectories, axis=0)      # [N, horizon, 2]
                
                # 保存为numpy文件
                pred_save_path = os.path.join(self.trajectory_save_dir, "pred_trajectories.npy")
                gt_save_path = os.path.join(self.trajectory_save_dir, "gt_trajectories.npy")
                
                np.save(pred_save_path, all_pred)
                np.save(gt_save_path, all_gt)
                
                log.info(f"Saved {len(all_pred)} trajectories to {self.trajectory_save_dir}")
                log.info(f"  - Predicted trajectories: {pred_save_path}")
                log.info(f"  - GT trajectories: {gt_save_path}")
        
        # =====================================
        # 8) 对所有 batch 的累积值取平均
        # =====================================
        eval_accuracy = eval_accuracy / cnt
        eval_l1_loss = eval_l1_loss / cnt
        eval_l2_loss = eval_l2_loss / cnt
        
        # =====================================
        # 9) 打印 summary 日志
        # =====================================
        log.info(f"Number of episodes: {cnt}")
        log_msg = f"Eval | l1 Loss: {eval_l1_loss.item():.3f} | "
        log_msg += f"l2 Loss: {eval_l2_loss.item():.3f} | "
        log_msg += " | ".join(
            [
                f"acc thres {threshold}: {accuracy.item():.3f}"
                for threshold, accuracy in zip(
                    self.eval_thresholds, eval_accuracy
                    )
                ]
            )
        log.info(log_msg)
        

        
    @log_execution_time(log)
    def load_checkpoint(self, path):
        """
        从训练阶段保存的 checkpoint 中加载模型权重。

        注意点：
          - 如果训练时模型被 torch.compile 包装过，state_dict 的 key 会带有 "_orig_mod." 前缀，
            这里通过 replace("_orig_mod.", "") 去掉前缀，以便在未编译的推理模型上正常加载。
          - 为了节省 GPU 显存，先用 map_location="cpu" 把权重加载到 CPU，再手动把模型 to(self.device)。
        """
        # 先全部加载到 CPU，再转到 GPU
        data = torch.load(path, weights_only=True, map_location="cpu")
        # 去除可能存在的 "_orig_mod." 前缀（来自 torch.compile）
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        # 严格模式加载权重（所有 key 必须匹配）
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")