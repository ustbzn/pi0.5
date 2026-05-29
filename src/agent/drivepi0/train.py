import logging
import math
import os
import random
import time
from collections import deque

import bitsandbytes as bnb
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, Qwen3VLProcessor

import swanlab
# import wandb
from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.drivepi0 import DrivePiZero
from src.model.DrivePi0.processing import VLAProcessor
from src.data.utils.normalization import Normalize
from src.utils.metric import (
    get_action_accuracy,
    get_action_accuracy_counts,
    get_action_l1_sum_count,
    get_bucketed_action_accuracy,
    get_denormalized_action_avg_l2_sum_count,
)
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions

log = logging.getLogger(__name__)

# os.environ["WANDB__SERVICE_WAIT"] = "300"


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """WeightedRandomSampler + DistributedSampler 的组合版。

    所有 rank 使用相同的 seed+epoch 生成同一份全局采样索引，
    然后每个 rank 只取属于自己的切片，保证：
      1) 各 rank 数据互不重叠
      2) 加权采样逻辑保持不变
      3) 调用 set_epoch() 后每个 epoch 的数据顺序不同
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, seed=0):
        import torch.distributed as dist
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        # 向上取整，保证各 rank 样本数相同
        self.num_samples_per_rank = int(math.ceil(num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 所有 rank 用同一 generator 生成全局索引
        indices = torch.multinomial(
            self.weights, self.total_size,
            replacement=self.replacement, generator=g,
        ).tolist()
        # 每个 rank 取间隔为 num_replicas 的子集
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples_per_rank
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch


class DrivePiZeroTrainAgent:
    def __init__(self, cfg):
        self.config = cfg
        # seeding
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # devices
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = cfg.multi_gpu
        world_size = 1  # single gpu
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            log.info(
                f"GPU local ID: {self.gpu_id}. Global rank: {global_rank}. Local rank: {local_rank}. Local world size: {local_world_size}. World size: {world_size}. Group rank: {group_rank}"
            )
            for i in range(torch.cuda.device_count()):
                log.info(
                    f"Local rank: {local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}"
                )
        self.main_rank = not self.multi_gpu or global_rank == 0

        # logging
        # self.use_wandb = cfg.get("wandb", False)
        # if self.use_wandb and self.main_rank:
        #     wandb.init(
        #         entity=cfg.wandb.entity,
        #         project=cfg.wandb.project,
        #         name=cfg.wandb.run,
        #         mode="offline",
        #         config=OmegaConf.to_container(cfg, resolve=True),
        #         id=self.wandb_id if hasattr(self, "wandb_id") else None,
        #         resume="allow",  # not using resume_from
        #     )
        # =========== SwanLab 初始化 ===========
        self.use_swanlab = cfg.get("swanlab", {}).get("enable", False)
        self.swanlab_run = None
        if self.use_swanlab and self.main_rank:
            swan_log_dir = os.path.join(os.getcwd(), "swanlog")
            os.makedirs(swan_log_dir, exist_ok=True)
            self.swanlab_run = swanlab.init(
                project=cfg.swanlab.project,
                experiment_name=cfg.swanlab.experiment_name,
                description=cfg.swanlab.get("description", None),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.swanlab.get("mode", "cloud"),
                entity=cfg.swanlab.get("entity", None),
                logdir=swan_log_dir,
            )
            log.info(f"SwanLab log directory set to: {swan_log_dir}")
        
        self.save_model_freq = int(cfg.save_model_freq)
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm
        self.use_amp = cfg.get("use_amp", True)

        # model
        assert not (
            (cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights
        ), "Please load pretrained weights if quantizing VLM or using Lora."
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )  # since the weights have requires_grad=False. However, we are not excluding the weights from the optimizer yet!
        # dtype 处理 (支持 fp16)
        if cfg.get("use_bf16", False):
            self.dtype = torch.bfloat16
        elif cfg.get("use_fp16", False):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        # 判断加载模式：
        #   resume_checkpoint_path → 断点续训（恢复权重+优化器+进度）
        #   finetune_checkpoint_path → 微调（只加载模型权重，优化器和进度从头开始）
        #   都没设置 → 首次训练（加载预训练权重）
        has_resume = bool(cfg.get("resume_checkpoint_path"))
        has_finetune = bool(cfg.get("finetune_checkpoint_path"))
        load_pretrained = not has_resume and not has_finetune

        self.model = DrivePiZero(cfg, use_ddp=self.multi_gpu, load_pretrained=load_pretrained)
        if has_resume:
            # Resume: inject LoRA first so checkpoint keys match
            if cfg.lora:
                self.model.inject_lora(cfg)
            self.load_checkpoint(cfg.resume_checkpoint_path)
        elif has_finetune:
            # Finetune: load base weights first, then inject LoRA
            self._load_model_weights_only(cfg.finetune_checkpoint_path)
            if cfg.lora:
                self.model.inject_lora(cfg)
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
            if cfg.lora:
                self.model.inject_lora(cfg)
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        # 如果不训练 VLM，则彻底冻结 VLM 相关权重（vision tower / projector / gemma）
        if not cfg.train_vlm:
            self.model.freeze_vlm_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            # self.model = torch.compile(
            #     self.model,
            #     mode="default",  # "reduce-overhead" speeds up a lot and reduces VRAM usage a lot more, but causes nan loss on L40, maybe issue with cudagraphs or 8-bit optimizer; max-autotune works on H100s, takes a while to compile
            #     # backend="inductor", # default: inductor; cudagraphs
            # )
            # vision_tower 使用官方实现，内部有 .item() 调用，不能被 compile
    # 只 compile 除 vision_tower 之外的部分
            self.model.joint_model = torch.compile(
                self.model.joint_model,
                mode="default",
            )
            self.model.action_encoder = torch.compile(
                self.model.action_encoder,
                mode="default",
            )
            self.model.action_decoder = torch.compile(
                self.model.action_decoder,
                mode="default",
            )
            self.model.proprio_encoder = torch.compile(
                self.model.proprio_encoder,
                mode="default",
            )
            log.info("Compiled joint_model, action_encoder, action_decoder, proprio_encoder")
            # vision_tower 不 compile，避免 graph break
            log.info("Skipped compiling vision_tower (contains .item() calls incompatible with compile)")
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        log.info(f"Using cuda device: {self.device}, dtype: {self.dtype}")
        if self.multi_gpu:
            log.info(
                f"Using {local_world_size} GPUs in each of the {cfg.n_nodes} nodes"
            )
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                gradient_as_bucket_view=True,
                static_graph=False,  # didn't see improvement with True
            )
            model = self.model.module
            dist.barrier()
        else:
            model = self.model
        log_allocated_gpu_memory(log, "loading model", self.gpu_id)

        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = max(
            cfg.global_batch_size // cfg.per_device_batch_size // world_size, 1
        )
        actual_global_batch_size = (
            cfg.per_device_batch_size * self.grad_accumulation_steps * world_size
        )

        # dataloader --- spawn one for each rank, num_workers=0
        train_dataset = Bench2DriveDataset(cfg.data.train).dataset

        # ---- 构建采样权重 (如果启用了加权采样) ----
        sample_weights = None

        # 加权采样：平衡 navigation_id 类别分布
        if cfg.get("nav_weighted_sampling", False):
            log.info("Building weighted sampler for navigation_id balance...")
            nav_labels = train_dataset.get_nav_labels()
            from collections import Counter
            label_counts = Counter(nav_labels)
            log.info(f"Navigation label distribution: {dict(sorted(label_counts.items()))}")
            # 每个类别的权重与其频率成反比
            sample_weights = [1.0 / label_counts[l] for l in nav_labels]

        # 加权采样：平衡轨迹运动模式（静止/转弯/直行）
        traj_sampling_cfg = cfg.get("trajectory_weighted_sampling", None)
        if traj_sampling_cfg and traj_sampling_cfg.get("enable", False):
            log.info("Building weighted sampler for trajectory type balance...")
            traj_weights, stats = train_dataset.get_trajectory_weights(
                static_threshold=traj_sampling_cfg.get("static_threshold", 1.0),
                turn_threshold=traj_sampling_cfg.get("turn_threshold", 2.0),
                static_weight=traj_sampling_cfg.get("static_weight", 2.0),
                turn_weight=traj_sampling_cfg.get("turn_weight", 3.0),
                straight_weight=traj_sampling_cfg.get("straight_weight", 1.0),
                # 方案 D: hard case 加权
                hard_case_weight_path=traj_sampling_cfg.get("hard_case_weight_path", None),
                hard_case_alpha=traj_sampling_cfg.get("hard_case_alpha", 1.0),
            )
            log.info(f"Trajectory type distribution: {stats}")
            log.info(
                f"Sampling weights: straight={traj_sampling_cfg.get('straight_weight', 1.0)}, "
                f"turn={traj_sampling_cfg.get('turn_weight', 3.0)}, "
                f"static={traj_sampling_cfg.get('static_weight', 2.0)}"
            )
            sample_weights = traj_weights

        # ---- 根据 multi_gpu 和加权采样构建最终 sampler ----
        if self.multi_gpu:
            if sample_weights is not None:
                # 多卡 + 加权采样：使用 DistributedWeightedSampler
                train_sampler = DistributedWeightedSampler(
                    weights=sample_weights,
                    num_samples=len(train_dataset),
                    replacement=True,
                    seed=self.seed,
                )
                log.info("Using DistributedWeightedSampler for multi-GPU weighted sampling")
            else:
                # 多卡 + 无加权：使用标准 DistributedSampler
                train_sampler = DistributedSampler(
                    train_dataset,
                    shuffle=True,
                    seed=self.seed,
                )
                log.info("Using DistributedSampler for multi-GPU training")
        else:
            if sample_weights is not None:
                # 单卡 + 加权采样：使用 WeightedRandomSampler
                train_sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )
            else:
                train_sampler = None

        self.train_sampler = train_sampler  # 保存引用，用于 set_epoch()

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
            num_workers=cfg.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )

        self.run_eval = cfg.data.get("val", False)
        if self.run_eval:
            # use validation split; rebuild iterator each eval to avoid StopIteration
            val_dataset = Bench2DriveDataset(cfg.data.val).dataset
            if self.multi_gpu:
                self.val_sampler = DistributedSampler(
                    val_dataset, shuffle=False,
                )
            else:
                self.val_sampler = None
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.per_device_batch_size,
                pin_memory=True,
                num_workers=cfg.num_workers,
                sampler=self.val_sampler,
            )
            self.eval_thresholds = cfg.eval_thresholds
            self.eval_freq = cfg.eval_freq
            self.eval_seed = int(cfg.get("eval_seed", self.seed))
            self.eval_repeats = max(int(cfg.get("eval_repeats", 1)), 1)
            self.eval_full_val = bool(cfg.get("eval_full_val", False))
            self.eval_avg_l2_normalizer = Normalize(
                cfg.data.val.statistics_path,
                norm_mode=cfg.data.val.get("norm_mode", "percentile"),
            )
            eval_map_modes = cfg.get("eval_map_modes", ["normal"])
            if isinstance(eval_map_modes, str):
                eval_map_modes = [eval_map_modes]
            self.eval_map_modes = list(eval_map_modes)
            valid_eval_map_modes = {"normal", "zero", "shuffle"}
            invalid_eval_map_modes = [
                mode for mode in self.eval_map_modes
                if mode not in valid_eval_map_modes
            ]
            if invalid_eval_map_modes:
                raise ValueError(f"Invalid eval_map_modes: {invalid_eval_map_modes}")

            if self.eval_full_val:
                self.per_device_num_eval_batch = (
                    len(self.val_dataloader) if len(self.val_dataloader) > 0 else 1
                )
            else:
                max_eval_batches = max(
                    int(cfg.eval_size) // cfg.per_device_batch_size // world_size,
                    1,
                )
                # cap by available val batches to avoid exhausting the iterator
                self.per_device_num_eval_batch = min(
                    max_eval_batches,
                    len(self.val_dataloader) if len(self.val_dataloader) > 0 else 1,
                )
            log.info(
                "Eval config: "
                f"batches_per_device={self.per_device_num_eval_batch}, "
                f"repeats={self.eval_repeats}, seed={self.eval_seed}, "
                f"full_val={self.eval_full_val}, map_modes={self.eval_map_modes}"
            )
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {actual_global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        # optimizer - trajectory only: 0.315B (0.333B with adaLN and time_dim=256),
        # rest: 2.291B (0.109B with lora rank 64, 0.055B with rank 32)
        self.train_vlm = cfg.train_vlm
        self.state_only_for_action = cfg.get("state_only_for_action", False)
        self.action_expert_params = model.action_expert_parameters
        action_params = list(model.action_expert_parameters)
        hdmap_params = list(model.hdmap_trainable_parameters)
        self.hdmap_params = hdmap_params
        self.use_separate_hdmap_optimizer_group = bool(
            cfg.get("separate_hdmap_optimizer_group", False) and len(hdmap_params) > 0
        )
        self.trained_parameters = action_params + hdmap_params

        if self.use_separate_hdmap_optimizer_group:
            self.hdmap_lr = float(cfg.get("hdmap_lr", cfg.action_lr))
            self.hdmap_weight_decay = float(
                cfg.get("hdmap_weight_decay", cfg.action_weight_decay)
            )
            hdmap_lr_scheduler_cfg = cfg.get("hdmap_lr_scheduler", {})
            self.hdmap_min_lr = float(
                hdmap_lr_scheduler_cfg.get("min_lr", cfg.action_lr_scheduler.min_lr)
            )
            action_optimizer_params = [
                {
                    "params": action_params,
                    "lr": cfg.action_lr,
                    "weight_decay": cfg.action_weight_decay,
                },
                {
                    "params": hdmap_params,
                    "lr": self.hdmap_lr,
                    "weight_decay": self.hdmap_weight_decay,
                },
            ]
        else:
            self.hdmap_lr = None
            self.hdmap_weight_decay = None
            self.hdmap_min_lr = None
            action_optimizer_params = self.trained_parameters

        self.action_optimizer = bnb.optim.AdamW8bit(
            action_optimizer_params,
            lr=cfg.action_lr,
            weight_decay=cfg.action_weight_decay,
        )
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=(
                [cfg.action_lr, self.hdmap_lr]
                if self.use_separate_hdmap_optimizer_group
                else cfg.action_lr
            ),
            min_lr=(
                [cfg.action_lr_scheduler.min_lr, self.hdmap_min_lr]
                if self.use_separate_hdmap_optimizer_group
                else cfg.action_lr_scheduler.min_lr
            ),
            warmup_steps=cfg.action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters (Trajectory): {get_num_params_in_billions(self.action_optimizer):.3f}B"
        )
        self.vlm_trained_params = []
        if self.use_separate_hdmap_optimizer_group:
            log.info(
                "HDMap optimizer group enabled: "
                f"hdmap_lr={self.hdmap_lr:.2e}, "
                f"hdmap_weight_decay={self.hdmap_weight_decay:.2e}, "
                f"hdmap_min_lr={self.hdmap_min_lr:.2e}"
            )
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            self.vlm_trained_params = vlm_trained_parameters
            self.trained_parameters += vlm_trained_parameters
            self.vlm_optimizer = bnb.optim.AdamW8bit(
                vlm_trained_parameters,
                lr=cfg.vlm_lr,
                weight_decay=cfg.vlm_weight_decay,
            )
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            log.info(
                f"Number of trained parameters (VLM): {get_num_params_in_billions(self.vlm_optimizer):.3f}B"
            )
        if cfg.resume_checkpoint_path:
            self.load_optimizer(cfg.resume_checkpoint_path)
        # finetune 时不恢复优化器，使用全新的学习率调度

        ########### Input processing ###########

        # flow matching timestep sampling
        self.flow_sampling = cfg.get("flow_sampling", "beta")
        assert self.flow_sampling in [
            "uniform",
            "beta",
        ], f"Invalid flow matching timestep sampling mode: {self.flow_sampling}"
        if self.flow_sampling == "beta":
            flow_alpha = cfg.get("flow_alpha", 1.5)
            flow_beta = cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        if self.state_only_for_action:
            self.tokenizer = None
            self.processor = None
            self.processor_padding = None
            self.processor_max_length = None
            self.image_token_id = None
            log.info("Strict state-only enabled; skipping Qwen3VLProcessor initialization.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.pretrained_model_path,
                trust_remote_code=True,
                padding_side="right",
            )

            self.processor = Qwen3VLProcessor.from_pretrained(
                cfg.pretrained_model_path,
                trust_remote_code=True,
            )
            # 单独保存，供调用时使用
            self.processor_padding = cfg.tokenizer_padding
            self.processor_max_length = cfg.max_seq_len
            self.image_token_id = self.processor.image_token_id
            print(f"image_token_id: {self.image_token_id}")
            # 验证聊天模板正常
            test_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}]
            test_text = self.processor.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True)
            print(f"Chat template test: {test_text[:100]}...")
        print('agent done')

        # HDMap 开关
        self.use_hdmap_image = cfg.data.train.get("use_hdmap_image", False)
        if self.use_hdmap_image:
            log.info("HDMap BEV image enabled (Plan A): will add BEV as an extra visual input")
        self.use_hdmap_cnn = cfg.data.train.get("use_hdmap_cnn", False)
        if self.use_hdmap_cnn:
            log.info("HDMap CNN encoder enabled (Plan C): will encode raw BEV into proprio")
        self.use_hdmap_vector = cfg.data.train.get("use_hdmap_vector", False)
        if self.use_hdmap_vector:
            log.info("HDMap vector tokens enabled (Plan D, PlanVOCA-style)")
        # 与 dataset 端 max_map_tokens 对齐，仅用于在 preprocess_batch 中传给 model.build_causal_mask
        self.max_map_tokens = int(cfg.data.train.get("max_map_tokens", 128))

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    @staticmethod
    def _snapshot_rng_state():
        state = {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        torch.random.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _set_eval_seed(seed: int):
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _collect_map_tensors(self, batch):
        """收集 PlanVOCA-style map token tensors，统一 dtype。

        浮点 tensor → self.dtype；整型/bool 按 MapEncoder 期望保持 long/bool。
        device 由调用方 .to(self.device) 统一处理。
        """
        out = {}
        if "map_point_position" in batch:
            out["map_point_position"] = batch["map_point_position"].to(self.dtype)
        if "map_point_vector" in batch:
            out["map_point_vector"] = batch["map_point_vector"].to(self.dtype)
        if "map_point_orientation" in batch:
            out["map_point_orientation"] = batch["map_point_orientation"].to(self.dtype)
        if "map_polygon_center" in batch:
            out["map_polygon_center"] = batch["map_polygon_center"].to(self.dtype)
        if "map_polygon_type" in batch:
            out["map_polygon_type"] = batch["map_polygon_type"].long()
        if "map_polygon_on_route" in batch:
            out["map_polygon_on_route"] = batch["map_polygon_on_route"].long()
        if "map_polygon_tl_status" in batch:
            out["map_polygon_tl_status"] = batch["map_polygon_tl_status"].long()
        if "map_polygon_has_speed_limit" in batch:
            out["map_polygon_has_speed_limit"] = batch["map_polygon_has_speed_limit"].bool()
        if "map_polygon_speed_limit" in batch:
            out["map_polygon_speed_limit"] = batch["map_polygon_speed_limit"].to(self.dtype)
        if "map_point_valid_mask" in batch:
            out["map_point_valid_mask"] = batch["map_point_valid_mask"].to(self.dtype)
        if "map_valid_mask" in batch:
            out["map_valid_mask"] = batch["map_valid_mask"].to(self.dtype)
        return out

    @staticmethod
    def _apply_eval_map_mode(inputs: dict, mode: str, seed: int):
        # 兼容 vector tokens：normal/zero/shuffle 同时影响 hdmap_bev 与 map_*。
        has_bev = "hdmap_bev" in inputs
        has_vector = "map_polygon_center" in inputs
        if mode == "normal" or (not has_bev and not has_vector):
            return inputs

        inputs = dict(inputs)
        if mode == "zero":
            if has_bev:
                inputs["hdmap_bev"] = torch.zeros_like(inputs["hdmap_bev"])
            if has_vector:
                # 把 map valid mask 置 0 即可让 attention 忽略所有 lane token
                if "map_valid_mask" in inputs:
                    inputs["map_valid_mask"] = torch.zeros_like(inputs["map_valid_mask"])
                if "map_point_valid_mask" in inputs:
                    inputs["map_point_valid_mask"] = torch.zeros_like(inputs["map_point_valid_mask"])
            return inputs

        if mode == "shuffle":
            ref = inputs.get("hdmap_bev")
            if ref is None and has_vector:
                ref = inputs["map_polygon_center"]
            if ref is None or ref.shape[0] <= 1:
                return inputs
            generator = torch.Generator(device=ref.device)
            generator.manual_seed(int(seed))
            perm = torch.randperm(ref.shape[0], device=ref.device, generator=generator)
            if has_bev:
                inputs["hdmap_bev"] = inputs["hdmap_bev"][perm]
                if "hdmap_valid_mask" in inputs:
                    inputs["hdmap_valid_mask"] = inputs["hdmap_valid_mask"][perm]
            if has_vector:
                map_keys = [
                    "map_point_position", "map_point_vector", "map_point_orientation",
                    "map_polygon_center", "map_polygon_type", "map_polygon_on_route",
                    "map_polygon_tl_status", "map_polygon_has_speed_limit",
                    "map_polygon_speed_limit", "map_point_valid_mask", "map_valid_mask",
                ]
                for k in map_keys:
                    if k in inputs:
                        inputs[k] = inputs[k][perm]
            return inputs

        raise ValueError(f"Unknown eval_map_mode: {mode}")

    def run(self):
        timer = Timer()
        train_start_time = time.time()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = (
            0 if not hasattr(self, "cnt_update") else self.cnt_update
        )  # resume training if loaded checkpoint
        cnt_update_start = cnt_update  # record start update for ETA calculation
        loss_train_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False
        eval_metrics_for_log = {}
        if self.multi_gpu:
            import torch.distributed as dist

            model = self.model.module
        else:
            model = self.model
        self.model.train()

        def preprocess_batch(batch, split_mask: bool, sample_fm_time: bool):
            state = batch["state"]
            trajectory = batch["trajectory"].squeeze(1)
            texts = batch["language_instruction"]

            if self.state_only_for_action:
                batch_size = state.shape[0]
                map_valid_mask_cpu = batch.get("map_valid_mask") if self.use_hdmap_vector else None
                (
                    causal_mask,
                    vlm_position_ids,
                    state_position_ids,
                    action_position_ids,
                ) = model.build_state_only_causal_mask_and_position_ids(
                    batch_size=batch_size,
                    device=state.device,
                    dtype=self.dtype,
                    map_valid_mask=map_valid_mask_cpu,
                )

                inputs = {
                    "input_ids": torch.empty((batch_size, 0), dtype=torch.long),
                    "pixel_values": torch.empty((0,), dtype=self.dtype),
                    "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
                    "vlm_position_ids": vlm_position_ids,
                    "proprio_position_ids": state_position_ids,
                    "action_position_ids": action_position_ids,
                    "proprios": state.to(self.dtype),
                    "actions": trajectory.to(self.dtype),
                }

                if self.use_hdmap_vector and "map_polygon_center" in batch:
                    inputs.update(self._collect_map_tensors(batch))

                if split_mask:
                    state_mask, action_mask = model.split_full_mask_into_submasks(
                        causal_mask
                    )
                    inputs["image_text_proprio_mask"] = state_mask
                    inputs["action_mask"] = action_mask
                else:
                    inputs["causal_mask"] = causal_mask

                if sample_fm_time:
                    inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                return inputs

            # 4 cameras (front, right, left, back) x 2 time steps (current + history) = 8 images
            cam_names = ['front', 'right', 'left', 'back']
            # 图像语义标签：与 image_list 顺序严格对应
            cam_labels = [
                'current front view', 'current right view',
                'current left view', 'current rear view',
                'previous front view', 'previous right view',
                'previous left view', 'previous rear view',
            ]
            image_list = []
            for cam in cam_names:
                img = batch[f"image_{cam}"]  # [B, H, W, C]
                img = einops.rearrange(img, "B H W C -> B C H W")
                image_list.append(img.unsqueeze(1))  # [B, 1, C, H, W]
            for cam in cam_names:
                img = batch[f"image_{cam}_time"]  # [B, H, W, C]
                img = einops.rearrange(img, "B H W C -> B C H W")
                image_list.append(img.unsqueeze(1))  # [B, 1, C, H, W]

            # 方案 A: HDMap BEV 作为第 9 张图像
            if self.use_hdmap_image and "image_hdmap_bev" in batch:
                hdmap_img = batch["image_hdmap_bev"]  # [B, H, W, C]
                hdmap_img = einops.rearrange(hdmap_img, "B H W C -> B C H W")
                image_list.append(hdmap_img.unsqueeze(1))  # [B, 1, C, H, W]
                cam_labels.append('HDMap BEV view')

            batch_size = image_list[0].shape[0]
            num_images_per_sample = len(image_list)

            # convert to PIL images for Qwen3VLProcessor
            pil_images = []
            for i in range(batch_size):
                for img_tensor in image_list:
                    img = Image.fromarray(
                        img_tensor[i, 0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                    )
                    pil_images.append(img)

            # build Qwen3VL chat format: each image with semantic label
            chat_texts = []
            for i, text in enumerate(texts):
                content = []
                for label in cam_labels:
                    content.append({"type": "text", "text": f"{label}:"})
                    content.append({"type": "image"})
                content.append({"type": "text", "text": text})

                messages = [{"role": "user", "content": content}]
                chat_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_texts.append(chat_text)
            
            model_inputs = self.processor(
                text=chat_texts,  # ✅ 传入修改后的聊天文本列表
                images=pil_images,
                return_tensors="pt",
                padding=self.processor_padding,
                max_length=self.processor_max_length,
                truncation=True,
            ).to(self.device)
            
            # 验证图像token数量（对应正确调用代码中的验证逻辑）
            num_image_tokens = (model_inputs["input_ids"] == self.processor.image_token_id).sum().item()
            # print(f"input_ids 中图像token数量: {num_image_tokens}, pixel_values shape: {model_inputs['pixel_values'].shape}")

            # build causal mask and position ids for trajectory
            map_valid_mask_cpu = batch.get("map_valid_mask") if self.use_hdmap_vector else None
            causal_mask, _vlm_pos_1d, state_position_ids, action_position_ids = (
                model.build_causal_mask_and_position_ids(
                    model_inputs["attention_mask"], self.dtype,
                    map_valid_mask=map_valid_mask_cpu,
                )
            )
            # P0-A: 使用官方 3D M-RoPE position_ids 替代 1D arange
            vlm_position_ids = model.build_qwen3_vl_position_ids(
                model_inputs["input_ids"],
                model_inputs["image_grid_thw"],
                model_inputs["attention_mask"],
            )

            inputs = {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": model_inputs["pixel_values"].to(self.dtype),
                "image_grid_thw": model_inputs["image_grid_thw"],
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": state_position_ids,
                "action_position_ids": action_position_ids,
                "proprios": state.to(self.dtype),
                "actions": trajectory.to(self.dtype),
            }

            # 方案 C: 传递原始 BEV 数据给模型 CNN 编码器
            if self.use_hdmap_cnn and "hdmap_bev_raw" in batch:
                inputs["hdmap_bev"] = batch["hdmap_bev_raw"].to(self.dtype)
            if self.use_hdmap_cnn and "hdmap_valid_mask" in batch:
                inputs["hdmap_valid_mask"] = batch["hdmap_valid_mask"].to(self.dtype)

            # 方案 D: PlanVOCA-style 结构化 lane token
            if self.use_hdmap_vector and "map_polygon_center" in batch:
                inputs.update(self._collect_map_tensors(batch))

            if split_mask:
                image_text_proprio_mask, action_mask = (
                    model.split_full_mask_into_submasks(causal_mask)
                )
                inputs["image_text_proprio_mask"] = image_text_proprio_mask
                inputs["action_mask"] = action_mask
            else:
                inputs["causal_mask"] = causal_mask

            # sample flow matching timesteps
            if sample_fm_time:
                inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # --- 方案 A: 辅助标签通过 model._aux_labels 传递 ---
            # 不放入 inputs，不修改 forward 签名，避免与 DDP/torch.compile 冲突
            if (
                sample_fm_time
                and hasattr(model, 'enable_aux_loss')
                and model.enable_aux_loss
                and "maneuver_cls" in batch
            ):
                model._aux_labels = {
                    "maneuver_cls": batch["maneuver_cls"].long().to(self.device),
                    "stop_cls": batch["stop_cls"].long().to(self.device),
                    "endpoint": batch["endpoint"].to(self.device).to(self.dtype),
                    "progress": batch["progress"].to(self.device).to(self.dtype),
                    "curvature": batch["curvature"].to(self.device).to(self.dtype),
                }

            return inputs

        epoch = 0
        while 1:
            # ---- 每个 epoch 开始时更新 sampler 的 epoch ----
            # 确保各 rank 在每个 epoch 采样到不同的数据
            if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
                log.info(f"Set train sampler epoch to {epoch}")

            for batch in self.train_dataloader:
                '''
                batch: dict with keys 'state', 'image_{front,right,left,back}', 'image_{front,right,left,back}_time', 'trajectory', 'language_instruction'
                image_{cam}: (torch.Size([bsz, H, W, 3], uint8)  -- 4 cameras
                image_{cam}_time: (torch.Size([bsz, H, W, 3], uint8)  -- 4 cameras history
                state: (torch.Size([bsz, window, state_dim])
                trajectory:  (torch.Size([bsz, horizon, trajectory_dim], float32)
                language_instruction: btz, str
                '''

                inputs = preprocess_batch(batch, split_mask=False, sample_fm_time=True)
                # make sure only syncing when taking gradient steps
                if (cnt_batch + 1) % self.grad_accumulation_steps != 0:
                    with self.model.no_sync():
                        with torch.autocast(
                            device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                        ):
                            loss_train = self.model(**inputs)
                        normalized_loss = loss_train / self.grad_accumulation_steps
                        normalized_loss.backward()
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                    ):
                        loss_train = self.model(**inputs)
                    normalized_loss = loss_train / self.grad_accumulation_steps
                    normalized_loss.backward()  # gradients synced

                    # step - 分别裁剪 action、hdmap 和 VLM 参数的梯度
                    torch.nn.utils.clip_grad_norm_(
                        self.action_expert_params,
                        max_norm=self.max_grad_norm,
                    )
                    if self.hdmap_params:
                        torch.nn.utils.clip_grad_norm_(
                            self.hdmap_params,
                            max_norm=self.max_grad_norm,
                        )
                    if self.train_vlm:
                        torch.nn.utils.clip_grad_norm_(
                            self.vlm_trained_params,
                            max_norm=self.max_grad_norm,
                        )
                    self.action_optimizer.step()
                    self.action_lr_scheduler.step()
                    if self.train_vlm:
                        self.vlm_optimizer.step()
                        self.vlm_lr_scheduler.step()
                    self.action_optimizer.zero_grad(set_to_none=True)
                    if self.train_vlm:
                        self.vlm_optimizer.zero_grad(set_to_none=True)
                    cnt_update += 1

                    # save model at the end of update, models just synced
                    if self.main_rank and (
                        cnt_update % self.save_model_freq == 0
                        or cnt_update == self.n_updates
                    ):
                        self.save_training(cnt_update, cnt_batch)

                # aggregate loss
                if self.multi_gpu:
                    dist.all_reduce(loss_train, op=dist.ReduceOp.SUM)
                    loss_train_deque.append(loss_train.item() / dist.get_world_size())
                else:
                    loss_train_deque.append(loss_train.item())

                # validation with action accuracy
                if self.run_eval and (cnt_batch + 1) % self.eval_freq == 0:
                    new_eval_from_last_log = True
                    eval_metrics_for_log = {}
                    self.model.eval()
                    if self.main_rank:
                        log.info(
                            f"Running evaluation for {self.per_device_num_eval_batch} batches, "
                            f"repeats={self.eval_repeats}, map_modes={self.eval_map_modes}..."
                        )
                    rng_state = self._snapshot_rng_state()
                    eval_results = {}
                    try:
                        with torch.no_grad():
                            for map_mode in self.eval_map_modes:
                                from collections import defaultdict

                                repeat_accuracies = []
                                repeat_l1_losses = []
                                repeat_avg_l2_losses = []
                                bucket_l1_sums = defaultdict(float)
                                bucket_avg_l2_sums = defaultdict(float)
                                bucket_acc_sums = defaultdict(lambda: defaultdict(float))
                                bucket_counts = defaultdict(int)
                                for repeat_idx in range(self.eval_repeats):
                                    eval_acc_correct = torch.zeros(
                                        len(self.eval_thresholds),
                                        device=self.device,
                                    )
                                    eval_acc_count = torch.tensor(
                                        0.0,
                                        device=self.device,
                                    )
                                    eval_l1_sum = torch.tensor(
                                        0.0,
                                        device=self.device,
                                    )
                                    eval_l1_count = torch.tensor(
                                        0.0,
                                        device=self.device,
                                    )
                                    eval_avg_l2_sum = torch.tensor(
                                        0.0,
                                        device=self.device,
                                    )
                                    eval_avg_l2_count = torch.tensor(
                                        0.0,
                                        device=self.device,
                                    )
                                    val_iterator = iter(self.val_dataloader)
                                    for eval_batch_idx in range(
                                        self.per_device_num_eval_batch
                                    ):
                                        try:
                                            batch_eval = next(val_iterator)
                                        except StopIteration:
                                            break
                                        inputs = preprocess_batch(
                                            batch_eval,
                                            split_mask=True,
                                            sample_fm_time=False,
                                        )
                                        gt_trajectory = inputs.pop("actions")
                                        trajectory_bucket = batch_eval.get(
                                            "trajectory_bucket",
                                            None,
                                        )
                                        if trajectory_bucket is not None:
                                            trajectory_bucket = trajectory_bucket.long().to(
                                                self.device,
                                            )
                                        eval_seed = (
                                            self.eval_seed
                                            + repeat_idx * 100000
                                            + eval_batch_idx
                                        )
                                        map_inputs = self._apply_eval_map_mode(
                                            inputs,
                                            map_mode,
                                            eval_seed + 10000000,
                                        )
                                        self._set_eval_seed(eval_seed)
                                        preds = model.infer_action(**map_inputs)
                                        acc_correct, acc_count = get_action_accuracy_counts(
                                            gt_trajectory,
                                            preds,
                                            self.eval_thresholds,
                                        )
                                        eval_acc_correct += acc_correct
                                        eval_acc_count += acc_count

                                        l1_sum, l1_count = get_action_l1_sum_count(
                                            gt_trajectory,
                                            preds,
                                        )
                                        eval_l1_sum += l1_sum
                                        eval_l1_count += l1_count

                                        avg_l2_sum, avg_l2_count = (
                                            get_denormalized_action_avg_l2_sum_count(
                                                gt_trajectory,
                                                preds,
                                                self.eval_avg_l2_normalizer,
                                            )
                                        )
                                        eval_avg_l2_sum += avg_l2_sum
                                        eval_avg_l2_count += avg_l2_count
                                        if trajectory_bucket is not None:
                                            bucket_results = get_bucketed_action_accuracy(
                                                gt_trajectory,
                                                preds,
                                                trajectory_bucket,
                                                thresholds=[0.05, 0.1],
                                                avg_l2_normalizer=self.eval_avg_l2_normalizer,
                                            )
                                            for bname, bmetrics in bucket_results.items():
                                                count = bmetrics["count"]
                                                if count <= 0:
                                                    continue
                                                bucket_l1_sums[bname] += (
                                                    bmetrics["l1"] * count
                                                )
                                                bucket_avg_l2_sums[bname] += (
                                                    bmetrics["avg_l2"] * count
                                                )
                                                bucket_counts[bname] += count
                                                for key, value in bmetrics.items():
                                                    if key.startswith("acc@"):
                                                        bucket_acc_sums[bname][key] += (
                                                            value * count
                                                        )

                                    if self.multi_gpu:
                                        dist.barrier()
                                        dist.all_reduce(
                                            eval_acc_correct,
                                            op=dist.ReduceOp.SUM,
                                        )
                                        dist.all_reduce(
                                            eval_acc_count,
                                            op=dist.ReduceOp.SUM,
                                        )
                                        dist.all_reduce(
                                            eval_l1_sum,
                                            op=dist.ReduceOp.SUM,
                                        )
                                        dist.all_reduce(
                                            eval_l1_count,
                                            op=dist.ReduceOp.SUM,
                                        )
                                        dist.all_reduce(
                                            eval_avg_l2_sum,
                                            op=dist.ReduceOp.SUM,
                                        )
                                        dist.all_reduce(
                                            eval_avg_l2_count,
                                            op=dist.ReduceOp.SUM,
                                        )

                                    eval_accuracy = eval_acc_correct / torch.clamp(
                                        eval_acc_count,
                                        min=1.0,
                                    )
                                    eval_l1_loss = eval_l1_sum / torch.clamp(
                                        eval_l1_count,
                                        min=1.0,
                                    )
                                    eval_avg_l2 = eval_avg_l2_sum / torch.clamp(
                                        eval_avg_l2_count,
                                        min=1.0,
                                    )

                                    repeat_accuracies.append(eval_accuracy.detach())
                                    repeat_l1_losses.append(eval_l1_loss.detach())
                                    repeat_avg_l2_losses.append(eval_avg_l2.detach())

                                if self.multi_gpu:
                                    bucket_keys = ["acc@0.05", "acc@0.1"]
                                    bucket_stats = torch.tensor(
                                        [
                                            [
                                                bucket_l1_sums.get(bname, 0.0),
                                                bucket_avg_l2_sums.get(bname, 0.0),
                                                float(bucket_counts.get(bname, 0)),
                                                *[
                                                    bucket_acc_sums.get(bname, {}).get(
                                                        key,
                                                        0.0,
                                                    )
                                                    for key in bucket_keys
                                                ],
                                            ]
                                            for bname in [
                                                "straight",
                                                "turn",
                                                "static",
                                                "high_curvature",
                                            ]
                                        ],
                                        device=self.device,
                                        dtype=torch.float64,
                                    )
                                    dist.all_reduce(bucket_stats, op=dist.ReduceOp.SUM)
                                    bucket_l1_sums.clear()
                                    bucket_avg_l2_sums.clear()
                                    bucket_acc_sums.clear()
                                    bucket_counts.clear()
                                    for bname, row in zip(
                                        [
                                            "straight",
                                            "turn",
                                            "static",
                                            "high_curvature",
                                        ],
                                        bucket_stats.cpu().tolist(),
                                    ):
                                        bucket_l1_sums[bname] = row[0]
                                        bucket_avg_l2_sums[bname] = row[1]
                                        bucket_counts[bname] = int(row[2])
                                        for key, value in zip(bucket_keys, row[3:]):
                                            bucket_acc_sums[bname][key] = value

                                acc_stack = torch.stack(repeat_accuracies, dim=0)
                                l1_stack = torch.stack(repeat_l1_losses, dim=0)
                                avg_l2_stack = torch.stack(
                                    repeat_avg_l2_losses,
                                    dim=0,
                                )
                                eval_results[map_mode] = {
                                    "acc_mean": acc_stack.mean(dim=0),
                                    "acc_std": (
                                        acc_stack.std(dim=0, unbiased=False)
                                        if acc_stack.shape[0] > 1
                                        else torch.zeros_like(acc_stack[0])
                                    ),
                                    "l1_mean": l1_stack.mean(),
                                    "l1_std": (
                                        l1_stack.std(unbiased=False)
                                        if l1_stack.shape[0] > 1
                                        else torch.zeros_like(l1_stack[0])
                                    ),
                                    "avg_l2_mean": avg_l2_stack.mean(),
                                    "avg_l2_std": (
                                        avg_l2_stack.std(unbiased=False)
                                        if avg_l2_stack.shape[0] > 1
                                        else torch.zeros_like(avg_l2_stack[0])
                                    ),
                                    "bucket": {},
                                }
                                for bname in ["straight", "turn", "static", "high_curvature"]:
                                    count = bucket_counts.get(bname, 0)
                                    if count <= 0:
                                        continue
                                    bucket_metrics = {
                                        "l1": bucket_l1_sums[bname] / count,
                                        "avg_l2": bucket_avg_l2_sums[bname] / count,
                                        "count": count,
                                    }
                                    for acc_key in sorted(
                                        bucket_acc_sums.get(bname, {}).keys()
                                    ):
                                        bucket_metrics[acc_key] = (
                                            bucket_acc_sums[bname][acc_key] / count
                                        )
                                    eval_results[map_mode]["bucket"][bname] = bucket_metrics
                    finally:
                        self._restore_rng_state(rng_state)
                        self.model.train()

                    if self.main_rank:
                        show_eval_std = self.eval_repeats > 1
                        for map_mode, result in eval_results.items():
                            l1_msg = f"{result['l1_mean'].item():.3f}"
                            avg_l2_msg = f"{result['avg_l2_mean'].item():.3f}"
                            if show_eval_std:
                                l1_msg += f"±{result['l1_std'].item():.3f}"
                                avg_l2_msg += (
                                    f"±{result['avg_l2_std'].item():.3f}"
                                )

                            acc_msgs = []
                            for threshold, accuracy, std in zip(
                                self.eval_thresholds,
                                result["acc_mean"],
                                result["acc_std"],
                            ):
                                acc_msg = f"acc thres {threshold}: {accuracy.item():.3f}"
                                if show_eval_std:
                                    acc_msg += f"±{std.item():.3f}"
                                acc_msgs.append(acc_msg)

                            log_msg = (
                                f"Eval [{map_mode}] | l1 Loss: {l1_msg} | "
                                f"Avg. L2: {avg_l2_msg} | "
                            )
                            log_msg += " | ".join(acc_msgs)
                            for bname, bucket_metrics in result["bucket"].items():
                                log_msg += (
                                    f" | {bname}_l1: {bucket_metrics['l1']:.3f}"
                                    f" | {bname}_avg_l2: "
                                    f"{bucket_metrics['avg_l2']:.3f}"
                                )
                                for acc_key in sorted(bucket_metrics.keys()):
                                    if not acc_key.startswith("acc@"):
                                        continue
                                    log_msg += (
                                        f" | {bname}_{acc_key}: "
                                        f"{bucket_metrics[acc_key]:.3f}"
                                    )
                            log.info(log_msg)
                            eval_metrics_for_log[f"eval/{map_mode}/l1_mean"] = (
                                result["l1_mean"].item()
                            )
                            eval_metrics_for_log[f"eval/{map_mode}/l1_std"] = (
                                result["l1_std"].item()
                            )
                            eval_metrics_for_log[f"eval/{map_mode}/avg_l2_mean"] = (
                                result["avg_l2_mean"].item()
                            )
                            eval_metrics_for_log[f"eval/{map_mode}/avg_l2_std"] = (
                                result["avg_l2_std"].item()
                            )
                            for threshold, accuracy, std in zip(
                                self.eval_thresholds,
                                result["acc_mean"],
                                result["acc_std"],
                            ):
                                eval_metrics_for_log[
                                    f"eval/{map_mode}/acc_{threshold}_mean"
                                ] = accuracy.item()
                                eval_metrics_for_log[
                                    f"eval/{map_mode}/acc_{threshold}_std"
                                ] = std.item()
                            for bname, bucket_metrics in result["bucket"].items():
                                eval_metrics_for_log[f"eval/{map_mode}/{bname}_l1"] = (
                                    bucket_metrics["l1"]
                                )
                                eval_metrics_for_log[
                                    f"eval/{map_mode}/{bname}_avg_l2"
                                ] = bucket_metrics["avg_l2"]
                                eval_metrics_for_log[f"eval/{map_mode}/{bname}_count"] = (
                                    bucket_metrics["count"]
                                )
                                for acc_key in sorted(bucket_metrics.keys()):
                                    if acc_key.startswith("acc@"):
                                        eval_metrics_for_log[
                                            f"eval/{map_mode}/{bname}_{acc_key}"
                                        ] = bucket_metrics[acc_key]

                        if "normal" in eval_results:
                            normal_result = eval_results["normal"]
                            eval_metrics_for_log["eval l1 loss"] = normal_result[
                                "l1_mean"
                            ].item()
                            eval_metrics_for_log["eval avg l2"] = normal_result[
                                "avg_l2_mean"
                            ].item()
                            eval_metrics_for_log.update(
                                {
                                    f"eval acc - thres {threshold}": accuracy.item()
                                    for threshold, accuracy in zip(
                                        self.eval_thresholds,
                                        normal_result["acc_mean"],
                                    )
                                }
                            )
                        self.save_training(cnt_update, cnt_batch)

                # log loss
                if self.main_rank and cnt_batch % self.log_freq == 0:
                    loss_train_metric = np.mean(loss_train_deque)
                    peak_vram = torch.cuda.max_memory_reserved(self.gpu_id) / (1024**3)
                    # estimate ETA based on elapsed time since training start
                    updates_done = cnt_update - cnt_update_start
                    if updates_done > 0:
                        elapsed = time.time() - train_start_time
                        secs_per_update = elapsed / updates_done
                        remaining_updates = self.n_updates - cnt_update
                        eta_secs = secs_per_update * remaining_updates
                        eta_h = int(eta_secs // 3600)
                        eta_m = int((eta_secs % 3600) // 60)
                        eta_str = f"{eta_h}h{eta_m:02d}m"
                    else:
                        eta_str = "N/A"
                    log_msg = f"Batch {cnt_batch} Update {cnt_update}: t {timer():8.4f} | vram {peak_vram:6.3f} | train loss {loss_train_metric:6.4f} | action lr {self.action_optimizer.param_groups[0]['lr']:10.8f} | eta {eta_str}"
                    if self.use_separate_hdmap_optimizer_group:
                        log_msg += f" | hdmap lr {self.action_optimizer.param_groups[1]['lr']:10.8f}"
                    log.info(log_msg)
                    # 构建 metrics 字典
                    metrics = {
                        "train/loss": loss_train_metric,
                        "train/gradient_steps": cnt_update,
                        "train/action_lr": self.action_optimizer.param_groups[0]["lr"],
                    }
                    if self.use_separate_hdmap_optimizer_group:
                        metrics["train/hdmap_lr"] = self.action_optimizer.param_groups[1]["lr"]
                    if self.train_vlm:
                        metrics["vlm lr"] = self.vlm_optimizer.param_groups[0]["lr"]
                    
                    if new_eval_from_last_log:
                        metrics.update(eval_metrics_for_log)
                        new_eval_from_last_log = False
                    # if self.use_wandb:
                    #     # 保持原有的 wandb
                    #     wandb.log(metrics, step=cnt_batch, commit=True)
                    # [新增] SwanLab Log
                    if self.use_swanlab:
                        swanlab.log(metrics, step=cnt_batch)

                # Count
                cnt_batch += 1
                if cnt_update >= self.n_updates:
                    return
            # ---- 一个 epoch 结束 ----
            epoch += 1

    @log_execution_time()
    def save_training(self, cnt_update, cnt_batch):
        # TODO(allenzren): skip saving redundant proprio weights
        data = {
            "cnt_update": cnt_update,
            "cnt_batch": cnt_batch,
            "model": (
                self.model.module.state_dict()
                if self.multi_gpu
                else self.model.state_dict()
            ),
            "action_optimizer": self.action_optimizer.state_dict(),
            "vlm_optimizer": self.vlm_optimizer.state_dict()
            if self.train_vlm
            else None,
            "action_lr_scheduler": self.action_lr_scheduler.state_dict(),
            "vlm_lr_scheduler": self.vlm_lr_scheduler.state_dict()
            if self.train_vlm
            else None,
            # "wandb_id": wandb.run.id if self.use_wandb else None,
        }
        savepath = os.path.join(self.checkpoint_dir, f"step{cnt_update}.pt")
        torch.save(data, savepath)
        checkpoint_size_in_gb = os.path.getsize(savepath) / (1024**3)
        log.info(f"Saved model to {savepath}, size: {checkpoint_size_in_gb:.3f} GB")

    @log_execution_time()
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.cnt_update = data["cnt_update"]
        self.cnt_batch = data["cnt_batch"]
        # self.wandb_id = data["wandb_id"]
        # remove "_orig_mod." prefix if saved model was compiled
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }
        self.model.load_state_dict(data["model"], strict=True)
        log.info(
            f"Loaded model from {path} at update {self.cnt_update} batch {self.cnt_batch}"
        )

    @log_execution_time()
    def load_optimizer(self, path):
        """load to cpu first, then move to gpu"""
        from src.utils.optim import optimizer_to

        data = torch.load(path, weights_only=True, map_location="cpu")
        self.action_optimizer.load_state_dict(data["action_optimizer"])
        optimizer_to(self.action_optimizer, self.device)
        self.action_lr_scheduler.load_state_dict(data["action_lr_scheduler"])

        if self.train_vlm:
            self.vlm_optimizer.load_state_dict(data["vlm_optimizer"])
            optimizer_to(self.vlm_optimizer, self.device)
            self.vlm_lr_scheduler.load_state_dict(data["vlm_lr_scheduler"])
        log.info(f"Loaded optimizer and scheduler states from {path}")

    @log_execution_time()
    def _load_model_weights_only(self, path):
        """Finetune 模式：只加载模型权重，不恢复优化器/调度器/训练计数器"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        ckpt = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in ckpt.items():
            if k not in model_state:
                skipped.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped.append(f"[shape mismatch] {k}")
                continue
            filtered[k] = v

        # proprio 因 tie_weights 合并到 action，跳过是正常的
        real_skipped = [s for s in skipped if "mixtures.proprio" not in s]
        if real_skipped:
            log.warning(f"Finetune: skipped {len(real_skipped)} keys: {real_skipped[:5]}")

        missing, _ = self.model.load_state_dict(filtered, strict=False)
        if missing:
            log.warning(f"Finetune: {len(missing)} keys missing from checkpoint (using random init)")

        log.info(f"Finetune: loaded {len(filtered)} / {len(model_state)} model weights from {path}")
        log.info("Optimizer and LR scheduler are freshly initialized (not restored)")
        
