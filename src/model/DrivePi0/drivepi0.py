"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
from typing import Optional, Tuple

import hydra
import torch
from torch import nn

from src.model.kv_cache import KVCache
from src.model.DrivePi0.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
from src.utils.decorator import NoSyncBase
from src.utils.monitor import log_execution_time

from transformers import Qwen3VLVisionModel
import pdb

log = logging.getLogger(__name__)


class DrivePiZero(nn.Module, NoSyncBase):
    @log_execution_time()
    def __init__(self, cfg, use_ddp: bool = False, load_pretrained: bool = True):
        """
        Args:
            load_pretrained: True  = 首次训练，从预训练模型加载权重
                             False = 恢复训练/推理，仅创建架构，权重由 checkpoint 提供
        """
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)
        self.mask_vlm_for_action = cfg.get("mask_vlm_for_action", False)
        self.state_only_for_action = cfg.get("state_only_for_action", False)

        # Trajectory auxiliary losses
        traj_loss_cfg = cfg.get("trajectory_loss", {})
        self.lambda_smooth = traj_loss_cfg.get("lambda_smooth", 0.0)
        self.lambda_accel = traj_loss_cfg.get("lambda_accel", 0.0)
        self.endpoint_weight_mode = traj_loss_cfg.get("endpoint_weight_mode", "none")
        self.endpoint_weight_max = traj_loss_cfg.get("endpoint_weight_max", 3.0)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision + Language：从 Qwen3-VL 构建子模块
        vision_tower, language_model = self._build_qwen3_modules(
            cfg.pretrained_model_path, load_pretrained
        )
        self.vision_tower = vision_tower
        self.multi_modal_projector = None

        # JointModel 直接接收 language_model
        from src.model.DrivePi0.joint_model import JointModel
        self.joint_model = JointModel(
            cfg.joint.config,
            pretrained_language_model=language_model,
        )

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(cfg.time_hidden_size)
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(self.action_hidden_size)
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

    @staticmethod
    def _build_qwen3_modules(model_path: str, load_pretrained: bool):
        """
        从 Qwen3-VL 构建 vision_tower 和 language_model。

        load_pretrained=True  → from_pretrained()  加载架构+预训练权重（首次训练）
        load_pretrained=False → AutoConfig + 空模型，仅创建架构（权重由 checkpoint 提供）
        """
        from transformers import Qwen3VLForConditionalGeneration, AutoConfig

        if load_pretrained:
            full_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.bfloat16,
            )
            log.info("Loaded Qwen3-VL pretrained weights")
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            full_model = Qwen3VLForConditionalGeneration(config)
            log.info("Created Qwen3-VL architecture (weights will come from checkpoint)")

        vision_tower = full_model.model.visual
        language_model = full_model.model.language_model
        # full_model 的其他部分（lm_head 等）不再需要，释放引用
        del full_model
        torch.cuda.empty_cache()
        return vision_tower, language_model

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            # + list(self.multi_modal_projector.parameters())# 去掉 multi_modal_projector，已集成在 vision_tower 内
            + self.trainable_qwen_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        # 去掉 multi_modal_projector，已集成在 vision_tower 内
        # for name, param in self.multi_modal_projector.named_parameters():
        #     if "lora_" in name:
        #         params.append(param)
        params.extend(self.trainable_lora_qwen_parameters)
        return params

    @property
    def trainable_qwen_parameters(self):
        """获取可训练的Qwen参数"""
        trainable_params = []
        for name, param in self.joint_model.named_parameters():
            if "mixtures.vlm" in name and not self._check_qwen_unused_parameter_by_name(name):
                trainable_params.append(param)
        return trainable_params

    @property
    def trainable_lora_qwen_parameters(self):
        """获取可训练的LoRA-Qwen参数"""
        trainable_params = []
        for name, param in self.named_parameters():
            if "lora_" in name and "mixtures.vlm" in name and not self._check_qwen_unused_parameter_by_name(name):
                trainable_params.append(param)
        return trainable_params

    @log_execution_time()
    def load_pretrained_weights(self):
        """vision, projector, lm from qwen3-vl"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k or "model.embed_tokens" in k:
                new_key = k.replace("model.embed_tokens.", "")
                if new_key in embed_tokens_state_dict:
                    embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower
        # vision_tower_state_dict = self.vision_tower.state_dict()
        # for k, v in tensors.items():
        #     if "visual" in k or "vision" in k or "visual_tower" in k:
        #         new_key = k.replace("visual.", "").replace("vision_tower.", "")
        #         if new_key in vision_tower_state_dict:
        #             vision_tower_state_dict[new_key] = v
        # self.vision_tower.load_state_dict(vision_tower_state_dict, strict=False)
        # log.info("Loaded pre-trained weights for vision tower")

        # load projector
        # multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        # for k, v in tensors.items():
        #     if "multi_modal_projector" in k or "img_projection" in k or "visual_projection" in k:
        #         new_key = k.replace("multi_modal_projector.", "").replace("img_projection.", "").replace("visual_projection.", "")
        #         if new_key in multi_modal_projector_state_dict:
        #             multi_modal_projector_state_dict[new_key] = v
        # self.multi_modal_projector.load_state_dict(multi_modal_projector_state_dict, strict=False)
        # log.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in joint_model_state_dict.keys():
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "model" in k and ("layers" in k or "norm" in k):
                new_key = k.replace("model.", "mixtures.vlm.")
                if new_key in joint_model_state_dict:
                    joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_qwen_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
        ):
            return True
        return False

    def inject_lora(self, cfg):
        """Inject LoRA adapters into configurable components.

        Controlled by cfg fields:
          - lora_vlm (bool):                    language model attention (q/k/v/o_proj)
          - lora_vlm_mlp (bool):                language model MLP (gate/up/down_proj)
          - lora_vision (bool):                 vision tower attention (qkv, proj)
          - lora_vision_mlp (bool):             vision tower MLP (linear_fc1/fc2)
          - lora_vision_merger (bool):          main vision-text merger
          - lora_vision_deepstack_merger (bool): deepstack mergers (currently unused by DrivePi0 forward)

        Must be called AFTER loading base weights (for finetune) or BEFORE
        loading a LoRA checkpoint (for resume).
        """
        from src.model.lora import LoRALinear

        r = cfg.lora_r
        lora_alpha = cfg.get("lora_alpha", r)
        lora_dropout = cfg.lora_dropout
        total = 0

        def _wrap(module, attr_name):
            original = getattr(module, attr_name)
            if isinstance(original, nn.Linear):
                setattr(module, attr_name, LoRALinear(
                    original, r=r, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                ))
                return 1
            return 0

        # --- VLM language model ---
        if cfg.get("lora_vlm", False):
            vlm_attn_targets = cfg.get(
                "lora_vlm_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            language_model = self.joint_model.mixtures["vlm"].model
            cnt = 0
            for layer in language_model.layers:
                for name in vlm_attn_targets:
                    if hasattr(layer.self_attn, name):
                        cnt += _wrap(layer.self_attn, name)
            log.info(f"LoRA VLM attention: {cnt} layers (targets={vlm_attn_targets})")
            total += cnt

        if cfg.get("lora_vlm_mlp", False):
            mlp_targets = ["gate_proj", "up_proj", "down_proj"]
            language_model = self.joint_model.mixtures["vlm"].model
            cnt = 0
            for layer in language_model.layers:
                for name in mlp_targets:
                    if hasattr(layer.mlp, name):
                        cnt += _wrap(layer.mlp, name)
            log.info(f"LoRA VLM MLP: {cnt} layers")
            total += cnt

        # --- Vision tower ---
        if cfg.get("lora_vision", False):
            cnt = 0
            for block in self.vision_tower.blocks:
                for name in ["qkv", "proj"]:
                    if hasattr(block.attn, name):
                        cnt += _wrap(block.attn, name)
            log.info(f"LoRA Vision attention: {cnt} layers")
            total += cnt

        if cfg.get("lora_vision_mlp", False):
            cnt = 0
            for block in self.vision_tower.blocks:
                for name in ["linear_fc1", "linear_fc2"]:
                    if hasattr(block.mlp, name):
                        cnt += _wrap(block.mlp, name)
            log.info(f"LoRA Vision MLP: {cnt} layers")
            total += cnt

        if cfg.get("lora_vision_merger", False):
            cnt = 0
            merger = self.vision_tower.merger
            for name in ["linear_fc1", "linear_fc2"]:
                if hasattr(merger, name):
                    cnt += _wrap(merger, name)
            log.info(f"LoRA Vision merger: {cnt} layers")
            total += cnt

        # DrivePi0 currently only consumes vision_output[0] from Qwen3-VL visual.
        # The returned deepstack features are ignored, so their merger parameters
        # would be marked unused by DDP if we injected trainable LoRA adapters here.
        if cfg.get("lora_vision_deepstack_merger", False):
            cnt = 0
            if hasattr(self.vision_tower, "deepstack_merger_list"):
                for ds_merger in self.vision_tower.deepstack_merger_list:
                    for name in ["linear_fc1", "linear_fc2"]:
                        if hasattr(ds_merger, name):
                            cnt += _wrap(ds_merger, name)
            log.info(f"LoRA Vision deepstack merger: {cnt} layers")
            total += cnt

        log.info(f"LoRA injection complete: {total} total layers (r={r}, alpha={lora_alpha})")

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        # for name, param in self.multi_modal_projector.named_parameters():
        #     param.requires_grad = True if "lora_" in name else False
        # log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_qwen_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_qwen_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_vlm_weights(self):
        """Freeze all VLM-related parameters (vision tower, projector, gemma)."""
        # vision encoder
        for _, param in self.vision_tower.named_parameters():
            param.requires_grad = False
        log.info("Froze all weights in vision tower")

        # # vision-text projector
        # for _, param in self.multi_modal_projector.named_parameters():
        #     param.requires_grad = False
        # log.info("Froze all weights in multi_modal_projector")

        # language backbone (gemma in joint_model mixtures["vlm"])
        for _, param in self.joint_model.mixtures["vlm"].named_parameters():
            param.requires_grad = False
        log.info("Froze all weights in VLM (gemma) part of the joint model")

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

        When mask_vlm_for_action=False (default):
                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x

        When mask_vlm_for_action=True (ablation — VLM fully isolated):
                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio                                        x
        action                                          x       x      x
        action                                          x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end   = self.max_image_text_tokens + self.num_proprio_tokens
        action_start  = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            if not self.mask_vlm_for_action:
                causal_mask[idx, proprio_start:, :cnt] = (
                    0  # proprio/action attend to image/text
                )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # ── 3. position ids（不变）──────────────────────────────────────
        vlm_position_ids = (
            torch.arange(0, self.max_image_text_tokens, device=attention_mask.device)
            .unsqueeze(0).expand(bsz, -1)
        )
        proprio_position_ids = (
            torch.arange(0, self.num_proprio_tokens, device=attention_mask.device)
            .unsqueeze(0).expand(bsz, -1)
        )
        action_position_ids = (
            torch.arange(0, self.num_action_tokens, device=attention_mask.device)
            .unsqueeze(0).expand(bsz, -1)
        )

        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def build_state_only_causal_mask_and_position_ids(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        total_tokens = self.num_proprio_tokens + self.num_action_tokens
        proprio_end = self.num_proprio_tokens

        causal_mask = torch.full(
            (batch_size, total_tokens, total_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )

        # proprio -> proprio
        causal_mask[:, :proprio_end, :proprio_end] = 0
        # action -> proprio + action
        causal_mask[:, proprio_end:, :] = 0

        causal_mask = causal_mask.unsqueeze(1)

        vlm_position_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        proprio_position_ids = (
            torch.arange(0, self.num_proprio_tokens, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        action_position_ids = (
            torch.arange(0, self.num_action_tokens, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        if self.state_only_for_action:
            state_mask = causal_mask[
                ...,
                : self.num_proprio_tokens,
                : self.num_proprio_tokens,
            ]
            action_mask = causal_mask[..., -self.num_action_tokens :, :]
            return state_mask, action_mask

        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        bsz = attention_mask.size(0)
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            past_len = kv_cache.num_items()
            position_ids = torch.full(
                (bsz, q_len),
                past_len,
                device=device,
                dtype=torch.long,
            )
        else:
            position_ids = torch.arange(
                0, q_len, device=device
            ).unsqueeze(0).expand(bsz, -1)
        return causal_mask, position_ids

    # ---------- Inference ----------#
    
    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Image_num, Channels, Height, Width] -> [Batch_Size,  Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        _, image_num, _, _, _ = pixel_values.shape
        split_pixel_values = torch.split(pixel_values, pixel_values.size(1) // image_num, dim=1)
        image_tokens = []
        for i, split_tensor in enumerate(split_pixel_values):
            split_tensor = torch.squeeze(split_tensor, 1)
            selected_image_feature = self.vision_tower(split_tensor)
            image_features = self.multi_modal_projector(selected_image_feature)
            image_tokens.append(image_features)
            
        concatenate_image_tokens = torch.cat(image_tokens, dim=1)# [btz, num_images * num_patches, hidden_size]
        
        # normalize the image features
        _, _, embed_dim = concatenate_image_tokens.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = concatenate_image_tokens / (self.image_text_hidden_size**0.5)
        scaled_image_features = scaled_image_features.to(dtype)

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )
        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding
    
    def _forward_qwen3_vision_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        dtype = self.embed_tokens.weight.dtype
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        # 1. 文本嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        final_embedding = inputs_embeds.clone()

        # 2. 视觉特征：官方 vision_tower 内部已包含 merger(projector)
        # 输入: pixel_values [total_patches, patch_dim], image_grid_thw [num_images, 3]
        # 输出: 返回 (hidden_states, deepstack_feature_lists) 元组
        if pixel_values is not None and pixel_values.numel() > 0:
            vision_output = self.vision_tower(
                pixel_values.to(dtype),
                grid_thw=image_grid_thw,
            )
            # import pdb; pdb.set_trace()
            image_features = vision_output[0]
            image_features = image_features.to(dtype)

            # 3. 替换 <|image_pad|> 位置
            image_token_mask = (input_ids == self.image_token_index)
            num_masked = image_token_mask.sum().item()
            num_features = image_features.shape[0]

            if num_masked != num_features:
                raise ValueError(
                    f"<|image_pad|> token数 ({num_masked}) 与视觉特征数 ({num_features}) 不匹配"
                )

            final_embedding[image_token_mask] = image_features.to(dtype)

        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        # vlm_causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = proprios.dtype, proprios.device
        bsz = proprios.size(0)

        if self.state_only_for_action:
            kv_caches = {"proprio": KVCache()}

            proprio_embeds = self.proprio_encoder(proprios)

            _, kv_caches = self.joint_model(
                attention_mask=image_text_proprio_mask,
                position_ids_all={
                    "proprio": proprio_position_ids,
                },
                embeds_all={
                    "proprio": proprio_embeds,
                },
                kv_caches=kv_caches,
                cache_mode="append",
                return_caches=True,
            )

            action = torch.randn(
                (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
            )

            delta_t = 1.0 / self.num_inference_steps
            t = torch.zeros(bsz, device=device, dtype=dtype)
            for _ in range(self.num_inference_steps):
                time_cond = self.time_embedding(t)
                if self.action_expert_adaptive_mode:
                    action_embeds = self.action_encoder(action)
                else:
                    action_embeds = self.action_encoder(action, time_cond)

                action_embeds = self.joint_model(
                    attention_mask=action_mask,
                    position_ids_all={"action": action_position_ids},
                    embeds_all={"action": action_embeds},
                    time_cond=time_cond,
                    kv_caches=kv_caches,
                    cache_mode="append_non_active",
                )["action"]

                action_vel = self.action_decoder(action_embeds)
                action += delta_t * action_vel
                t += delta_t

            if self.final_action_clip_value is not None:
                action = torch.clamp(
                    action,
                    -self.final_action_clip_value,
                    self.final_action_clip_value,
                )
            return action

        kv_caches = self.joint_model.build_mixture_caches()
        # print("pixel_values.shape:", pixel_values.shape)
        # print("image_grid_thw.shape:", image_grid_thw.shape)  

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_qwen3_vision_and_text_embedding(
            input_ids, pixel_values, image_grid_thw=image_grid_thw  # 传递图像网格信息以正确处理视觉输入
        )

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            # vlm_attention_mask=vlm_causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            cache_mode="append",
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                # vlm_attention_mask=vlm_causal_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = input_ids.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_qwen3_vision_and_text_embedding(
            input_ids, pixel_values, image_grid_thw=image_grid_thw  # 传递图像网格信息以正确处理视觉输入
        )

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_qwen3_vision_and_text_embedding(
            input_ids, pixel_values, image_grid_thw=image_grid_thw  # 传递图像网格信息以正确处理视觉输入
        )

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        image_grid_thw: torch.LongTensor,
        causal_mask: torch.FloatTensor,
        # vlm_causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)

        if self.state_only_for_action:
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "proprio": proprio_embeds,
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches={},  # no caching during training
            )["action"]
        else:
            # text tokens + image tokens
            inputs_embeds = self._forward_qwen3_vision_and_text_embedding(
                input_ids,
                pixel_values,
                image_grid_thw=image_grid_thw,
            )

            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds,
                    "proprio": proprio_embeds,
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches={},  # no caching during training
            )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        residual = v_psi - d_psi  # [B, H, A]

        # --- distance-weighted MSE: give higher weight to farther points ---
        if self.endpoint_weight_mode == "linear":
            # weight linearly increases: 1.0, 1+k, 1+2k, ..., endpoint_weight_max
            H = residual.shape[1]
            w = torch.linspace(1.0, self.endpoint_weight_max, H,
                               device=residual.device, dtype=residual.dtype)
            point_weights = w[None, :, None]  # [1, H, 1]
            mse_loss = torch.mean(residual ** 2 * point_weights)
        else:
            mse_loss = torch.mean(residual ** 2)

        loss = mse_loss

        # --- smoothness / accel loss on RECONSTRUCTED trajectory, not raw velocity ---
        # v_psi predicts d_psi = x1 - (1-σ_min)*x0.  Recover implied x1:
        #   x1_hat = psi_t + (1-t) * v_psi  (Euler single-step estimate)
        # This removes the per-point noise x0 that makes raw v_psi non-smooth.
        if self.lambda_smooth > 0 or self.lambda_accel > 0:
            t_expand = t[:, None, None]  # [B, 1, 1]
            x1_hat = psi_t + (1 - t_expand) * v_psi

        if self.lambda_smooth > 0:
            traj_diff = x1_hat[:, 1:, :] - x1_hat[:, :-1, :]
            gt_diff = x1[:, 1:, :] - x1[:, :-1, :]
            smooth_loss = torch.mean((traj_diff - gt_diff) ** 2)
            loss = loss + self.lambda_smooth * smooth_loss

        if self.lambda_accel > 0 and x1.shape[1] >= 3:
            traj_accel = x1_hat[:, 2:, :] - 2 * x1_hat[:, 1:-1, :] + x1_hat[:, :-2, :]
            gt_accel = x1[:, 2:, :] - 2 * x1[:, 1:-1, :] + x1[:, :-2, :]
            accel_loss = torch.mean((traj_accel - gt_accel) ** 2)
            loss = loss + self.lambda_accel * accel_loss

        return loss.to(v_psi.dtype)


class DrivePiZeroInference(DrivePiZero):
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return super().infer_action(
            input_ids,
            pixel_values,
            image_grid_thw,
            image_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )


if __name__ == "__main__":
    import argparse
    import time

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from src.model.DrivePi0.processing import VLAProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_pretrained_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = OmegaConf.load("config/train/DrivePi0/base.yaml")
    device = "cpu" if args.cpu else "cuda"
    model = DrivePiZero(config)
    model.tie_action_proprio_weights()
    if args.load_pretrained_weights:
        model.load_pretrained_weights()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model.to(device)
    model.to(dtype)
    model.eval()
    print(f"Using {device} and {dtype}...")

    # dummy image --- replace the first image with a real one
    bsz = 2
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 3, 224, 224), dtype=torch.uint8
    ) # 3 images in total

    # text and proprio
    dummy_texts = [
        "Please predict future trajectory ",
        "The future trajectory is ",
    ][:bsz]
    dummy_proprio = torch.rand(bsz, config.cond_steps, config.state_dim)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # processor
    num_image_tokens = config.vision.config.num_image_tokens
    processor = VLAProcessor(tokenizer, num_image_tokens, config.max_seq_len)

    # process image and text
    model_inputs = processor(text=dummy_texts, images=dummy_images)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"].to(dtype)

    # inference
    start_time = time.time()
    if args.loss_only:
        dummy_actions = torch.randn(bsz, config.horizon_steps, config.action_dim)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        t = torch.rand(bsz)
        with torch.inference_mode():
            loss = model(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                causal_mask=causal_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
                actions=dummy_actions.to(dtype).to(device),
                t=t.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Loss:", loss)
    else:  # dummy action generation
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        with torch.inference_mode():
            actions = model.infer_action(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Final action dimensions:", actions.shape)
        print("Final action values:", actions)
    print("Time taken:", time.time() - start_time)
    print("============================\n\n")
