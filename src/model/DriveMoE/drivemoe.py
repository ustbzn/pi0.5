import time
import torch
import hydra
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.model.DriveMoE.router import Router
from src.model.DrivePi0.drivepi0 import DrivePiZero
from src.model.DrivePi0.processing import VLAProcessor
from src.model.DriveMoE.mixture import MixtureDecoderLayer

log = logging.getLogger(__name__)

class DriveMoE(DrivePiZero):
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__(cfg, use_ddp)
        
        self.camera_router = Router(
            token_dim=cfg.vision_projector.config.vision_config.projection_dim, 
            num_labels=cfg.num_camera_views_selected,
        )
        self.view_embeddings = nn.Embedding(
            cfg.num_camera_views_selected, 
            cfg.vision_projector.config.vision_config.projection_dim
        )
        
        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.stage = cfg.stage
        
    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + list(self.camera_router.parameters())
            + list(self.view_embeddings.parameters())
            + self.trainable_gemma_parameters
        )
        
    @property
    def _action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
            + list(self.joint_model.action_router.parameters())
        )
        

    @property
    def proprio_expert_parameters(self):
        return (
            list(self.joint_model.mixtures["proprio"].parameters())
        )

    @property
    def action_expert_parameters(self):
        param_dict = {}
        
        for param in self._action_expert_parameters:
            param_dict[id(param)] = param
        
        for param in self.proprio_expert_parameters:
            param_dict[id(param)] = param
        
        return list(param_dict.values())
        
    def tie_action_proprio_weights(self):
        action_layers = self.joint_model.mixtures["action"].layers
        proprio_layers = self.joint_model.mixtures["proprio"].layers
        
        for i, (action_layer, proprio_layer) in enumerate(zip(action_layers, proprio_layers)):
            if type(action_layer) == type(proprio_layer) == MixtureDecoderLayer:
                proprio_layers[i] = action_layer
        
    def _forward_with_view_pe(self, final_tokens, selected_idx):
        view_enc = self.view_embeddings(selected_idx) 
        final_tokens = final_tokens + view_enc.unsqueeze(1)
        return final_tokens
        
    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        waypoints: torch.Tensor,
        camera_id: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        assert pixel_values.shape[1] == 7, "7 images are needed in DriveMoE"

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)
        bs, seq_len = input_ids.shape

        # image features from siglip and projector
        if self.training and camera_id is None:
            # stage2 training mode: need to calculate all of the vision tokens of the 7 images here
            _, image_num, _, _, _ = pixel_values.shape
            split_pixel_values = torch.split(pixel_values, pixel_values.size(1) // image_num, dim=1)
            image_tokens = []
            for split_tensor in split_pixel_values:
                split_tensor = torch.squeeze(split_tensor, 1)
                selected_image_feature = self.vision_tower(split_tensor)
                image_features = self.multi_modal_projector(selected_image_feature)
                image_tokens.append(image_features.unsqueeze(1))
            concatenate_image_tokens = torch.cat(image_tokens, dim=1)   # [bs, num_images, num_patches, hidden_size]
            # normalize the image features
            scaled_image_features = concatenate_image_tokens / (self.image_text_hidden_size**0.5)
            scaled_image_features = scaled_image_features.to(dtype)
            cam_front_features = scaled_image_features[:, 0, :, :]
            cam_front_his_features = scaled_image_features[:, 1, :, :]
            candidate_features = scaled_image_features[:, 2:, :, :]
            
            # go through camera router
            camera_logits = self.camera_router(cam_front_features, waypoints)
            soft_weights = F.gumbel_softmax(camera_logits, tau=1.0, hard=True)
            selected_idx = torch.argmax(camera_logits, dim=-1)
            selected_tokens = torch.einsum('bk,bktn->btn', soft_weights, candidate_features)
            selected_tokens = self._forward_with_view_pe(selected_tokens, selected_idx)
            cat_image_features = torch.cat((cam_front_features, cam_front_his_features, selected_tokens), dim=1)

        else:
            # inference mode or stage1 training mode: only calculate selected tokens    
            _, image_num, _, _, _ = pixel_values.shape
            split_pixel_values = torch.split(pixel_values, pixel_values.size(1) // image_num, dim=1)
            image_tokens = []
            for split_tensor in split_pixel_values[:2]:
                split_tensor = torch.squeeze(split_tensor, 1)
                selected_image_feature = self.vision_tower(split_tensor)
                image_features = self.multi_modal_projector(selected_image_feature)
                image_tokens.append(image_features.unsqueeze(1))
            concatenate_image_tokens = torch.cat(image_tokens, dim=1)
            # normalize the image features
            scaled_image_features = concatenate_image_tokens / (self.image_text_hidden_size**0.5)
            scaled_image_features = scaled_image_features.to(dtype)
            cam_front_features = scaled_image_features[:, 0, :, :]
            cam_front_his_features = scaled_image_features[:, 1, :, :]
            
            # go through camera router
            camera_logits = self.camera_router(cam_front_features, waypoints)
            stacked_pixel_values = torch.stack(split_pixel_values, dim=0)[2:]
            stacked_pixel_values = stacked_pixel_values.transpose(0, 1)
            selected_idx = torch.argmax(camera_logits, dim=-1)
            if camera_id is not None and self.training:
                # stage1 training mode
                selected_idx = camera_id.long()
            batch_indices = torch.arange(bs, device=device)
            final_pixel_values = stacked_pixel_values[batch_indices, selected_idx]
            
            split_tensor = torch.squeeze(final_pixel_values, 1)
            selected_image_feature = self.vision_tower(split_tensor)
            image_features = self.multi_modal_projector(selected_image_feature)
            scaled_image_features = image_features / (self.image_text_hidden_size**0.5)
            scaled_image_features = scaled_image_features.to(dtype)
            scaled_image_features = self._forward_with_view_pe(scaled_image_features, selected_idx)
            cat_image_features = torch.cat((cam_front_features, cam_front_his_features, scaled_image_features), dim=1)

        # put embedding together - image, text, padding
        embed_dim = cat_image_features.shape[-1]
        final_embedding = torch.full(
            (bs, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )
        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bs):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = cat_image_features[
                i, :num_image_tokens
            ]
        return final_embedding, camera_logits
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
        waypoints: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
        scenario_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        if self.stage == 2:
            inputs_embeds, camera_logits = self._forward_siglip_and_text_embedding(input_ids, pixel_values, waypoints, None)
        else:
            inputs_embeds, camera_logits = self._forward_siglip_and_text_embedding(input_ids, pixel_values, waypoints, camera_ids)

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
        embeds, action_logits = self.joint_model(
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
            action_logits=None,
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )
        action_embeds = embeds["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)
        
        # compute total loss
        return self.criterion(
            flow_sig_min=self.flow_sig_min,
            x0=x0,
            x1=x1,
            v_psi=v_psi,
            camera_logits=camera_logits,
            camera_ids=camera_ids,
            action_logits=action_logits,
            scenario_ids=scenario_ids,
        )
    
    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        waypoints: torch.Tensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds, camera_logits = self._forward_siglip_and_text_embedding(input_ids, pixel_values, waypoints)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass thru the vlm and proprio, cache the kv
        _, action_logits, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            action_logits=None,
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        action_logits_list = []
        
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
            embeds, action_logits = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                action_logits=None,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )
            action_embeds = embeds["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t
            action_logits_list.append(action_logits)

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action, camera_logits, action_logits_list
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_pretrained_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = OmegaConf.load("config/train/DriveMoE/test_stage1.yaml")
    device = "cpu" if args.cpu else "cuda"
    model = DriveMoE(config)
    # model.tie_action_proprio_weights()
    if args.load_pretrained_weights:
        model.load_pretrained_weights()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model.to(device)
    model.to(dtype)
    print(f"Using {device} and {dtype}...")

    # dummy images
    bs = 2
    dummy_images = torch.randint(
        0, 256, (bs, 7, 3, 224, 224), dtype=torch.uint8
    ) # 7 images in total

    # text and proprio
    dummy_texts = [
        "Predict Trajectory.",
        "The future trajectory is ",
    ]
    dummy_proprio = torch.rand(bs, config.cond_steps, config.proprio_dim)
    waypoints = torch.rand(bs, 2)
    camera_ids = torch.tensor([3, 2])
    scenario_ids = torch.tensor([2, 3])

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
        dummy_actions = torch.randn(bs, config.horizon_steps, config.action_dim)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        t = torch.rand(bs)
        total_loss, action_loss, camera_loss, scenario_loss = model(
            input_ids=input_ids.to(device),
            pixel_values=pixel_values.to(dtype).to(device),
            causal_mask=causal_mask.to(device),
            vlm_position_ids=vlm_position_ids.to(device),
            proprio_position_ids=proprio_position_ids.to(device),
            action_position_ids=action_position_ids.to(device),
            proprios=dummy_proprio.to(dtype).to(device),
            actions=dummy_actions.to(dtype).to(device),
            t=t.to(dtype).to(device),
            waypoints=waypoints.to(dtype).to(device),
            camera_ids=camera_ids.to(dtype).to(device),
            scenario_ids=scenario_ids.to(dtype).to(device),
        )
        print("total loss:", total_loss)
        print("action loss:", action_loss)
        print("camera loss:", camera_loss)
        print("scenario loss:", scenario_loss)
        # check grad
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        optimizer.zero_grad()
        action_loss.backward()
        camera_router_grad = model.camera_router.mlp[-1].weight.grad
        if camera_router_grad is None or torch.all(camera_router_grad == 0):
            print("No grad in camera router, this is not expected when camera_ids is None!")
        action_router_grad = model.joint_model.action_router.router_linear.weight.grad
        if action_router_grad is None or torch.all(action_router_grad == 0):
            print("No grad in action router, this is not expected!")
    else:
        model.eval()
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        with torch.inference_mode():
            actions, _, _ = model.infer_action(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
                waypoints=waypoints.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Final action dimensions:", actions.shape)
    print("Time taken:", time.time() - start_time)
    print("============================\n\n")