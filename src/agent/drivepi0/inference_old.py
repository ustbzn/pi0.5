import logging
import os

import hydra
import einops
import imageio
from tqdm import *
import numpy as np
import torch
from transformers import AutoTokenizer

from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)

class DrivePiZeroInferenceAgent:
    def __init__(self, cfg):
        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.model = DrivePiZeroInference(cfg, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        print('Agent done')
    
    
    def run(self, data_dict):
        images_front = data_dict["image_front"]
        images_front = images_front.unsqueeze(0)
        images_front_history = data_dict["image_front_history"]
        images_front_history = images_front_history.unsqueeze(0)
        images_back = data_dict["image_back"]
        images_back = images_back.unsqueeze(0)
            
        state = data_dict["state"].unsqueeze(0)
        texts = [data_dict["language_instruction"]]
            
        images_front = einops.rearrange(images_front, "B H W C -> B C H W")  # remove cond_steps dimension
        images_front = images_front.unsqueeze(1)
        images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")  # remove cond_steps dimension
        images_front_history = images_front_history.unsqueeze(1)
        images_back = einops.rearrange(images_back, "B H W C -> B C H W")  # remove cond_steps dimension
        images_back = images_back.unsqueeze(1)
            
        # images = torch.cat((images_front, images_front_history, images_back), dim=1)
        images = torch.cat((images_front, images_front_history), dim=1)
        model_inputs = self.processor(text=texts, images=images)

        # build causal mask and position ids for trajectory
        causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": state_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": state.to(self.dtype),
        }
                
        image_text_proprio_mask, action_mask = (
            self.model.split_full_mask_into_submasks(causal_mask)
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        trajectory = self.model.infer_action(**inputs)
        return trajectory
    
    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from typing import Sequence, Optional
    import os
    import torch
    import pickle
    import argparse
    import numpy as np
    from PIL import Image
    import json
    import tensorflow as tf
    from src.data.utils.augmentations import augment_image
    from src.data.utils.normalization import Normalize
    from src.data.utils.image import read_resize_encode_image_pytorch, concatenate_images, image_normalization
    parser = argparse.ArgumentParser() # usage: python src/agent/drivepi0/inference.py --test_data_path "YOUR_PKL_FILE_PATH"
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/eval/DrivePi0/base.yaml")
    parser.add_argument("--statistic_path", type=str, default="statistics.json")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)
    statistic_path = args.statistic_path
    inference_agent = DrivePiZeroInferenceAgent(config)
    
    # prepare data for testing
    pkl_data_path = args.test_data_path
    with open(pkl_data_path, 'rb') as f:
        all_data = pickle.load(f)
        # prepare state
        normalize = Normalize.get_instance(statistic_path)
        state = normalize.prepare_state(all_data)
        # prepare 
        data_dict = {}
        image_front_path = all_data['his_cam_front'].numpy().decode('utf-8')
        data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
        image_back_path = all_data['his_cam_back'].numpy().decode('utf-8')
        data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
        image_front_time_path = all_data['his_cam_front_time'].numpy().decode('utf-8')
        data_dict['image_front_history'] = image_normalization(image_front_time_path, (224, 224))
    
        # prepare text
        data_dict["language_instruction"] = "predict trajectory"

        data_dict['state'] = state
        trajectory = inference_agent.run(data_dict=data_dict)
        x_pred, y_pred = normalize.infer_traj(trajectory)
        print('x:', x_pred)
        print('y:', y_pred)