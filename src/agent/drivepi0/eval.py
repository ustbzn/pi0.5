import torch
import einops
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.agent.dataset import Bench2DriveDataset
from src.utils.metric import get_action_accuracy
from src.data.utils.normalization import Normalize
from src.model.DrivePi0.processing import VLAProcessor
from src.model.DrivePi0.drivepi0 import DrivePiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)


class DrivePiZeroEvalAgent:
    def __init__(self, cfg):
        self.log_dir = cfg.log_dir
        self.config = cfg
        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.model = DrivePiZeroInference(cfg, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        
        # dataloader 
        self.val_dataloader = DataLoader(
            Bench2DriveDataset(cfg.data).dataset, 
            batch_size=cfg.device_batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=cfg.num_workers
        )
        self.eval_thresholds = cfg.eval_thresholds
        
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

    def run(self):
        cnt_episode = 0
        successes = []
        
        eval_accuracy = torch.zeros(len(self.eval_thresholds), device=self.device)
        eval_l1_loss = torch.tensor(0.0, device=self.device)
        eval_l2_loss = torch.tensor(0.0, device=self.device)
        
        def preprocess_batch(batch, split_mask: bool, sample_fm_time: bool):
            images_front = batch["image_front"]
            images_front_history = batch["image_front_time"]
            
            state = batch["state"]
            trajectory = batch["trajectory"].squeeze(1)  # remove the time dimension
            texts = batch["language_instruction"]
            images_front = einops.rearrange(images_front, "B H W C -> B C H W")  # remove cond_steps dimension
            images_front = images_front.unsqueeze(1)
            images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")  # remove cond_steps dimension
            images_front_history = images_front_history.unsqueeze(1)
            
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
                "actions": trajectory.to(self.dtype),
            }
            if split_mask:
                image_text_proprio_mask, action_mask = (
                    self.model.split_full_mask_into_submasks(causal_mask)
                )
                inputs["image_text_proprio_mask"] = image_text_proprio_mask
                inputs["action_mask"] = action_mask
            else:
                inputs["causal_mask"] = causal_mask

            # sample flow matching timesteps
            if sample_fm_time:
                inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        
        cnt = 0
        print('Running evaluation ... ')
            
        for batch in tqdm(self.val_dataloader):
            inputs = preprocess_batch(batch, split_mask=True, sample_fm_time=False)
            gt_trajectory = inputs.pop("actions")
            preds = self.model.infer_action(**inputs)
            eval_accuracy += get_action_accuracy(gt_trajectory, preds, self.eval_thresholds)
            normalize = Normalize.get_instance('config/statistics/b2d_statistics.json')
            pred_trajectory = []
            label_trajectory = []
            for pred in preds:
                pred = pred.unsqueeze(0)
                x_pred, y_pred = normalize.infer_traj(pred) # use true trajectory to calculate l1 loss and l2 loss
                pred = torch.stack([x_pred, y_pred], dim=1)
                pred = pred.unsqueeze(0)
                pred_trajectory.append(pred)
            for label in gt_trajectory:
                label = label.unsqueeze(0)
                x_label, y_label = normalize.infer_traj(label)
                label = torch.stack([x_label, y_label], dim=1)
                label = label.unsqueeze(0)
                label_trajectory.append(label)
            
            pred_trajectory = torch.cat(pred_trajectory, dim=0)
            label_trajectory = torch.cat(label_trajectory, dim=0)
            cnt += 1
            
            eval_l1_loss += torch.nn.functional.l1_loss(pred_trajectory, label_trajectory)
            eval_l2_loss += torch.nn.functional.mse_loss(pred_trajectory, label_trajectory)
            
        eval_accuracy = eval_accuracy / cnt
        eval_l1_loss = eval_l1_loss / cnt
        eval_l2_loss = eval_l2_loss / cnt
        
        
        # summary
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
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")