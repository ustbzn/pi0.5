import torch
import einops
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.metric import get_action_accuracy
from src.model.DriveMoE.drivemoe import DriveMoE
from src.agent.dataset import Bench2DriveDataset
from src.data.utils.normalization import Normalize
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)

class DriveMoEEvalAgent:
    def __init__(self, cfg):
        self.log_dir = cfg.log_dir
        self.config = cfg
        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.model = DriveMoE(cfg)
        self.load_checkpoint(cfg)
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

    def run(self):
        eval_accuracy = torch.zeros(len(self.eval_thresholds), device=self.device)
        eval_l1_loss = torch.tensor(0.0, device=self.device)
        eval_l2_loss = torch.tensor(0.0, device=self.device)
        
        all_camera_true = []
        all_camera_pred = []
        all_scenario_true = []
        all_scenario_pred = []
        
        def preprocess_batch(batch):
            images_front = batch["image_front"]
            images_front_history = batch["image_front_time"]
            images_front_left = batch["image_front_left"]
            images_front_right = batch["image_front_right"]
            images_back = batch["image_back"]
            images_back_left = batch["image_back_left"]
            images_back_right = batch["image_back_right"]
            
            state = batch["state"]
            trajectory = batch["trajectory"].squeeze(1)  # remove the time dimension
            texts = batch["language_instruction"]
            waypoints = batch["waypoints"]
            camera_ids = batch.get("cam_id", None)
            scenario_ids = batch.get("scenario_id", None)
            images_front = einops.rearrange(images_front, "B H W C -> B C H W")
            images_front = images_front.unsqueeze(1)
            images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")
            images_front_history = images_front_history.unsqueeze(1)
            images_front_left = einops.rearrange(images_front_left, "B H W C -> B C H W")
            images_front_left = images_front_left.unsqueeze(1)
            images_front_right = einops.rearrange(images_front_right, "B H W C -> B C H W")
            images_front_right = images_front_right.unsqueeze(1)
            images_back = einops.rearrange(images_back, "B H W C -> B C H W")
            images_back = images_back.unsqueeze(1)
            images_back_left = einops.rearrange(images_back_left, "B H W C -> B C H W")
            images_back_left = images_back_left.unsqueeze(1)
            images_back_right = einops.rearrange(images_back_right, "B H W C -> B C H W")
            images_back_right = images_back_right.unsqueeze(1)
            images = torch.cat((
                images_front, 
                images_front_history,
                images_front_left,
                images_front_right,
                images_back,
                images_back_left, 
                images_back_right
            ), dim=1)
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
                "waypoints": waypoints.to(self.dtype),
                "camera_ids": camera_ids,
                "scenario_ids": scenario_ids,
            }
            image_text_proprio_mask, action_mask = (
                self.model.split_full_mask_into_submasks(causal_mask)
            )
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
                        
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        
        cnt = 0
        print('Running evaluation ... ')
            
        for batch in tqdm(self.val_dataloader):
            inputs = preprocess_batch(batch)
            gt_trajectory = inputs.pop("actions")
            cam_id_gt = inputs.pop("camera_ids")
            scenario_id_gt = inputs.pop("scenario_ids")
            preds, camera_logits, scenario_logits = self.model.infer_action(**inputs)
            
            camera_selected_idx = torch.argmax(camera_logits, dim=-1)
            camera_true = cam_id_gt.cpu().numpy()
            camera_pred = camera_selected_idx.cpu().numpy()
            all_camera_true.extend(camera_true)
            all_camera_pred.extend(camera_pred)
            
            final_scenario_logits = scenario_logits[-1] 
            scenario_selected_idx = torch.argmax(final_scenario_logits, dim=-1)
            all_scenario_true.extend(scenario_id_gt.cpu().numpy())
            all_scenario_pred.extend(scenario_selected_idx.cpu().numpy())
            
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
        
        camera_names = ['Front Left', 'Front Right', 'Back', 'Back Left', 'Back Right']
        scenario_names = ['Merging', 'Parking Exit', 'Overtaking', 'Emergency Brake', 'Giveway', 'Traffic Sign', 'Normal']
        
        print("\n" + "="*20 + " Camera Classification Report " + "="*20)
        camera_report_dict = classification_report(
            all_camera_true, 
            all_camera_pred, 
            target_names=camera_names,
            output_dict=True,
            zero_division=0,
        )
        
        print(classification_report(
            all_camera_true, 
            all_camera_pred, 
            target_names=camera_names,
        ))
        
        log.info(f"# camera summary")
        log.info(f"Number of samples: {len(all_camera_true)}")
        
        log_msg = f"Eval | Acc: {camera_report_dict['accuracy']:.3f} | "
        log_msg += f"Macro F1: {camera_report_dict['macro avg']['f1-score']:.3f} | "

        category_f1s = []
        for label in camera_names:
            f1 = camera_report_dict[label]['f1-score']
            category_f1s.append(f"{label} F1: {f1:.3f}")

        log_msg += " | ".join(category_f1s)
    
        print("\n" + "="*20 + " Scenario Classification Report " + "="*20)
        scenario_report_dict = classification_report(
            all_scenario_true, 
            all_scenario_pred, 
            target_names=scenario_names,
            output_dict=True,
            zero_division=0
        )
        print(classification_report(
            all_scenario_true, 
            all_scenario_pred, 
            target_names=scenario_names,
        ))
        
        log.info(f"# scenario summary")
        log.info(f"Number of samples: {len(all_scenario_true)}")
        
        log_msg = f"Eval | Acc: {scenario_report_dict['accuracy']:.3f} | "
        log_msg += f"Macro F1: {scenario_report_dict['macro avg']['f1-score']:.3f} | "

        category_f1s = []
        for label in scenario_names:
            f1 = scenario_report_dict[label]['f1-score']
            category_f1s.append(f"{label} F1: {f1:.3f}")

        log_msg += " | ".join(category_f1s)
        
        log_msg += f"\nEval | l1 Loss: {eval_l1_loss.item():.3f} | "
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
    def load_checkpoint(self, cfg):
        """load to cpu first, then move to gpu"""
        path = cfg.checkpoint_path
        data = torch.load(path, weights_only=True, map_location="cpu")
        assert data["stage"] == 2, "The ckpt is not stage2 ckpt!"
        assert data["horizon_steps"] == cfg.horizon_steps, \
            f"Mismatched horizon_steps! Checkpoint has {data['horizon_steps']}, but config has {cfg.horizon_steps}"
        assert data["cond_steps"] == cfg.cond_steps, \
            f"Mismatched cond_steps! Checkpoint has {data['cond_steps']}, but config has {cfg.cond_steps}"
        
        # Check and warn about horizon_steps
        if data["horizon_steps"] != 20:
            log.warning("=" * 80)
            log.warning(f"⚠️  HORIZON_STEPS = {data['horizon_steps']} (not 20) ⚠️")
            log.warning("=" * 80)
            log.warning("❗ COMPARISON WITH OTHER BASELINES MAY BE UNFAIR ❗")
            log.warning("   L1 and L2 losses are NOT directly comparable")
            log.warning("   to baselines trained with horizon_steps = 20")
            log.warning("=" * 80)    
        
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")