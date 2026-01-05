## Training
If you don't need to train the __DrivePi0__ or __DriveMoE__ models from scratch and wish to use our pre-trained models directly, you may skip ahead to the [Evaluation](evaluation.md) section.
### Drive $\pi_0$
Edit the following key parameters in the [config file](/config/train/DrivePi0/base.yaml)
```yaml
current_dir: YOUR_REPO_PATH         # You need to set path, eg: .../DriveMoE
pretrained_model_path: PALIGEMMA
resume_checkpoint_path: CKPT_PATH   # If you want to resume from checkpoint
```
Then run the script to start training:
```bash
bash script/training/train_drivepi0.sh
```