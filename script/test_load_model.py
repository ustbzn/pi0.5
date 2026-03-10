import torch
from omegaconf import OmegaConf
from src.model.DrivePi0.drivepi0 import DrivePiZero

# 加载配置
cfg = OmegaConf.load("config/train/DrivePi0/qwen3.yaml")
# 初始化模型
model = DrivePiZero(cfg)
# 加载权重
model.load_pretrained_weights()
# 打印模型结构
print("Model loaded successfully!")
print(f"VLM layers: {model.joint_model.num_hidden_layers}")
print(f"Action layers: {model.joint_model.num_action_layers}")
print(f"Vision hidden size: {model.vision_tower.config.hidden_size}")

import sys

# 将模型结构输出到txt
with open("Qwen3VL.txt", "w", encoding="utf-8") as f:
    # 基本信息
    f.write("=" * 60 + "\n")
    f.write("Model Basic Info\n")
    f.write("=" * 60 + "\n")
    f.write(f"VLM layers: {model.joint_model.num_hidden_layers}\n")
    f.write(f"Action layers: {model.joint_model.num_action_layers}\n")
    f.write(f"Vision hidden size: {model.vision_tower.config.hidden_size}\n\n")

    # 完整模型结构
    f.write("=" * 60 + "\n")
    f.write("Full Model Structure\n")
    f.write("=" * 60 + "\n")
    f.write(str(model) + "\n\n")

    # 各模块参数量统计
    f.write("=" * 60 + "\n")
    f.write("Parameter Count per Module\n")
    f.write("=" * 60 + "\n")
    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        f.write(f"{name}: {params:,} params\n")
    f.write(f"\nTotal: {total_params:,} params\n")

print("Model structure saved to Qwen3VL.txt")