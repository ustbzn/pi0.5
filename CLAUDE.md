# CLAUDE.md

本文件是 Claude Code（claude.ai/code）在本仓库工作时的操作指南与规范契约。请在生成代码、修改配置或运行脚本前阅读本文件。

## 项目概览

仓库名：**DriveMoE / Drive-π0（DrivePi_8GPU）**。这是一个基于 Vision-Language-Action（VLA）范式的自动驾驶轨迹预测项目，输入多相机图像 + 自车状态 + 导航指令（+ 可选 HD 地图），输出未来轨迹（默认 6 步，每步 [x, y]）。

- 主分支：`master`；当前开发分支：`v2.1_qwen3_hdmap`
- 项目根目录：`/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU`
- Python：`==3.10.*`（见 `pyproject.toml`）
- 默认对话语言：**中文（简体）**

支持两套数据：
- **MOGO**：内部真车 ROS bag → pkl 流水线，是当前主战场（路径含 `mogo_dataset_*`）
- **nuScenes**：通过 `nuscene-dataset-tools/` 转换为同样的 pkl 格式

## 技术栈

- 训练框架：PyTorch 2.5.0、`torchrun` 多卡 DDP、`bitsandbytes` 8-bit AdamW
- 配置：Hydra + OmegaConf；所有训练 / 推理入口都通过 `--config-path` + `--config-name` 加载 yaml
- 模型骨架：HuggingFace `transformers`（`Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`），早期版本走 `paligemma + siglip`
- 实验记录：SwanLab（默认开启，`SWANLAB_API_KEY` 写在训练脚本里），WandB 已禁用 `WANDB_DISABLED=true`
- 精度：`use_bf16: True` 默认，`use_amp: True`，可选 `use_torch_compile`（compile 时跳过 `vision_tower`，因含 `.item()` 调用）
- 数据预处理：旧链路用 `tensorflow==2.15.0`/`tensorflow_datasets`，新 MOGO 链路只用 numpy + pkl（实际 dataset 走 `torch.utils.data.Dataset`）
- 完整依赖：见 `pyproject.toml`

## 顶层目录速查

```
config/
  train/DrivePi0/        # 训练 yaml（Hydra），命名形如 0430_qwen3_hdmap_C1C2_8gpu.yaml
  inference/DrivePi0/    # 推理 yaml
  eval/DrivePi0/         # 评测 yaml
  test_dataloader/       # dataloader 单元自测配置
script/
  run.py                 # 唯一训练入口（Hydra @main）
  training/*.sh          # torchrun 多卡训练脚本
  evaluation/*.sh        # 评测脚本（含 AB 诊断）
  visualization/*.sh     # 轨迹可视化 + 合成视频
  data_preprocessing/*.sh# get_statistics / window 等
  analysis/              # 离线分析工具（如 hard case 权重）
src/
  agent/drivepi0/
    train.py             # DrivePiZeroTrainAgent（训练主类）
    inference_mogo_test.py  # MOGO 推理入口（脚本式 CLI，非 Hydra）
    inference_nuscene*.py   # nuScenes 推理
    eval.py              # 评测 Agent
  data/
    dataset.py           # TorchDataset / prepare_b2d_dataset
    camera_scenario_map.py
    utils/{image,normalization,augmentations,hdmap_features,hdmap_vector}.py
  model/
    DrivePi0/drivepi0.py     # 顶层 VLA 模型（DrivePiZero / DrivePiZeroInference）
    DrivePi0/joint_model.py  # mixture-of-experts 联合 attention
    DrivePi0/mixture.py      # action / proprio mixture（Qwen3 风格 layer）
    DrivePi0/qwen3mixture.py # 把 Qwen3VLTextModel 适配成 mixture（VLMMixtureAdapter）
    DrivePi0/map_encoder.py  # PlanVOCA-style 结构化车道 token 编码（方案 D）
    DrivePi0/hdmap_encoder.py# 轻量 BEV CNN 编码（方案 C）
    DrivePi0/processing*.py  # VLAProcessor（兼容旧 PaliGemma）
    qwen3/                   # 自实现的 Qwen3 模块（RMSNorm / RotaryEmbed / MLP）
    paligemma/               # 旧 PaliGemma + SigLIP（仍可被 base.yaml 引用）
    lora.py / kv_cache.py / router.py / utils.py
  utils/
    analyzing_traj_mogo.py   # MOGO 可视化 + Avg.L2 评测主脚本
    analyzing_traj.py        # nuScenes 可视化（含 --num_workers 多进程）
    metric.py                # acc@thr / L1 / 反归一化 Avg.L2
    optim.py                 # CosineAnnealingWarmupRestarts
    merge_images.py          # 帧序列合成视频/GIF
mogo-dataset-tools/      # MOGO bag → json → pkl + HDMap 切片
nuscene-dataset-tools/   # nuScenes → pkl 转换
docs/                    # 架构 / 训练 / 评估 / HDMap 设计文档（中文）
logs/                    # 训练 / 推理 / 评测产物（重要：不要清理用户在跑的目录）
swanlog/                 # SwanLab 本地缓存
inference_out/           # 推理产物
visual_out/              # 可视化产物
```

## VLA 模型架构（`DrivePiZero`）

核心入口：`src/model/DrivePi0/drivepi0.py`，类 `DrivePiZero`（训练）/ `DrivePiZeroInference`（推理）。

```
input  ──┬─► Vision Tower (Qwen3VLVisionModel，from_pretrained)
         │       └──► token 序列（image_grid_thw 决定）
         ├─► Text  embed_tokens (Qwen3 vocab=151936)
         │       └──► 二者拼成 VLM mixture 输入
         ├─► proprios (B, cond_steps, proprio_dim) → proprio_encoder(Linear)
         │       └──► 可选: HDMap CNN 特征拼接 / MapEncoder lane token 拼接
         └─► actions (B, horizon, action_dim=2) → ActionEncoder + Time embedding
                 (Flow Matching 训练，Beta(α=1.5,β=1) 采样 t)

JointModel (mixture-of-experts 联合 attention，按层联合 forward)
  ├── mixtures["vlm"]    = VLMMixtureAdapter(Qwen3VLTextModel)  ← 复用预训练权重
  ├── mixtures["proprio"]= Mixture(hidden=1024, intermediate=4096)
  └── mixtures["action"] = Mixture(hidden=1024, intermediate=4096)
        因果 mask: VLM ↔ proprio(+map) ↔ action 之间方向受控
        位置编码: VLM 走官方 3D M-RoPE (build_qwen3_vl_position_ids)
                  proprio / action 走 1D position_ids

action_decoder = Linear(action_hidden, action_dim)
loss = flow-matching MSE
       + 可选 trajectory_loss.lambda_smooth/lambda_accel（速度场平滑/加速度二阶差）
       + 可选 endpoint_weight_mode=linear（远端点加权）
       + 可选 aux_loss（maneuver/stop/endpoint/progress/curvature 5 个 head）
```

关键开关（顶层 cfg 字段）：
- `train_vlm` / `lora` 及 `lora_vlm{,_mlp}` / `lora_vision{,_mlp,_merger}`：是否训 VLM、是否注 LoRA
- `state_only_for_action`：完全屏蔽视觉文本，只用 proprio（消融用）
- `mask_vlm_for_action`：保留 VLM forward，但 action/proprio 不 attend VLM token（推理时也可用 `--mask_vlm`）
- `use_hdmap_image`（方案 A）：BEV PNG 作为第 9 张图喂入 vision tower
- `use_hdmap_proprio`（方案 B）：从 NPZ 抽取标量 HDMap 特征拼到 state
- `use_hdmap_cnn`（方案 C）：下采样 BEV → `HDMapCNNEncoder(in=4)` → fuse 到 proprio（`hdmap_fusion_mode: concat | gated_residual`）
- `use_hdmap_vector`（方案 D，PlanVOCA-style）：从 pkl `hdmap_vector_*` 字段读取 lane polygon → `MapEncoder` → 拼到 proprio mixture 序列后
- `flow_sampling: beta|uniform`，`num_inference_steps: 10`，`final_action_clip_value: 1.0`

参数分组与优化器（`src/agent/drivepi0/train.py`）：
- action expert / map encoder / aux heads → `action_optimizer`（`bnb.optim.AdamW8bit`，`action_lr`）
- HDMap CNN 相关参数可单独成组：`separate_hdmap_optimizer_group: True` + `hdmap_lr` / `hdmap_weight_decay`
- VLM（仅在 `train_vlm: True` 时）→ `vlm_optimizer`（`vlm_lr`）
- 调度器：`CosineAnnealingWarmupRestarts`（`src/utils/optim.py`）

加载模式（互斥）：
- `resume_checkpoint_path`：完整恢复（权重 + 优化器 + 进度）
- `finetune_checkpoint_path`：只加权重，优化器/进度从头
- 都为空 + `load_pretrained_weights: True`：从 `pretrained_model_path` 加载 Qwen3-VL 预训练

## 数据加载

入口：`src/agent/dataset.py:Bench2DriveDataset` → `src/data/dataset.py:prepare_b2d_dataset` → `TorchDataset`。

- 数据形态：每个样本是一个 `.pkl`，由 `mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py` 生成
- 训练 / 验证：`{dynamic_dir}/train` 与 `{dynamic_dir}/val` 下平铺一堆 pkl，`split` 字段决定走哪个
- 多相机字段：`cam_front` / `cam_left` / `cam_right` / `cam_back`（含 `_prev_1` 历史帧），新格式还兼容旧 `his_cam_*`
- 自车状态字段：`his_speed` / `his_acceleration` / `his_angular_velocity` / `his_theta` / `his_x` / `his_y` / `navigation_id`（或从 `navigation_command` 推断）；`proprio_dim` 由 yaml 决定
- 标签字段：`fur_trajectory`（[H, 2]，列序 `[y, x]`）
- HDMap：`hdmap_bev`（指向 NPZ 路径）+ `hdmap_vector_*`（PlanVOCA-style 结构化字段）
- 归一化：`src/data/utils/normalization.py:Normalize` 单例；`norm_mode ∈ {percentile, percentile_clip, minmax}`，统计文件格式为 `statistics.json`
- 辅助标签：`build_aux_targets` 从 `fur_trajectory` 自动派生 `maneuver_cls/stop_cls/endpoint/progress/curvature` 与 `trajectory_bucket`（用于分桶 eval 与方案 D）
- 加权采样：
  - `trajectory_weighted_sampling.enable: True`：按转弯/静止/直行加权（`turn_weight/static_weight/straight_weight`）
  - `nav_weighted_sampling: True`：按 `navigation_id` 反频次加权
  - 多卡 + 加权：自定义 `DistributedWeightedSampler`，所有 rank 用同一 generator 切片，避免重叠
- 语言指令：从 `navigation_command_descreption` 拼接，HDMap 启用时自动追加 BEV 上下文短语

> ⚠️ `Normalize.get_instance(...)` 是单例。**变更 statistics 路径或 norm_mode 时务必清理单例（重启进程），否则推理与训练归一化会错位**。

## 入口脚本与命令

下列五个入口是仓库的"主接口"，凡是改动训练/推理/评测/可视化/数据生成行为，请先回到这五处看是否需要同步更新：

### 1. 训练入口 `script/run.py`
Hydra 主程序。所有训练 yaml 都通过它启动；`_target_` 指向 `src.agent.drivepi0.train.DrivePiZeroTrainAgent`。多卡通过 `torchrun` 注入 `LOCAL_RANK / RANK / WORLD_SIZE`，单进程时 `gpu_id=0`。

常用 wrapper：
```bash
# MOGO 主训练（自动检测 GPU 数量，standalone 模式）
sh script/training/train_drivepi0_mogo.sh
# 编辑该脚本中的 BASE_CONFIG 来切换 yaml；脚本会 export LOG_DIR / SWANLAB_API_KEY / PYTHONPATH

# 调试训练（覆盖 lr=1e-8、n_updates=1，用于快速跑通流水线）
sh script/train_debug.sh

# Phase 1/2 MoE 训练
sh script/training/train_drivepi0_action_moe_phase1.sh
sh script/training/train_drivepi0_image_moe_phase2.sh
```

直接命令行：
```bash
export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU:$PYTHONPATH"
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs"
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=8 --standalone \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=0430_qwen3_hdmap_C1C2_8gpu.yaml \
  +n_train=59778 +n_epochs=40   # Hydra 字段覆盖
```

Checkpoint 落盘到 `${LOG_DIR}/train/${name}/${run_name}/checkpoint/step{cnt_update}.pt`，包含 `model / action_optimizer / vlm_optimizer / *_lr_scheduler / cnt_update / cnt_batch`。

### 2. 推理入口 `src/agent/drivepi0/inference_mogo_test.py`
**注意**：这是脚本式 CLI（`argparse`），**不是** Hydra。命令：
```bash
CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_mogo_test.py \
  --mode batch \
  --val_data_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/val \
  --config_path config/inference/DrivePi0/0514_qwen3_hdmap_all_features_8gpu.yaml \
  --statistic_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics.json \
  --output_dir inference_out/<run>/inference \
  [--max_samples N]                # 调试用
  [--rank 0 --world_size 8]        # 多卡数据切片，外层用 for 循环 + CUDA_VISIBLE_DEVICES
  [--zero_his_traj] [--mask_vlm]   # 诊断开关
```

8 卡并行模板：`script/inference.sh`（每张卡跑一个进程，按 rank 切分 val 集，结果合并到同一 `OUTPUT_DIR`）。

输出：`<output_dir>/<timestamp>_pred_trajectories.npy` 与 `<timestamp>_gt_trajectories.npy`（**已反归一化**），供可视化/评测复用。

### 3. 评测脚本 `script/evaluation/eval_mogo_action_avg_l2.sh`
两步流水：先用 `inference_mogo_test.py` 推理 → 再调 `src/utils/analyzing_traj_mogo.py --mode error` 计算真实坐标系 Avg.L2。
```bash
# 默认：跑推理 + 出 Avg.L2 报告
bash script/evaluation/eval_mogo_action_avg_l2.sh

# 只重算指标（需先有 trajectories/*.npy）
bash script/evaluation/eval_mogo_action_avg_l2.sh analyze_only

# 用环境变量覆盖
GPU_ID=1 CHECKPOINT_PATH=/path/to/step.pt CONFIG_PATH=... \
  bash script/evaluation/eval_mogo_action_avg_l2.sh
```
关键约束：**statistics 必须与训练时一致**（脚本注释明确提示），否则反归一化会有恒定坐标偏置；`norm_mode` 自动从 inference yaml 读取，回退默认 `minmax`。

AB 诊断：`script/evaluation/eval_mogo_ab_diagnostic.sh`（baseline / zero_his_traj / mask_vlm / ablate_both 四组实验，用于定位"自车不主动启动"等行为问题）。

### 4. 可视化脚本 `script/visualization/visual.sh`
封装 `src/utils/analyzing_traj_mogo.py`，模式：
- `--mode dual`：BEV 双视图（GT + Pred）
- `--mode error`：评测出 Avg.L2 报告
- `--show_objects`：叠加感知目标
- 单 scene 用 `--scene_names 20260312161041`，全部 scene 直接省略

合成视频：`script/visualization/video.sh` → `src/utils/merge_images.py`（支持 `--format gif/mp4`，`--fps`，单 scene `--scene`）。

旧 nuScenes 链路：`src/utils/analyzing_traj.py`（带 `--num_workers` 多进程）。

### 5. 训练数据生成 `mogo-dataset-tools/3_processAllData.sh`
当前主流水线：从 ROS bag 直接生成带 HDMap 的 pkl。
```bash
python3 mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py \
  --rosbags-root  /pcpt/pcpt/data/VLA/MOGO/rosbags/0313-1 \
  --images-root   /pcpt/pcpt/data/VLA/MOGO/images/0313-1 \
  --output-root   /pcpt/pcpt/data/VLA/MOGO/0312_0319pkl/0313-1 \
  --hdmap-sql     /pcpt/pcpt/data/VLA/MOGO/hm_beijing_3.4.9_20260521.sqlite \
  --hdmap-utm-zone 50
```
辅助：`1_general_json.sh`（py2.7 解 bag → json）、`2_general_pkl.sh`（json → pkl）、`move_pkl_files.py`（按 scene 划分 train/val）、`hdmap_vector_generator.py`（生成 PlanVOCA-style lane token）。

生成统计文件（归一化所需）：
```bash
sh script/data_preprocessing/get_statistics.sh
# 实际调用 src/data_processing/get_statistics.py，按 percentile / minmax 双写到 statistics.json
```
旧 b2d 数据集还需窗口化：`sh script/data_preprocessing/window.sh`（`--window_size` / `--horizon` 必须与 yaml 中 `cond_steps / horizon_steps` 一致）。

## 配置约定

- **配置命名**：训练 yaml 用 `MMDD_<结构关键词>_<GPU 数>gpu.yaml`，例如 `0430_qwen3_hdmap_C1C2_8gpu.yaml`。新增配置请保留这种前缀，便于按时间索引。
- **路径**：`current_dir` / `pretrained_model_path` / `dynamic_dir` / `statistics_path` 都是绝对路径。修改 yaml 时**先确认目标机器上路径存在**，避免硬塞他人的本地路径。
- **batch 与更新次数**：`global_batch_size = per_device_batch_size * num_gpus`，`n_updates` 用 `${eval:'((${n_train} + ${global_batch_size} - 1) // ${global_batch_size}) * ${n_epochs}'}` 自动算，**改 GPU 数或 per_device_batch_size 必须同步检查 `n_train`**。
- **token 预算**：`max_seq_len` 必须能容纳"图像 token + 文本 token + map_token + proprio + action"，HDMap 方案 A 增加 1 张图像 → 至少留 64~88 token；方案 D 默认 `max_map_tokens: 128`。
- **state_dim**：必须与 `Normalize.prepare_state` 实际拼接维度一致，常见构成在每个 yaml 注释里有列：例如 `proprio_dim=7  # his_speed(2)+his_acceleration(2)+his_angular_velocity(1)+his_theta(1)+navigation_id(1)`。
- **归一化模式**：`percentile` / `percentile_clip` / `minmax`。**训练与推理必须一致**，否则反归一化产生坐标偏置。
- **预训练模型路径**：Qwen3-VL-2B 权重默认在 `/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct`；旧 PaliGemma 配置走 `/D2/.../paligemma-3b-pt-224`。

## Claude Code 在本仓库的操作规范

1. **写代码 = 改正确的 yaml 而非加 if 分支**。新功能优先暴露成 yaml 字段（与现有 HDMap A/B/C/D 方案的开关风格一致）；`drivepi0.py` / `train.py` 已有大量 `cfg.get("xxx", default)` 入口，沿用同样的"开关 + 默认 False"模式即可。
2. **跑长时间命令前确认 GPU 占用**。本机是共享多卡环境，启动训练或多卡推理前用 `nvidia-smi` 确认空闲 GPU；脚本里默认 `nvidia-smi --list-gpus | wc -l` 取卡数，实际使用要靠 `CUDA_VISIBLE_DEVICES` 隔离。
3. **不要随意 push、不要直接 commit**。本仓库 `git status` 长期带很多 `??` 与 `M`（实验性产物如 `*.pkl` / `output*.txt` / `model_problem.txt`）。除非用户明确要求，**不要**做 `git add .` / commit / push；提交时显式列出文件名。
4. **保护他人的 logs**。`logs/train/...` 与 `logs/eval/...` 下经常有别人正在跑的目录；删除前先看时间戳与 `step*.pt` 是否仍在被写入。
5. **改 dataset 字段务必同步三处**：`src/data/dataset.py:TorchDataset.__getitem__`、`src/data/utils/normalization.py:Normalize.prepare_state` 和对应训练 yaml 的 `proprio_dim / state_dim`。三者维度对不齐会以 `RuntimeError: shape mismatch` 在第 1 步训练就炸。
6. **HDMap 方案不要混用没声明的两路**。`use_hdmap_image / proprio / cnn / vector` 各自独立；C 方案 `hdmap_fusion_mode` 只支持 `concat | gated_residual`，加新模式需要改 `_fuse_hdmap_feat`。
7. **推理脚本是 argparse、不是 Hydra**：传 yaml 路径用 `--config_path`，需要覆盖字段时先在 yaml 里改、或者像 `eval_mogo_action_avg_l2.sh` 那样写一个临时 yaml；不要尝试用 `+key=value` Hydra 语法。
8. **`Normalize` 是单例**：在同一进程里改 statistics 必须重启或显式 `Normalize._instance = None`，否则后续读到的是首次加载的值。
9. **torch.compile 限制**：`vision_tower` 不参与 compile（含 `.item()`）；新增模型分支若要 compile，照 `train.py` 200~228 行只对 `joint_model / action_encoder / action_decoder / proprio_encoder` 单独 compile。
10. **加 dependency 谨慎**：`pyproject.toml` 锁定了 `numpy==1.26.4 / protobuf==3.20.3 / tensorflow==2.15.0 / torch==2.5.0`，不要随手升级；新增包尽量加到 `dependencies` 末尾并保持顺序。
11. **commit / PR 信息用中文**：仓库历史 commit 多为中文短句（如 `inference debug`、`add yaml`、`obj visual`），保持风格。
12. **不要新建 markdown 文档**：`docs/` 下已经有详细的中文设计文档（DrivePi0 架构、HDMap 方案、评估流程、训练流程等）。除非用户明确要求，回答里不要为了"留档"而写新的 .md。

## 测试与自检命令

仓库**没有**统一的 `pytest` / `unittest` 测试套件；自检方式是跑下面这些"短任务"：

- **dataloader 自测**：`python src/agent/dataset.py`（依赖 `config/test_dataloader/*.yaml`，会逐 batch 打印 shape）
- **快速训练流水线**：`sh script/train_debug.sh`（`n_updates=1` + `eval_freq=1`，验证模型构建 / 数据流 / 评测路径全通）
- **少量样本推理**：在 `inference_mogo_test.py` 加 `--max_samples 8` 跑一遍，确认 `*_pred_trajectories.npy` 形状正确
- **小规模评测**：`MAX_SAMPLES=32 bash script/evaluation/eval_mogo_action_avg_l2.sh`（带 max_samples 走完整 inference + 分析链路）
- **lint**：`pyproject.toml` 配置了 ruff（`select = [A,B,E,F,I,RUF,W]`，`line-length = 88`）；可以 `ruff check src` 但仓库当前并未在 CI 中强制
- **加载预训练权重**：`python script/test_load_model.py`

> 不存在的命令请不要瞎编（`pytest` / `make test` 在本仓库都不会生效）。

## 关键已知坑位

- **statistics 错配 → 评测坐标恒偏置**：`eval_mogo_action_avg_l2.sh` 的注释里反复强调，新增数据集时优先用同一份 statistics，或重训。
- **`his_trajectory` shortcut**：模型容易把 his_trajectory 直接外推当成预测；`zero_his_traj` 诊断开关就是为此而生。新增依赖历史轨迹的特征前先跑一次 AB 诊断。
- **VLM 实际贡献**：训练时 `train_vlm: False` + `mask_vlm_for_action: False` 是常见组合，但 mask 后指标变化不大常常意味着 action expert 没真的用上 VLM——评估新结构时记得跑 `mask_vlm` 对照。
- **map_token 与 max_seq_len**：`use_hdmap_vector` 启用后 `total_num_tokens = max_image_text_tokens + cond_steps + max_map_tokens + horizon_steps`，需检查 `joint_model.max_position_embeddings` 是否够用。
- **Qwen3-VL `image_grid_thw`**：图像 token 数由 processor 动态决定，`build_qwen3_vl_position_ids` 依赖它生成 3D M-RoPE；自定义图像输入（如方案 A 的 BEV）若改了 patch size，需要同步检查 token 数与 attention mask。

## 当前活跃方向（来自 git status / 文档）

- `v2.1_qwen3_hdmap` 分支正在迭代 HDMap A/B/C/D 四个方案的对比，最新工作在 `0430_qwen3_hdmap_C1C2_8gpu.yaml`、`0514_qwen3_hdmap_all_features_8gpu.yaml`、`0528_qwen3_hdmap_D_vector_8gpu.yaml`。
- `docs/` 下新增的 `HDMap_ABC方案问题分析与改动建议.md` / `planvoca_style_structured_map_token_design.md` / `moe_design_v2_适配方案.md` 是当前正在推进的设计文档，改对应模块前先读。
- 评测产物固定在 `logs/eval/drive-pi0/mogo_action_avg_l2/<run>/`，不要把临时实验写进同一目录。

