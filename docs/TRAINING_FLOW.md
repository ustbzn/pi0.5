# DrivePi0 训练流程详解

## 目录
1. [训练启动流程](#训练启动流程)
2. [数据集读取（PKL文件）](#数据集读取pkl文件)
3. [数据预处理](#数据预处理)
4. [模型前向传播](#模型前向传播)
5. [损失计算与优化](#损失计算与优化)
6. [训练循环](#训练循环)

---

## 训练启动流程

### 1.1 入口点
**文件**: `DriveMoE/script/run.py`

```python
# 使用 Hydra 管理配置
@hydra.main(config_path=..., config_name="base.yaml")
def main(cfg: OmegaConf):
    _main(cfg)
```

### 1.2 初始化流程

```
run.py::_main()
  ├─ 检测GPU数量，判断是否多GPU训练
  ├─ 如果是多GPU: 初始化分布式训练 (DDP)
  │   └─ torch.distributed.init_process_group(backend="nccl")
  ├─ 设置 gpu_id 和 multi_gpu 标志
  └─ 创建并运行训练Agent
      └─ agent = DrivePiZeroTrainAgent(cfg)
      └─ agent.run()
```

### 1.3 启动方式

**单GPU调试**:
```bash
python script/run.py --config-path=../config/train/DrivePi0 --config-name=base_debug.yaml
```

**多GPU训练** (SLURM):
```bash
torchrun --nproc_per_node=2 script/run.py --config-path=../config/train/DrivePi0 --config-name=base.yaml
```

---

## 数据集读取（PKL文件）

### 2.1 PKL文件结构

PKL文件包含以下字段（参考 `src/data/dataset.py` 和 `src/data_processing/get_statistics.py`）：

```python
all_data = {
    # 历史状态信息 (cond_steps=5 个时间步)
    'his_speed': np.array,              # 速度历史 [5]
    'his_acceleration': np.array,        # 加速度历史 [5]
    'his_angular_velocity': np.array,   # 角速度历史 [5]
    'his_theta': np.array,              # 角度历史 [5]
    
    # 目标点信息
    'x_command_far': np.array,          # 远目标点x坐标 [horizon_steps=10]
    'y_command_far': np.array,          # 远目标点y坐标 [horizon_steps=10]
    'x_command_near': np.array,          # 近目标点x坐标 [horizon_steps=10]
    'y_command_near': np.array,         # 近目标点y坐标 [horizon_steps=10]
    
    # 轨迹信息 (horizon_steps=10 个未来时间步)
    'fur_x': np.array,                  # 未来x坐标 [horizon_steps=10]
    'fur_y': np.array,                  # 未来y坐标 [horizon_steps=10]
    
    # 图像路径 (字符串，需要解码)
    'his_cam_front': bytes,              # 前摄像头历史图像路径
    'his_cam_front_time': bytes,         # 前摄像头时间序列图像路径
    'his_cam_back': bytes,               # 后摄像头历史图像路径
    'his_cam_front_left': bytes,         # 前左摄像头图像路径
    'his_cam_front_right': bytes,        # 前右摄像头图像路径
    'his_cam_back_left': bytes,          # 后左摄像头图像路径
    'his_cam_back_right': bytes,         # 后右摄像头图像路径
    'skill_cam': bytes,                  # 技能摄像头图像路径 (固定摄像头模式)
    
    # 其他信息
    'cam_id': bytes,                    # 摄像头ID (动态摄像头模式)
}
```

### 2.2 数据集加载流程

**文件**: `src/data/dataset.py::TorchDataset`

```python
# 1. 初始化时扫描目录
self.list_data_dict = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]

# 2. __getitem__ 时加载单个PKL文件
def __getitem__(self, i):
    pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
    with open(pkl_data_path, 'rb') as f:
        all_data = pickle.load(f)  # 加载PKL文件
        
    # 3. 数据归一化处理
    normalize = Normalize.get_instance(self.statistics_path)
    state = normalize.prepare_state(all_data)      # 归一化状态
    trajectory = normalize.prepare_traj(all_data)  # 归一化轨迹
    
    # 4. 图像加载
    if use_fixed_images:
        # 固定摄像头模式: 使用 his_cam_front, his_cam_front_time, skill_cam
        image_front = image_normalization(image_front_path, (224, 224))
        image_front_time = image_normalization(image_front_time_path, (224, 224))
    else:
        # 动态摄像头模式: 使用多个摄像头视角
        image_front = image_normalization(his_cam_front, (224, 224))
        image_back = image_normalization(his_cam_back, (224, 224))
        # ... 其他摄像头
    
    # 5. 返回数据字典
    return {
        'state': state,                    # [cond_steps, state_dim=10]
        'trajectory': trajectory,          # [horizon_steps=10, action_dim=2]
        'image_front': image_front,        # [H=224, W=224, C=3]
        'image_front_time': image_front_time,  # [H=224, W=224, C=3]
        'language_instruction': 'predict trajectory',
        'cam_id': cam_id,                  # 摄像头ID
        'scenario_id': scenario_id,        # 场景ID
    }
```

### 2.3 数据集路径结构

```
work_dir/
├── b2d_dynamic_camera/          # 动态摄像头数据集
│   ├── train/
│   │   └── *.pkl               # 训练集PKL文件
│   └── val/
│       └── *.pkl               # 验证集PKL文件
└── b2d_fixed_camera/           # 固定摄像头数据集
    └── *.pkl                   # PKL文件
```

### 2.4 统计信息文件 (statistics.json)

用于数据归一化，包含各字段的1%和99%分位数：

```json
{
    "speed": [percentile_01, percentile_99],
    "acceleration": [percentile_01, percentile_99],
    "angular_velocity": [percentile_01, percentile_99],
    "theta": [percentile_01, percentile_99],
    "fur_x": [percentile_01, percentile_99],
    "fur_y": [percentile_01, percentile_99],
    "command_far_x": [percentile_01, percentile_99],
    "command_far_y": [percentile_01, percentile_99],
    "command_near_x": [percentile_01, percentile_99],
    "command_near_y": [percentile_01, percentile_99]
}
```

---

## 数据预处理

### 3.1 状态归一化 (`prepare_state`)

**文件**: `src/data/utils/normalization.py`

将原始状态数据归一化到 [-1, 1] 范围：

```python
def prepare_state(self, all_data):
    # 归一化历史状态 (cond_steps=5)
    his_speed = standard_normalize(all_data['his_speed'], speed_01, speed_99)
    his_acceleration = standard_normalize(all_data['his_acceleration'], ...)
    his_angular_velocity = standard_normalize(...)
    his_theta = standard_normalize(...)
    
    # 归一化目标点 (horizon_steps=10)
    target_far_x = standard_normalize(all_data['x_command_far'], ...)
    target_far_y = standard_normalize(all_data['y_command_far'], ...)
    
    # 拼接成状态向量 [cond_steps, state_dim=10]
    state = torch.tensor([
        his_speed,              # [5]
        his_acceleration,       # [5]
        his_angular_velocity,   # [5]
        his_theta,             # [5]
        target_far_x,          # [10]
        target_far_y,          # [10]
    ])  # -> [5, 10] (实际是 [cond_steps, state_dim])
```

**state_dim = 10** 的组成：
- `his_speed`: 1维
- `his_acceleration`: 1维  
- `his_angular_velocity`: 1维
- `his_theta`: 1维
- `target_far_x`: 1维
- `target_far_y`: 1维
- (其他维度根据实际配置)

### 3.2 轨迹归一化 (`prepare_traj`)

```python
def prepare_traj(self, all_data):
    # 归一化未来轨迹 (horizon_steps=10, action_dim=2)
    fur_x = standard_normalize(all_data['fur_x'], fur_x_01, fur_x_99)  # [10]
    fur_y = standard_normalize(all_data['fur_y'], fur_y_01, fur_y_99)  # [10]
    
    # 拼接成轨迹 [horizon_steps, action_dim]
    trajectory = torch.tensor([fur_x, fur_y]).transpose(1, 0)  # [10, 2]
    return trajectory
```

### 3.3 批次预处理 (`preprocess_batch`)

**文件**: `src/agent/drivepi0/train.py::preprocess_batch()`

```python
def preprocess_batch(batch, split_mask: bool, sample_fm_time: bool):
    # 输入batch结构:
    #   batch['state']: [B, cond_steps=5, state_dim=10]
    #   batch['trajectory']: [B, 1, horizon_steps=10, action_dim=2]
    #   batch['image_front']: [B, cond_steps=5, H=224, W=224, C=3]
    #   batch['image_front_time']: [B, cond_steps=5, H=224, W=224, C=3]
    #   batch['language_instruction']: List[str] (B个字符串)
    
    # 1. 处理图像: 合并 front 和 front_time
    images_front = einops.rearrange(images_front, "B H W C -> B C H W")
    images_front_history = einops.rearrange(images_front_history, "B H W C -> B C H W")
    images = torch.cat((images_front, images_front_history), dim=1)  # [B, 2, C, H, W]
    
    # 2. 文本和图像处理 (VLAProcessor)
    model_inputs = processor(text=texts, images=images)
    # 返回: {
    #   'input_ids': [B, max_seq_len=532],
    #   'attention_mask': [B, max_seq_len],
    #   'pixel_values': [B, num_images=2, C=3, H=224, W=224]
    # }
    
    # 3. 构建因果掩码和位置ID
    causal_mask, vlm_position_ids, state_position_ids, action_position_ids = \
        model.build_causal_mask_and_position_ids(model_inputs["attention_mask"], dtype)
    
    # 4. 采样Flow Matching时间步 (训练时)
    if sample_fm_time:
        t = sample_fm_time(batch_size)  # [B] 在 [0, 1] 之间
    
    # 5. 组装输入字典
    inputs = {
        'input_ids': model_inputs["input_ids"],
        'pixel_values': model_inputs["pixel_values"].to(dtype),
        'vlm_position_ids': vlm_position_ids,
        'proprio_position_ids': state_position_ids,
        'action_position_ids': action_position_ids,
        'proprios': state.to(dtype),           # [B, cond_steps, state_dim]
        'actions': trajectory.to(dtype),       # [B, horizon_steps, action_dim]
        't': t.to(dtype),                      # [B] Flow Matching时间步
        'causal_mask': causal_mask,            # 或 split_mask=True时分为两个mask
    }
    
    return {k: v.to(device) for k, v in inputs.items()}
```

---

## 模型前向传播

### 4.1 模型架构

**DrivePiZero** 包含以下组件：

```
DrivePiZero
├── Vision Encoder (SigLIP)
│   └── 处理图像: pixel_values -> vision embeddings
├── Text Encoder (Gemma Tokenizer + Embeddings)
│   └── 处理文本: input_ids -> text embeddings
├── Vision-Text Fusion
│   └── 融合视觉和文本特征 -> vlm_embeds
├── Proprio Encoder
│   └── 编码状态: proprios [B, cond_steps, state_dim] -> proprio_embeds
├── Action Encoder
│   └── 编码动作: actions [B, horizon_steps, action_dim] -> action_embeds
├── Joint Model (Mixture of Experts)
│   └── 融合所有模态: vlm + proprio + action -> joint_embeds
├── Action Decoder
│   └── 解码动作: action_embeds -> predicted_velocity [B, horizon_steps, action_dim]
└── Time Embedding
    └── Flow Matching时间条件嵌入
```

### 4.2 前向传播流程 (`forward`)

**文件**: `src/model/DrivePi0/drivepi0.py::DrivePiZero.forward()`

```python
def forward(self, input_ids, pixel_values, causal_mask, 
            vlm_position_ids, proprio_position_ids, action_position_ids,
            proprios, actions, t):
    """
    训练时的Flow Matching损失计算
    """
    
    # 1. Flow Matching: 生成带噪声的动作
    x0 = torch.randn_like(actions)  # 随机噪声 [B, horizon_steps, action_dim]
    x1 = actions                      # 真实动作 [B, horizon_steps, action_dim]
    psi_t = self.psi_t(x0, x1, t)    # 插值: (1-t)*x0 + t*x1 [B, horizon_steps, action_dim]
    
    # 2. 视觉-语言编码
    inputs_embeds = self._forward_siglip_and_text_embedding(
        input_ids, pixel_values
    )  # [B, seq_len, embed_dim]
    
    # 3. 状态编码
    proprio_embeds = self.proprio_encoder(proprios)  
    # [B, cond_steps, state_dim] -> [B, cond_steps, embed_dim]
    
    # 4. 时间条件嵌入
    time_cond = self.time_embedding(t)  # [B] -> [B, embed_dim]
    
    # 5. 动作编码 (带时间条件)
    if self.action_expert_adaptive_mode:
        action_embeds = self.action_encoder(psi_t)  # 自适应模式
    else:
        action_embeds = self.action_encoder(psi_t, time_cond)  # 时间条件模式
    
    # 6. Joint Model: 融合所有模态
    action_embeds = self.joint_model(
        attention_mask=causal_mask,
        position_ids_all={
            "vlm": vlm_position_ids,
            "proprio": proprio_position_ids,
            "action": action_position_ids,
        },
        embeds_all={
            "vlm": inputs_embeds,      # 视觉-语言特征
            "proprio": proprio_embeds, # 状态特征
            "action": action_embeds,    # 动作特征
        },
        time_cond=time_cond,
        kv_caches={},  # 训练时不使用缓存
    )["action"]  # 只返回动作部分的嵌入
    
    # 7. 动作解码: 预测速度场
    v_psi = self.action_decoder(action_embeds)  
    # [B, horizon_steps, embed_dim] -> [B, horizon_steps, action_dim]
    
    # 8. 计算目标速度场
    d_psi = x1 - (1 - self.flow_sig_min) * x0
    
    # 9. Flow Matching损失: MSE
    loss = torch.mean((v_psi - d_psi) ** 2)
    return loss
```

### 4.3 Flow Matching原理

Flow Matching是一种生成模型训练方法：

- **目标**: 学习从噪声分布到数据分布的连续流
- **时间步 t**: 在 [0, 1] 之间采样
- **插值**: `psi_t = (1-t)*x0 + t*x1`，其中 x0 是噪声，x1 是真实数据
- **速度场**: 模型预测速度场 `v_psi`，应该匹配真实速度场 `d_psi`
- **损失**: `L = ||v_psi - d_psi||²`

---

## 损失计算与优化

### 5.1 梯度累积

```python
# 训练循环中
for batch in train_dataloader:
    inputs = preprocess_batch(batch, ...)
    
    # 非最后一步: 不同步梯度
    if (cnt_batch + 1) % grad_accumulation_steps != 0:
        with model.no_sync():  # DDP时不同步
            loss = model(**inputs)
            (loss / grad_accumulation_steps).backward()
    
    # 最后一步: 同步梯度并更新
    else:
        loss = model(**inputs)
        (loss / grad_accumulation_steps).backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
        
        # 优化器更新
        action_optimizer.step()
        action_lr_scheduler.step()
        if train_vlm:
            vlm_optimizer.step()
            vlm_lr_scheduler.step()
        
        # 清零梯度
        optimizer.zero_grad()
```

### 5.2 优化器配置

- **优化器**: `bitsandbytes.AdamW8bit` (8-bit优化器，节省显存)
- **学习率调度器**: `CosineAnnealingWarmupRestarts`
- **梯度裁剪**: `max_grad_norm=1.0`
- **混合精度**: `torch.autocast` + `bfloat16`

### 5.3 双优化器策略

- **Action Optimizer**: 优化轨迹预测相关参数 (~0.315B参数)
- **VLM Optimizer**: 优化视觉-语言模型参数 (~2.291B参数，如果train_vlm=True)

---

## 训练循环

### 6.1 主训练循环

**文件**: `src/agent/drivepi0/train.py::DrivePiZeroTrainAgent.run()`

```python
def run(self):
    cnt_batch = 0
    cnt_update = 0
    loss_train_deque = deque(maxlen=grad_accumulation_steps)
    
    while True:
        for batch in train_dataloader:
            # 1. 预处理批次
            inputs = preprocess_batch(batch, split_mask=False, sample_fm_time=True)
            
            # 2. 前向传播和反向传播 (带梯度累积)
            if (cnt_batch + 1) % grad_accumulation_steps != 0:
                with model.no_sync():
                    with torch.autocast(...):
                        loss = model(**inputs)
                    (loss / grad_accumulation_steps).backward()
            else:
                with torch.autocast(...):
                    loss = model(**inputs)
                (loss / grad_accumulation_steps).backward()
                
                # 3. 优化器更新
                clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                cnt_update += 1
                
                # 4. 保存模型
                if cnt_update % save_model_freq == 0:
                    save_training(cnt_update, cnt_batch)
            
            # 5. 记录损失
            loss_train_deque.append(loss.item())
            
            # 6. 验证 (可选)
            if cnt_batch % eval_freq == 0:
                run_evaluation()
            
            # 7. 日志记录
            if cnt_batch % log_freq == 0:
                log_metrics(...)
                swanlab.log(metrics, step=cnt_batch)
            
            cnt_batch += 1
            
            # 8. 检查是否完成
            if cnt_update >= n_updates:
                return
```

### 6.2 验证流程

```python
def run_evaluation():
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = preprocess_batch(batch, split_mask=True, sample_fm_time=False)
            gt_trajectory = inputs.pop("actions")
            
            # 推理模式: 使用infer_action
            preds = model.infer_action(**inputs)
            
            # 计算准确率 (在不同阈值下)
            eval_accuracy += get_action_accuracy(gt_trajectory, preds, thresholds)
            eval_l1_loss += F.l1_loss(preds, gt_trajectory)
    
    model.train()
```

### 6.3 关键配置参数

**批次配置**:
- `global_batch_size`: 1024 (全局批次大小)
- `per_device_batch_size`: 16 (每设备批次大小)
- `grad_accumulation_steps`: `global_batch_size // (per_device_batch_size * world_size)`

**训练配置**:
- `n_updates`: 总更新步数 = `n_train // global_batch_size * n_epochs`
- `log_freq`: 16 (每16个batch记录一次)
- `eval_freq`: 2000 (每2000个batch验证一次)
- `save_model_freq`: 每个epoch保存一次

**数据配置**:
- `cond_steps`: 5 (历史状态时间步)
- `horizon_steps`: 10 (未来轨迹时间步)
- `action_dim`: 2 (动作维度: x, y)
- `state_dim`: 10 (状态维度)

---

## 总结

### 数据流

```
PKL文件 
  → TorchDataset.__getitem__() 
    → 归一化 (Normalize)
      → DataLoader 批次化
        → preprocess_batch()
          → VLAProcessor (文本+图像)
            → 模型输入字典
              → DrivePiZero.forward()
                → Flow Matching损失
                  → 反向传播
                    → 优化器更新
```

### 关键文件

- **入口**: `script/run.py`
- **训练逻辑**: `src/agent/drivepi0/train.py`
- **数据集**: `src/data/dataset.py`
- **归一化**: `src/data/utils/normalization.py`
- **模型**: `src/model/DrivePi0/drivepi0.py`
- **处理**: `src/model/DrivePi0/processing.py`
- **配置**: `config/train/DrivePi0/base_debug.yaml`

