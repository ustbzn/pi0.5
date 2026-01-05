# Drive-π0 完整模型架构说明

本文档详细说明 Drive-π0 模型的完整架构，包括数据输入、模型结构和输出。

---

## 一、数据输入

### 1.1 原始数据格式

从数据集中加载的数据包含以下字段：

```python
batch = {
    'state': torch.Tensor,          # [B, cond_steps, proprio_dim] - 状态序列（4步历史）
    'trajectory': torch.Tensor,     # [B, 1, horizon_steps, action_dim] - 未来轨迹（6步）
    'image_front': torch.Tensor,    # [B, H, W, C] - 当前前视图像
    'image_front_time': torch.Tensor, # [B, H, W, C] - 历史前视图像
    'language_instruction': List[str] # 语言指令（如 "predict trajectory"）
}
```

**关键维度说明**：
- `cond_steps = 4`：历史状态步数
- `horizon_steps = 6`：预测轨迹步数
- `action_dim = 2`：动作维度（EEF_POS）
- `proprio_dim = 10`：状态维度（POS_EULER）

### 1.2 数据预处理流程

#### 图像处理

```python
# 1. 图像重排: [B, H, W, C] -> [B, C, H, W]
images_front = einops.rearrange(images_front, "B H W C -> B C H W")
images_front = images_front.unsqueeze(1)  # [B, 1, C, H, W]

# 2. 拼接多张图像
images = torch.cat((images_front, images_front_history), dim=1)  # [B, 2, C, H, W]

# 3. 通过 VLAProcessor 处理
model_inputs = processor(text=texts, images=images)
# 输出: input_ids, attention_mask, pixel_values
```

#### VLAProcessor 处理步骤

1. **图像归一化**：`[0, 255] -> [0, 1]`，然后标准化
2. **文本 tokenization**：添加图像 token 占位符 `<image>` × 512
3. **生成输入**：`input_ids` 和 `attention_mask`

**代码位置**：`src/model/DrivePi0/processing.py:96-137`

---

## 二、模型结构

### 2.1 整体架构组件

Drive-π0 包含以下主要组件：

```
DrivePiZero
├── Vision Tower (SigLIP)          # 视觉编码器
├── Multi-modal Projector           # 视觉-文本投影层
├── Text Embedding                  # 文本嵌入层
├── Proprio Encoder                 # 状态编码器
├── Action Encoder                  # 动作编码器（支持时间条件）
├── Time Embedding                  # 时间步嵌入（Sinusoidal）
├── Joint Model                     # 联合模型（Mixture of Experts）
│   ├── VLM Mixture (Gemma)         # 视觉-语言混合专家
│   ├── Proprio Mixture             # 状态混合专家
│   └── Action Mixture              # 动作混合专家
└── Action Decoder                  # 动作解码器
```

### 2.2 各组件详细说明

#### 1. Vision Tower (SigLIP)

- **位置**：`src/model/paligemma/siglip.py`
- **配置**：27 层，hidden_size=1152
- **功能**：将图像 `[B, 2, 3, 224, 224]` 编码为图像特征
- **输出**：`[B, num_patches, 1152]`（每张图 256 patches，共 512）

#### 2. Multi-modal Projector

- **功能**：将视觉特征投影到 VLM 的隐藏空间
- **输入**：`[B, 512, 1152]`
- **输出**：`[B, 512, 2048]`（VLM hidden_size）

#### 3. Text Embedding

- **功能**：将 token IDs 转换为嵌入向量
- **输入**：`input_ids` `[B, max_seq_len]`
- **输出**：`[B, max_seq_len, 2048]`

#### 4. 图像-文本融合

**代码位置**：`src/model/DrivePi0/drivepi0.py:389-435`

```python
def _forward_siglip_and_text_embedding():
    # 1. 文本嵌入
    inputs_embeds = self.embed_tokens(input_ids)
    
    # 2. 图像特征提取和投影
    for image in pixel_values:
        image_features = vision_tower(image)
        image_tokens = multi_modal_projector(image_features)
    
    # 3. 特征缩放
    scaled_image_features = image_tokens / sqrt(hidden_size)
    
    # 4. 融合：将图像 token 插入到文本序列中
    final_embedding[image_mask] = scaled_image_features
    final_embedding[text_mask] = inputs_embeds[text_mask]
```

#### 5. Proprio Encoder

- **结构**：`nn.Linear(proprio_dim, proprio_hidden_size)`
- **输入**：`[B, cond_steps, 10]`
- **输出**：`[B, cond_steps, 1024]`

#### 6. Action Encoder

**代码位置**：`src/model/DrivePi0/modules.py:27-55`

- **结构**：3 层 MLP，支持时间条件
- **输入**：`[B, horizon_steps, action_dim]` + 可选时间嵌入
- **输出**：`[B, horizon_steps, 1024]`
- **模式**：
  - `time_cond=True`：时间嵌入与动作拼接
  - `time_cond=False`：仅动作编码（用于 adaLN 模式）

#### 7. Time Embedding

**代码位置**：`src/model/DrivePi0/modules.py:9-24`

- **结构**：Sinusoidal 位置编码
- **输入**：时间步 `t` `[B]`
- **输出**：`[B, time_hidden_size]` 或 `[B, action_hidden_size]`

#### 8. Joint Model

**代码位置**：`src/model/DrivePi0/joint_model.py:307-381`

核心：**Mixture of Experts** 架构，包含三个混合专家：

**a) VLM Mixture (Gemma)**
- 18 层 Transformer Decoder
- hidden_size=2048
- 处理图像-文本融合特征

**b) Proprio Mixture**
- 18 层 Transformer Decoder（与 Action 共享权重）
- hidden_size=1024
- 处理状态特征

**c) Action Mixture**
- 18 层 Transformer Decoder（与 Proprio 共享权重）
- hidden_size=1024
- 处理动作特征

### 2.3 Block Attention 机制

**代码位置**：`src/model/DrivePi0/drivepi0.py:284-337`

```
注意力模式（块状因果掩码）:
                img/text  proprio  action
img/text          ✓         ✗        ✗
proprio           ✓         ✓        ✗
action            ✓         ✓        ✓
```

**说明**：
- Image/Text tokens 只能看到自己
- Proprio tokens 可以看到 Image/Text 和自己
- Action tokens 可以看到 Image/Text、Proprio 和自己

### 2.4 联合注意力计算流程

**代码位置**：`src/model/DrivePi0/joint_model.py:130-304`

在 `forward_mixture_attn()` 函数中：

1. **计算各 mixture 的 Query**
2. **从 KV cache 加载非活跃 mixture 的 K/V**（推理时）
3. **计算活跃 mixture 的新 K/V**
4. **拼接所有 mixture 的 Q、K、V**
5. **计算注意力**：`attn_weights = Q @ K^T / sqrt(head_dim)`
6. **应用 softmax 和 mask**
7. **计算输出**：`attn_output = attn_weights @ V`
8. **分割回各 mixture**

---

## 三、数据流和计算流程

### 3.1 训练阶段（Flow Matching）

**代码位置**：`src/model/DrivePi0/drivepi0.py:628-682`

```python
# 1. 输入准备
x0 = torch.randn_like(actions)  # 噪声
x1 = actions                     # 真实动作
t = sample_fm_time(bsz)         # 采样时间步

# 2. Flow 插值
psi_t = (1 - (1 - sig_min) * t) * x0 + t * x1

# 3. 编码
inputs_embeds = _forward_siglip_and_text_embedding(input_ids, pixel_values)
proprio_embeds = proprio_encoder(proprios)
time_cond = time_embedding(t)
action_embeds = action_encoder(psi_t, time_cond)

# 4. 通过联合模型
action_embeds = joint_model(
    embeds_all={
        "vlm": inputs_embeds,
        "proprio": proprio_embeds,
        "action": action_embeds
    },
    kv_caches={}  # 训练时不使用 cache
)["action"]

# 5. 解码
v_psi = action_decoder(action_embeds)

# 6. 计算损失
d_psi = x1 - (1 - sig_min) * x0
loss = mean((v_psi - d_psi)^2)
```

### 3.2 推理阶段（ODE 积分）

**代码位置**：`src/model/DrivePi0/drivepi0.py:437-511`

```python
# 1. 初始化
action = torch.randn([B, horizon_steps, action_dim])  # 从噪声开始
t = 0

# 2. 预计算 VLM 和 Proprio 的 KV cache
kv_caches = joint_model.build_mixture_caches()
_, kv_caches = joint_model(
    embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
    kv_caches=kv_caches,
    return_caches=True
)

# 3. ODE 积分循环（Euler 方法）
for step in range(num_inference_steps):
    # 编码当前动作和时间
    time_cond = time_embedding(t)
    action_embeds = action_encoder(action, time_cond)
    
    # 通过联合模型（复用 VLM 和 Proprio 的 KV cache）
    action_embeds = joint_model(
        embeds_all={"action": action_embeds},
        kv_caches=kv_caches,
        cache_mode="append_non_active"  # 使用非活跃 mixture 的 cache
    )["action"]
    
    # 解码速度
    action_vel = action_decoder(action_embeds)
    
    # 更新动作
    action += delta_t * action_vel
    t += delta_t

# 4. 裁剪最终输出
action = clamp(action, -clip_value, clip_value)
```

### 3.3 Flow Matching 时间步采样

**代码位置**：`src/agent/drivepi0/train.py:267-275`

支持两种采样方式：

1. **Uniform 采样**：`t ~ Uniform(0, 1)`
2. **Beta 采样**：`z ~ Beta(alpha, beta)`, `t = t_max * (1 - z)`

---

## 四、输出

### 4.1 训练输出

- **损失值**：`loss = mean((v_psi - d_psi)^2)`，标量

### 4.2 推理输出

- **动作序列**：`[B, horizon_steps, action_dim]`
  - `B`：batch size
  - `horizon_steps=6`：预测步数
  - `action_dim=2`：动作维度（如 EEF_POS）

---

## 五、关键设计特点

### 5.1 Mixture of Experts

- **VLM、Proprio、Action 三个专家**
- **Proprio 和 Action 共享权重**（`tie_action_proprio_weights`）
- **独立的隐藏空间和注意力计算**

### 5.2 Block Attention

- **块状因果掩码**，支持跨模态注意力
- **高效的注意力计算**，避免不必要的计算

### 5.3 KV Cache 优化

- **推理时缓存 VLM 和 Proprio 的 KV**
- **加速 ODE 积分过程**，避免重复计算

### 5.4 Flow Matching

- **使用条件流匹配训练动作预测**
- **支持不同的时间步采样策略**

### 5.5 时间条件

- **支持时间嵌入**，适配 flow matching 的 ODE 过程
- **可选的 adaLN 模式**，使用自适应层归一化

---

## 六、模型参数规模

### 6.1 参数统计

- **VLM 部分（Gemma）**：~2.3B（可 LoRA 微调）
- **Action Expert**：~0.315B（action encoder + decoder + joint model action mixture）
- **总参数量**：约 2.6B

### 6.2 可训练参数

- **如果 `train_vlm=True`**：
  - VLM 部分可训练（可能使用 LoRA）
  - Action Expert 可训练
- **如果 `train_vlm=False`**：
  - VLM 部分冻结
  - 仅 Action Expert 可训练

---

## 七、关键代码文件

### 7.1 模型定义

- `src/model/DrivePi0/drivepi0.py`：主模型类
- `src/model/DrivePi0/joint_model.py`：联合模型
- `src/model/DrivePi0/mixture.py`：混合专家定义
- `src/model/DrivePi0/modules.py`：编码器模块

### 7.2 数据处理

- `src/model/DrivePi0/processing.py`：图像-文本处理
- `src/data/dataset.py`：数据集加载
- `src/agent/dataset.py`：训练数据集包装

### 7.3 训练和推理

- `src/agent/drivepi0/train.py`：训练代码
- `src/model/kv_cache.py`：KV cache 实现

---

## 八、配置参数

### 8.1 关键配置

```yaml
cond_steps: 4              # 历史状态步数
horizon_steps: 6           # 预测轨迹步数
action_dim: 2              # 动作维度
proprio_dim: 10            # 状态维度
num_inference_steps: 10    # ODE 积分步数
flow_sampling: beta        # 时间步采样方式
flow_sig_min: 0.001        # Flow matching 参数
```

### 8.2 模型配置

```yaml
# VLM Mixture
vlm:
  hidden_size: 2048
  num_hidden_layers: 18

# Proprio/Action Mixture
proprio/action:
  hidden_size: 1024
  num_hidden_layers: 18
```

---

## 九、总结

Drive-π0 是一个基于 Mixture of Experts 架构的视觉-语言-动作模型，通过 Flow Matching 方法训练动作预测。模型的核心特点包括：

1. **多模态融合**：图像、文本、状态和动作的统一表示
2. **高效推理**：通过 KV cache 优化 ODE 积分过程
3. **灵活训练**：支持 VLM 微调和 Action Expert 训练
4. **可扩展性**：模块化设计，易于添加新的模态或专家

---

**文档生成时间**：2024年
**版本**：基于 DriveMoE 代码库

