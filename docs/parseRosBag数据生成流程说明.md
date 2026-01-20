# parseRosBag-2O_py27.py 数据生成流程说明

## 1. 脚本概述

`parseRosBag-2O_py27.py` 是一个用于从 ROS bag 文件生成训练数据集的 Python 2.7 脚本。该脚本主要功能包括：

- 解析 ROS bag 文件中的定位数据、轨迹数据
- 加载已存在的相机图像文件
- 对定位数据进行 2Hz 采样（每 0.5 秒一帧）
- 生成 JSON 格式的训练样本数据
- 支持批量处理多个 bag 文件夹

## 2. 输入输出

### 2.1 输入要求

**目录结构：**
```
bags_root_dir/
├── bag_folder_1/
│   ├── *-rosmaster-*.bag      # Master bag 文件（必需）
│   ├── *-rosslave-*.bag       # Slave bag 文件（必需）
│   ├── front/                  # 前相机图像目录
│   │   ├── timestamp1.jpg
│   │   └── timestamp2.jpg
│   ├── back/                   # 后相机图像目录
│   ├── left/                   # 左相机图像目录
│   ├── right/                  # 右相机图像目录
│   ├── left_back/              # 左后相机图像目录
│   └── right_back/             # 右后相机图像目录
├── bag_folder_2/
└── ...
```

**ROS Topic 要求：**
- `/localization/global`: 定位数据（Localization 消息）
- `/planning/trajectory` 或包含 "trajectory" 的 topic: 轨迹数据（Trajectory 消息）

### 2.2 输出结果

**输出目录：**
```
bag_folder/
└── drivepi_json/               # 生成的 JSON 样本目录
    ├── scene_name_step1.json
    ├── scene_name_step2.json
    └── ...
```

**JSON 文件格式：**
每个 JSON 文件代表一个训练样本，包含图像路径、历史运动状态、导航目标点和未来轨迹等信息。

## 3. 数据生成流程

### 3.1 整体流程

脚本的处理流程按照以下顺序执行：

```
1. 初始化 BagDataHandler
   ↓
2. 加载已存在的相机图像 (load_existing_cameras)
   ↓
3. 按 front 相机对齐相机缓存 (align_camera_cache_by_front)
   ↓
4. 解析定位数据 (parse_localization)
   ↓
5. 对定位数据进行 2Hz 采样 (sample_localization_2hz)
   ↓
6. 解析轨迹数据 (parse_trajectory)
   ↓
7. 生成 JSON 样本 (generate_pkl_samples)
   ↓
8. 关闭 bag 文件
```

### 3.2 详细步骤说明

#### 步骤 1: 初始化 BagDataHandler

**功能：**
- 检查并查找 master/slave bag 文件
- 打开 bag 文件准备读取
- 创建输出目录 `drivepi_json/`

**关键代码：**
```python
def __init__(self, bag_dir, output_dir=None, work_dir=None):
    self.bag_dir = bag_dir
    self.output_dir = output_dir or bag_dir
    self.json_samples_dir = os.path.join(self.output_dir, "drivepi_json")
    self._check_and_find_bags()  # 查找 bag 文件
    self._open_bags()            # 打开 bag 文件
```

#### 步骤 2: 加载已存在的相机图像

**功能：**
- 从指定目录加载所有相机图像（front, back, left, right, left_back, right_back）
- 从文件名提取时间戳（文件名格式：`timestamp.jpg`）
- 按时间戳排序并缓存图像路径和时间戳

**关键代码：**
```python
def load_existing_cameras(self):
    camera_keys = ["back", "front", "left", "left_back", "right", "right_back"]
    for key in camera_keys:
        camera_dir = os.path.join(self.output_dir, key)
        # 扫描 jpg 文件，提取时间戳
        # 存储到 self.camera_cache[key] = {"times": [...], "paths": [...]}
```

**输出：**
- `self.camera_cache`: 字典，key 为相机名称，value 包含 `times` 和 `paths` 列表

#### 步骤 3: 对齐相机缓存（以 front 为基准）

**功能：**
- 以 front 相机的时间戳序列为基准
- 对其他相机进行时间对齐（最大时间差：0.05 秒）
- 生成按 front 时间戳分组的相机路径字典

**关键代码：**
```python
def align_camera_cache_by_front(self, max_delta=0.05):
    # 以 front 时间戳为基准，对齐其他相机
    # 返回 grouped = {front_ts: {cam_key: path}}
```

**输出：**
- `self.camera_group_by_front`: 字典，key 为 front 时间戳，value 为所有相机的路径字典

#### 步骤 4: 解析定位数据

**功能：**
- 从 master bag 中读取 `/localization/global` topic
- 解析 Localization 消息，提取位置、航向角等信息
- 存储时间戳和定位数据

**关键代码：**
```python
def parse_localization(self):
    for topic, msg, t in self.master_bag.read_messages(topics=["/localization/global"]):
        loc_data = Localization()
        loc_data.ParseFromString(msg.data)
        ts = calculate_header_timestamp(loc_data.header)
        self.valid_timestamps.append(ts)
        self.valid_locations.append(loc_data)
```

**输出：**
- `self.valid_timestamps`: 定位数据时间戳列表
- `self.valid_locations`: Localization 对象列表

#### 步骤 5: 2Hz 采样定位数据

**功能：**
- 对定位数据进行 2Hz 采样（每 0.5 秒采样一次）
- 找到最接近采样时间点的定位数据（最大时间差：0.25 秒）
- 生成均匀采样的定位数据序列

**关键代码：**
```python
def sample_localization_2hz(self, sample_rate_hz=2.0):
    sample_interval = 1.0 / sample_rate_hz  # 0.5秒
    current_time = start_time
    while current_time <= end_time:
        # 找到最接近的定位数据索引
        idx = bisect.bisect_left(self.valid_timestamps, current_time)
        # 选择最接近的索引
        # 添加到采样结果
        current_time += sample_interval
```

**输出：**
- `self.sampled_2hz_timestamps`: 2Hz 采样后的时间戳列表
- `self.sampled_2hz_locations`: 2Hz 采样后的定位数据列表
- `self.sampled_2hz_indices`: 原始索引列表

**为什么需要 2Hz 采样？**
- 原始定位数据可能频率不固定（通常更高）
- 2Hz 采样确保每 0.5 秒一个样本，用于后续生成训练样本

#### 步骤 6: 解析轨迹数据

**功能：**
- 从 master bag 中查找轨迹 topic（优先查找 `/planning/trajectory`）
- 解析 Trajectory 消息
- 存储时间戳和轨迹数据

**关键代码：**
```python
def parse_trajectory(self):
    # 查找轨迹 topic
    trajectory_topics = ["/planning/trajectory", "/trajectory", "/planning/trajectory_plan"]
    # 解析轨迹消息
    for topic, msg, t in self.master_bag.read_messages(topics=[found_topic]):
        traj_data = Trajectory()
        traj_data.ParseFromString(msg.data)
```

**输出：**
- `self.trajectory_timestamps`: 轨迹数据时间戳列表
- `self.trajectory_data`: Trajectory 对象列表

**注意：** 当前版本中轨迹数据已解析但未在样本生成中使用，可能用于未来功能。

#### 步骤 7: 生成 JSON 样本

**功能：**
- 基于 2Hz 采样后的定位数据生成训练样本
- 跳过第一帧，从第二帧开始生成
- 每个样本包含历史状态、当前图像、导航目标点和未来轨迹

**关键代码：**
```python
def generate_pkl_samples(self, sample_rate_hz=2.0):
    # 使用 2Hz 采样后的时间戳
    for idx, sampled_ts in enumerate(sampled_times):
        if idx == 0:
            continue  # 跳过第一帧
        # 生成样本数据
        sample_data = self._create_sample_data(sampled_ts, scene_id, sample_idx, ...)
        # 保存为 JSON 文件
        json_filename = "{}_step{}.json".format(scene_name, sample_idx + 1)
```

**输出：**
- 多个 JSON 文件，每个文件代表一个训练样本
- 文件名格式：`{scene_name}_step{number}.json`

## 4. 样本数据结构详解

每个 JSON 样本包含以下字段：

### 4.1 图像路径字段（bytes 字符串）

**当前帧图像：**
- `his_cam_front`: 前相机图像路径（bytes）
- `his_cam_back`: 后相机图像路径（bytes）
- `his_cam_left`: 左相机图像路径（bytes）
- `his_cam_right`: 右相机图像路径（bytes）
- `his_cam_left_back`: 左后相机图像路径（bytes）
- `his_cam_right_back`: 右后相机图像路径（bytes）

**前一帧图像（0.5 秒前）：**
- `his_cam_front_time`: 前相机前一帧图像路径（bytes）
- `his_cam_back_time`: 后相机前一帧图像路径（bytes）
- `his_cam_left_time`: 左相机前一帧图像路径（bytes）
- `his_cam_right_time`: 右相机前一帧图像路径（bytes）
- `his_cam_left_back_time`: 左后相机前一帧图像路径（bytes）
- `his_cam_right_back_time`: 右后相机前一帧图像路径（bytes）

### 4.2 历史运动状态（5 帧历史数据）

**his_speed** (ndarray, shape: (5,), dtype: float32)
- 历史 5 帧的速度大小（自车坐标系下）
- 单位：m/s
- 计算方式：通过位置差分计算速度，转换到自车坐标系后取模

**his_acceleration** (ndarray, shape: (5, 3), dtype: float32)
- 历史 5 帧的加速度（自车坐标系，x/y/z 三个分量）
- 单位：m/s²
- 计算方式：通过速度差分计算加速度，转换到自车坐标系
- **注意：** 保留了 x、y、z 三个分量

**his_angular_velocity** (ndarray, shape: (5, 3), dtype: float32)
- 历史 5 帧的角速度（roll/pitch/yaw 三个分量）
- 单位：rad/s
- 计算方式：通过航向角差分计算 yaw 角速度，roll 和 pitch 为 0

**his_theta** (ndarray, shape: (5,), dtype: float32)
- 历史 5 帧相对于当前帧的航向角变化
- 单位：rad
- 当前帧（最后一帧）为 0.0

### 4.3 导航目标点

**x_command_far** (ndarray, shape: (5,), dtype: float32)
- 3 秒后的导航目标点在自车坐标系下的 y 坐标（注意：交换了 x 和 y）
- 单位：m
- 重复 5 次（对应 5 帧历史）

**y_command_far** (ndarray, shape: (5,), dtype: float32)
- 3 秒后的导航目标点在自车坐标系下的 x 坐标（注意：交换了 x 和 y）
- 单位：m
- 重复 5 次（对应 5 帧历史）

**计算方式：**
1. 基于 2Hz 采样数据，查找当前帧后 6 帧（3 秒后）的定位数据
2. 将导航点的 UTM 坐标转换到当前帧的自车坐标系
3. **注意：** 存储时交换了 x 和 y 坐标（与加速度和角速度的分量对应）

### 4.4 未来轨迹（7 帧：当前帧 + 未来 6 帧）

**fur_x** (ndarray, shape: (1, 7), dtype: float32)
- 未来 7 帧在自车坐标系下的 y 坐标（注意：交换了 x 和 y）
- 单位：m
- 当前帧（第 0 帧）为 0.0
- 后续 6 帧：未来 0.5 秒、1.0 秒、1.5 秒、2.0 秒、2.5 秒、3.0 秒

**fur_y** (ndarray, shape: (1, 7), dtype: float32)
- 未来 7 帧在自车坐标系下的 x 坐标（注意：交换了 x 和 y）
- 单位：m
- 当前帧（第 0 帧）为 0.0

**计算方式：**
1. 基于 2Hz 采样数据，查找未来各帧的定位数据
2. 将未来帧的 UTM 坐标转换到当前帧的自车坐标系
3. **注意：** 存储时交换了 x 和 y 坐标

### 4.5 状态字段

**status** (string)
- 样本状态标识
- 格式：`"scene-{scene_id}_step{step_number}"`
- 示例：`"scene-0001_step1"`

## 5. 坐标系说明

### 5.1 全局坐标系（UTM）
- **X 轴：** 东向（East）
- **Y 轴：** 北向（North）
- **Z 轴：** 垂直向上

### 5.2 自车坐标系
- **X 轴：** 车头前方
- **Y 轴：** 车身左侧
- **Z 轴：** 垂直向上

### 5.3 坐标转换

**从全局坐标系到自车坐标系：**
1. 平移：以当前车辆位置为原点
2. 旋转：使用旋转矩阵（由航向角 yaw 构建）

**旋转矩阵构建：**
```python
def build_rotation_matrix_from_yaw(yaw):
    # yaw: 航向角（从北向逆时针旋转的角度）
    R = np.array([
        [sin_yaw,  cos_yaw, 0.0],  # X轴：前进方向
        [-cos_yaw, sin_yaw, 0.0],  # Y轴：左侧方向
        [0.0,      0.0,     1.0]   # Z轴：垂直向上
    ])
    return R
```

### 5.4 坐标交换说明

**重要：** 在存储导航目标点和未来轨迹时，代码交换了 x 和 y 坐标：
- 存储的 `x_command_far` / `fur_x` 实际上是自车坐标系下的 **y 坐标**
- 存储的 `y_command_far` / `fur_y` 实际上是自车坐标系下的 **x 坐标**

**原因：** 与加速度和角速度的分量对应（可能是历史代码的约定）。

## 6. 使用方法

### 6.1 命令行用法

```bash
python2.7 parseRosBag-2O_py27.py <bags_root_dir>
```

**参数说明：**
- `bags_root_dir`: 包含多个 bag 文件夹的根目录（必需）

**示例：**
```bash
python2.7 /D2/NXT/vlm/miaobinqi/DriveMoE/mogo-dataset-tools/parseRosBag-2O_py27.py \
    /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
```

### 6.2 前置条件

1. **Python 2.7 环境**
2. **依赖库：**
   - `rosbag` (ROS Python API)
   - `numpy`
   - `pickle`
   - `json`
   - `protobuf` (用于解析 proto 消息)
3. **图像文件已提取：** 相机图像应已从 bag 文件中提取并保存在对应目录中
4. **Proto 文件：** 需要能够导入以下 proto 消息：
   - `proto.common.localization_pb2.Localization`
   - `proto.common.trajectory_pb2.Trajectory`
   - `proto.common.sensor_meta_pb2.RegisteredSensor`

### 6.3 处理流程

1. **脚本自动遍历** `bags_root_dir` 下的所有子目录
2. **检查每个目录** 是否包含 master 和 slave bag 文件
3. **依次处理** 每个有效的 bag 文件夹
4. **生成 JSON 样本** 到 `{bag_folder}/drivepi_json/` 目录

## 7. 关键算法说明

### 7.1 2Hz 采样算法

**目的：** 将不规则的定位数据采样为均匀的 2Hz 序列

**算法：**
1. 计算采样间隔：`interval = 1.0 / 2.0 = 0.5 秒`
2. 从第一个定位数据时间戳开始，每隔 0.5 秒采样一次
3. 对于每个采样时间点，使用二分查找找到最接近的定位数据
4. 如果时间差 ≤ 0.25 秒，则使用该数据；否则跳过

### 7.2 时间对齐算法

**目的：** 将多个相机的图像按时间对齐

**算法（以 front 相机为基准）：**
1. 遍历 front 相机的所有时间戳
2. 对于每个 front 时间戳，在其他相机中使用二分查找找到最接近的时间戳
3. 如果时间差 ≤ max_delta（默认 0.05 秒），则对齐成功
4. 生成对齐后的相机路径字典

### 7.3 历史特征计算

**历史窗口：** 5 帧（2Hz 采样，每帧间隔 0.5 秒，共 2.5 秒）

**速度计算：**
1. 通过位置差分计算全局坐标系下的速度：`vel_global = (pos_next - pos) / dt`
2. 转换到自车坐标系：`vel_ego = R_T * vel_global`
3. 计算速度大小：`speed = ||vel_ego[:2]||`

**加速度计算：**
1. 通过速度差分计算全局坐标系下的加速度：`acc_global = (vel_next - vel) / dt`
2. 转换到自车坐标系：`acc_ego = R_T * acc_global`
3. 保留 x、y、z 三个分量

**角速度计算：**
1. 计算航向角差分：`dyaw = normalize_angle(yaw_next - yaw)`
2. 计算 yaw 角速度：`angular_velocity_z = dyaw / dt`
3. roll 和 pitch 角速度为 0

### 7.4 未来轨迹计算

**未来窗口：** 7 帧（当前帧 + 未来 6 帧，共 3 秒）

**算法：**
1. 当前帧（i=0）：自车坐标系下为 (0, 0)
2. 未来帧（i=1..6）：目标时间 = 当前时间 + i * 0.5 秒
3. 查找最接近目标时间的定位数据（使用 2Hz 采样数据）
4. 将未来帧的 UTM 坐标转换到当前帧的自车坐标系
5. **交换 x 和 y 坐标**后存储

## 8. 注意事项

### 8.1 数据格式

- 图像路径存储为 **bytes 字符串**（UTF-8 编码）
- 数值数据使用 **numpy.ndarray**，序列化为 JSON 时需要特殊处理（脚本已实现）
- 坐标 **交换了 x 和 y**（与加速度/角速度分量对应）

### 8.2 时间同步

- 定位数据通过 2Hz 采样保证时间同步
- 相机图像按 front 相机时间戳对齐（最大时间差 0.05 秒）
- 如果某个相机找不到对齐的图像，对应字段为空字符串

### 8.3 边界情况处理

- **找不到定位数据：** 跳过该样本（返回 None）
- **找不到图像：** 图像路径字段为空字符串
- **找不到未来帧：** 使用最后一个成功找到的未来帧坐标填充
- **找不到导航点：** 导航目标点坐标为 (0, 0)

### 8.4 性能考虑

- 脚本使用二分查找提高查找效率
- 图像加载和定位数据解析在内存中缓存
- 生成的 JSON 文件按样本索引命名，便于后续处理

## 9. 输出示例

**生成的 JSON 文件结构示例：**

```json
{
  "his_cam_front": {
    "__bytes__": true,
    "data": "L2RhdGEvaW1hZ2VzL2Zyb250LzE2OTk5OTk5OTkuNTAwMDAwLmpwZw=="
  },
  "his_cam_front_time": {
    "__bytes__": true,
    "data": "L2RhdGEvaW1hZ2VzL2Zyb250LzE2OTk5OTk5OTkuMDAwMDAwLmpwZw=="
  },
  "his_speed": {
    "__numpy_array__": true,
    "data": [12.5, 12.8, 13.0, 12.9, 12.7],
    "dtype": "float32",
    "shape": [5]
  },
  "his_acceleration": {
    "__numpy_array__": true,
    "data": [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "dtype": "float32",
    "shape": [5, 3]
  },
  "fur_x": {
    "__numpy_array__": true,
    "data": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    "dtype": "float32",
    "shape": [1, 7]
  },
  "fur_y": {
    "__numpy_array__": true,
    "data": [[0.0, 6.25, 12.5, 18.75, 25.0, 31.25, 37.5]],
    "dtype": "float32",
    "shape": [1, 7]
  },
  "status": "scene-0001_step1"
}
```

## 10. 常见问题

### Q1: 为什么跳过第一帧？

**A:** 因为需要计算历史状态，第一帧没有足够的历史数据（需要 5 帧历史）。

### Q2: 为什么坐标交换了 x 和 y？

**A:** 这是代码中的约定，与加速度和角速度的分量对应。在使用这些数据时需要注意坐标交换。

### Q3: 2Hz 采样是否必须？

**A:** 是的，采样确保数据的时间间隔均匀，便于后续训练。原始定位数据频率可能不一致。

### Q4: 如何处理缺失的相机图像？

**A:** 如果某个相机找不到对齐的图像，对应的图像路径字段将为空字符串（空 bytes）。下游处理代码需要处理这种情况。

### Q5: 为什么使用 JSON 而不是 PKL？

**A:** 虽然函数名为 `generate_pkl_samples`，但实际输出是 JSON 格式。JSON 格式更便于跨语言使用和调试。

## 11. 后续处理建议

生成的 JSON 文件可以：

1. **转换为 PKL 格式：** 如果训练代码需要 PKL 格式，可以编写转换脚本
2. **数据统计：** 计算数据集的统计信息（均值、方差等），用于归一化
3. **数据验证：** 检查数据完整性和合理性
4. **数据增强：** 根据需要添加数据增强策略

---

**文档版本：** 1.0  
**最后更新：** 2025-01-XX  
**维护者：** DriveMoE Team

