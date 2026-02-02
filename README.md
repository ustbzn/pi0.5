# Driveπ0-MOGO

根据文件名后缀train和val分成两个文件夹:

* scene: train-850，val-150

* sample: total-(29069-20=29049), train-23930，val-5119

## 数据文件结构

初始数据目录：
```bash
20-bags/
├── 20251210141637/
│   ├── back/
│   ├── front/
│   ├── left/
│   ├── left_back/
│   ├── right/
│   ├── right_back/
│   ├── drivepi/
│   ├── drivepi_json/
│   ├── 20251210141637-1765347397140360221-rosmaster-JLBJA20073D.bag
│   ├── 20251210141637-1765347397140360221-rosslave-JLBJA20073D.bag
│   ├── 31765347397140360221.json
│   ├── calibrated_sensor.pb.txt
│   └── process.log
├── 20251210144033/
├── 20251210144451/
├── 20251210144851/
├── 20251210150954/
├── 20251210151440/
└── 20251210153138/
```
训练数据目录：
```bash
mogo-test/
├── dual/
├── inference/
├── train/
├── val/
├── video/
└── statistics.json    # 归一化数据
```
## 数据生成
文件名格式：`{scene_name}_step{number}.json`，**生成json文件[`python2.7`]**：
> 注意：这里是python2的环境，其他均为python3环境

```bash
cd DriveMoE/mogo-dataset-tools
# - 替换指定的bag包路径
python2.7 parseRosBag-2O_py27.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
```

**转换为pkl文件[`python3`]：**
```bash
# --skip-existing: 默认覆盖已存在的pkl文件
python json_to_pkl.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
```

**移动文件：**

```bash
# 移动文件用于手动划分数据集train/val，后续可自行更改
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir datasets/mogo-test/train \
    --exclude 20251210141637
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir datasets/mogo-test/val \
    --include 20251210141637
```
**生成归一化数据：**
```
cd DriveMoE
python src/data_processing/get_statistics.py \
    --data_path datasets/mogo-test/train \
    --output_path datasets/mogo-test/statistics.json
```
## 训练、推理、可视化
### 训练

```bash
sh script/training/train_drivepi0_mogo.sh
```
配置文件：`config/train/DrivePi0/base_mogo.yaml`

### 推理
```bash
CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_mogo_test.py --mode batch \
    --val_data_dir datasets/mogo-test/val \
    --config_path config/inference/DrivePi0/base_mogo_havetrain.yaml \
    --statistic_path datasets/mogo-test/statistics.json \
    --output_dir datasets/mogo-test/inference
```
### 可视化
```bash
# 可指定可视化单个scene
python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir datasets/mogo-test/inference \
    --pkl_root_dir datasets/mogo-test/val \
    --mode dual \
    --bev_xlim -1 15 \
    --bev_ylim -8 8 \
    --output_dir datasets/mogo-test/dual \
    --scene_names 20251210141637
# 可视化所有scene
python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir datasets/mogo-test/inference \
    --pkl_root_dir datasets/mogo-test/val \
    --mode dual \
    --bev_xlim -1 15 \
    --bev_ylim -8 8 \
    --output_dir datasets/mogo-test/dual
# 合并为视频
python src/utils/mogo_test/merge_images.py \
    --img-dir datasets/mogo-test/dual/20251210141637 \
    --out datasets/mogo-test/video/20251210141637.mp4 \
    --fps 4
```
> 当前可视化代码可能处理较慢，可使用 `src/utils/analyzing_traj.py`，含有多进程处理参数`--num_workers`.

# Driveπ0-nuScenes

## 数据生成
> `drivepi_nusc_converter_train.sh` 其中需要nuscene-all数据集的路径，默认的是本地绝对路径：`/D2/NXT/vlm/dataset/nuScenes/nuscene-all/`

```bash
# nuScenes数据集的生成
sh nuscene-dataset-tools/run_rest_parallel.sh
# 根据文件名后缀划分成train和val两个文件夹
python nuscene-dataset-tools/collect_split_json.py \
    --base-dir /D2/NXT/vlm/miaobinqi/datasets/nuscene-train-pi0-all-0109-1 \
    --out-dir  datasets/nuscene-0109
# json文件转换成pkl文件
python nuscene-dataset-tools/json_to_pkl_converter.py \
  /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-0109/train_json \
  --output datasets/nuscene-0109/train \
  --recursive
python nuscene-dataset-tools/json_to_pkl_converter.py \
  /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-0109/val_json \
  --output datasets/nuscene-0109/val \
  --recursive
```
## 训练、推理、可视化
**生成归一化数据：**

```bash
# 生成归一化数据, 更改输入输出路径
sh script/data_preprocessing/get_statistics.sh
```
### 训练
```bash
# 1. 更改配置文件 base_nuscene.yaml
# - run_name, 训练日志文件夹名
# - norm_mode, 归一化方法
# 2. 更改 train_drivepi0.sh 文件中的 SWANLAB_API_KEY
sh script/training/train_drivepi0.sh
```
配置文件：`config/train/DrivePi0/base_nuscene.yaml`

### 推理
```bash
# 推理得到轨迹数据，替换inference文件中的相关配置：
# - 训练后模型的pt文件路径(base_nuscene.yaml)
# - 归一化数据文件路径
# - 输出文件目录
CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_nuscene.py \
    --mode batch \
    --val_data_dir datasets/nuscene-0111/val \
    --config_path config/inference/DrivePi0/base_nuscene.yaml \
    --statistic_path datasets/nuscene-0111/statistics.json \
    --output_dir datasets/nuscene-0111/inference/1_percentile_epoch40
```

### 可视化
```bash
# 可视化
python src/utils/analyzing_traj.py \
	--trajectory_dir datasets/nuscene-0111/inference/1_percentile_epoch40 \
    --pkl_dir datasets/nuscene-0111/val \
    --mode dual \
    --bev_xlim -1 50 \
    --bev_ylim -20 20 \
    --output_dir datasets/nuscene-0111/visualizations/1_percentile_epoch40/dual \
    --num_workers 8
# 合并为视频
python src/utils/merge_images.py \
	--dual-dir datasets/nuscene-0111/visualizations/1_percentile_epoch40/dual \
	--out-dir datasets/nuscene-0111/visualizations/1_percentile_epoch40/video \
	--fps 5
```
