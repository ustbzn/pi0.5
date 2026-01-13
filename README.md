<div align="center">   

# Drive-π0

</div>

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
> 注意：这里是python2的环境，其他均为python环境

```bash
cd /D2/NXT/vlm/miaobinqi/DriveMoE/mogo-dataset-tools
# - 替换指定的bag包路径
python2.7 parseRosBag-2O_py27.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
```

**转换为pkl文件[`python3`]：**
```bash
python json_to_pkl.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
```

> 如果要重新生成数据，需要删除之前文件：`find /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags -maxdepth 2 -type d -name "drivepi" -exec rm -rf {} +`

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
cd /D2/NXT/vlm/miaobinqi/DriveMoE
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