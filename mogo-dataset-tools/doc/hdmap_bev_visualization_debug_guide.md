# `hdmap_bev` 可视化调试说明

本文档说明如何用：

- `/pcpt/pcpt/data/VLA/MOGO/export_hdmap_training_images.py`

对生成后的 `hdmap_bev` 做两种增强可视化：

1. `debug overlay`
2. `world-stabilized`

依赖的 sample 字段由：

- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py`

写入：

- `sample_data["ego_utm_position"]`
- `sample_data["ego_yaw"]`

## 1. 背景

默认 `hdmap_bev` 是以“当前 sample 的 ego 局部坐标系”为参考裁切并编码的。

这意味着：

- 每一帧地图都跟随当前 ego pose 重新裁切
- 连续播放时，地图相对 ego 看起来会持续变化
- 这不一定是坐标系错位，也可能只是正常的 ego-centric 重裁切结果

为了区分“正常重裁切”与“坐标/朝向实现错误”，新增了两种调试模式。

## 2. Debug Overlay

命令示例：

```bash
python3 /pcpt/pcpt/data/VLA/MOGO/export_hdmap_training_images.py \
  --input-root /path/to/output_sequence_dir \
  --overwrite \
  --draw-debug-overlay \
  --include-future-trajectory
```

输出目录默认会变成：

```text
hdmap_bev_img_debug
```

该模式会额外画出：

- 10m 参考网格
- 局部坐标轴
- 当前输出坐标系下的 pose marker
- 调试文字面板

当前图像中的方向约定应按实际渲染解释为：

- 向右是 `+x_local`
- 向上是 `+y_local`
- `+y_local` 同时也是当前 ego forward 方向

因此如果历史轨迹主要落在 ego 后方，通常会看到：

- `his_trajectory` 更多分布在负 `y_local`
- `fur_trajectory` 更多分布在正 `y_local`

## 3. World-Stabilized

命令示例：

```bash
python3 /pcpt/pcpt/data/VLA/MOGO/export_hdmap_training_images.py \
  --input-root /path/to/output_sequence_dir \
  --overwrite \
  --draw-debug-overlay \
  --world-stabilized \
  --include-future-trajectory
```

输出目录默认会变成：

```text
hdmap_bev_img_world_stabilized_debug
```

该模式会：

1. 以同一 sequence 的第一帧 pose 作为参考系
2. 将后续每帧 `hdmap_bev` 从“当前 ego 局部系”重新投影到“首帧局部系”
3. 使地图尽量保持静止，让 ego 和轨迹在图中运动

debug 面板会额外显示：

- `ref_ts`
- `delta_local`
- `delta_yaw_deg`

其中：

- `delta_local`
  - 当前帧 ego 原点在参考首帧局部系中的位置
- `delta_yaw_deg`
  - 当前帧与参考首帧的 yaw 差

## 4. 如何解读结果

如果 `debug overlay` 下看到：

- 历史点主要在负 `y_local`
- 未来点主要在正 `y_local`
- 地图几何与轨迹方向一致

那通常说明：

- 地图局部坐标和轨迹局部坐标是对齐的
- “地图在动”更可能只是 ego-centric 裁切带来的视觉效果

如果 `world-stabilized` 下地图能基本固定，而 ego 位置随时间变化，则说明：

- 地图重投影方向大体正确
- 问题不在于简单的 yaw 符号反了或轴互换

## 5. 模式限制

`world-stabilized` 固定的是“首帧裁切范围”。

因此当车辆已经跑出首帧视野很远时，后续帧可能出现：

- 大面积黑边
- 当前 ego marker 落到画面外
- 地图主体只剩首帧附近仍被覆盖的区域

这不是重投影失败，而是固定视野窗口的直接结果。

例如如果某帧相对首帧已经前进约 `260m`，而原始 `hdmap_bev` 只覆盖约 `180m` 左右的纵向范围，那么该帧大部分内容超出参考窗口是预期现象。

## 6. 适用场景

建议这样使用：

1. 先跑 `--draw-debug-overlay`
2. 再跑 `--draw-debug-overlay --world-stabilized`
3. 先确认坐标轴和轨迹方向
4. 再判断地图变化是“正常 ego-centric 裁切”还是“实现错误”

## 7. 当前结论

在已验证样例中：

- `his_trajectory` / `fur_trajectory` 与地图局部坐标系是一致的
- 普通模式下“地图相对 ego 往前移动”的观感，主要来自每帧按当前 ego 重新裁切
- `world-stabilized` 可以作为排查工具，但不是训练数据默认导出模式
