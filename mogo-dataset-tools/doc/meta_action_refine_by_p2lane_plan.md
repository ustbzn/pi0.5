# 基于 p2lane 的 meta_action 离线修正方案

## 目标

根据已有 `.pkl` 中的 `traj_p2lane_stable` 信息，以及同一目录下前后时间戳样本的时序上下文，离线修正
`/pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane` 下 `.pkl` 文件的 `meta_action` 字段。

修正只针对原始值为 `lane_keeping_and_centering` 的样本。其他 `meta_action` 保持原样。所有样本都新增
`raw_meta_action` 字段，用于保存修改前的原始值。

## 可用字段

`parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py` 生成的样本中，`traj_p2lane_stable` 来自当前样本附近的
2Hz 定位点，通常包含 11 个点：

- 4 帧历史
- 当前帧
- 6 帧未来

每个点格式为：

```python
(stable_lane_id, stable_s, stable_d, lane_relation)
```

字段含义：

- `stable_lane_id`: 稳定车道 ID
- `stable_s`: 沿车道方向 Frenet 坐标
- `stable_d`: 横向偏移，本文主要使用该值
- `lane_relation`: 相邻稳定车道关系，可能是 `keep`、`toleft`、`toright` 或空字符串

其中 `stable_d > 0` 按现有生成逻辑解释为偏左，`stable_d < 0` 解释为偏右。

## 总体策略

采用两阶段处理：

1. 单样本初判：只使用当前 `.pkl` 内的 `traj_p2lane_stable` 计算候选动作和置信度。
2. 同场景时序修正：按同一目录、时间戳或 step 顺序读取前后 `.pkl`，对候选动作做平滑、补洞和冲突消解。

这样既保留单样本可解释性，又避免因为单帧 p2lane 抖动导致标签频繁跳变。

## 导航保护规则

在实际改写 `meta_action` 之前，当前实现增加了一条基于 `navigation_command` 的保留条件。

如果 `.pkl` 中：

- `navigation_command` 是非空 list
- `navigation_command[0]` 是一个长度不少于 2 的子 list
- 该子 list 中不包含字符串 `'None'`
- 且 `navigation_command[0][1] < 15`

则该样本的 `meta_action` 不做修改，直接保留原 `.pkl` 中的结果。

这个规则的目的，是在距离当前导航指令目标较近时，避免仅凭 `traj_p2lane_stable` 的横向偏移去覆盖原始 `meta_action`。

实际数据里，`navigation_command` 常见结构类似：

```python
[
    ['straight', 47.3, 1, 2],
    ['None', nan, nan, nan],
]
```

命中保留条件时，脚本内部会记录：

```text
reason = preserve_by_navigation_command
```

并在 summary 中累计：

```text
preserved_by_navigation_command
```

## 单样本初判

### 有效点过滤

只使用满足以下条件的点：

- 点是 list 或 tuple
- 长度不少于 4
- `stable_d` 是有限数值，排除 `nan`、`inf`

如果有效点数少于 5，认为信息不足，保持原始 `meta_action`。

### 基础统计

对有效点计算：

```text
ds = 所有 stable_d
abs_ds = abs(stable_d)
relations = 所有 lane_relation
majority = len(ds) // 2 + 1
median_d = median(ds)
current_d = 当前帧 stable_d，优先取 11 点中的第 5 个点；缺失时使用 median_d
future_ds = 当前帧之后的 stable_d
```

同时计算趋势：

```text
delta_d = median(future_ds) - median(history_ds)
```

`delta_d` 用于判断是否正在持续向左或向右偏移。

### 标签集合

建议输出以下标签：

- `lane_keeping_and_centering`
- `in_lane_left_nudge`
- `in_lane_right_nudge`
- `out_of_lane_left`
- `out_of_lane_right`

### 判定优先级

优先级建议为：

```text
out_of_lane > lane_keeping_and_centering > in_lane_nudge > 保持原始值
```

原因是 `out_of_lane` 是更强语义，一旦满足明确越界证据，不应被历史帧中低偏移点稀释。

### out_of_lane 判定

触发条件：

```text
max(abs_ds) >= 0.9
且 lane_relation 中存在明确方向
且明确方向不左右混杂
```

方向规则：

- 明确方向集合为 `{"toleft"}`，输出 `out_of_lane_left`
- 明确方向集合为 `{"toright"}`，输出 `out_of_lane_right`
- 同时存在 `toleft` 和 `toright`，认为时序不稳定，暂不改

优化点：不要求所有点都是 `toleft` 或 `toright`。因为一个样本包含历史帧、当前帧和未来帧，真实变道或出车道过程中，前半段可能仍然是 `keep`。

### lane_keeping_and_centering 判定

触发条件：

```text
abs(d) < 0.3 的点数 >= majority
且 max(abs_ds) < 0.9
```

输出：

```text
lane_keeping_and_centering
```

### in_lane_nudge 判定

触发条件：

```text
abs(d) <= 0.9 的点数 >= majority
且 lane_relation 全部属于 {"keep", ""}
且不满足 lane_keeping_and_centering
```

方向规则：

- `median_d > 0.3`，输出 `in_lane_left_nudge`
- `median_d < -0.3`，输出 `in_lane_right_nudge`
- `abs(median_d) <= 0.3`，保持 `lane_keeping_and_centering`

增强规则：

如果 `median_d` 接近阈值，但 `current_d` 和未来多数点方向一致，也可以输出 nudge。例如：

```text
abs(median_d) <= 0.3
但 abs(current_d) > 0.3
且 future_ds 中同方向点数过半
```

这种情况说明偏移主要发生在当前帧之后，单纯用全窗口中位数可能被历史点稀释。

## 同目录时序修正

### 分组方式

按 `.pkl` 所在叶子目录分组。当前数据路径类似：

```text
/.../pkl_rout_action_scene_hdmap_lane/0316/2026-03-16-14-57-18/20260316150414/*.pkl
```

同一叶子目录通常对应同一段 scene。组内排序优先级：

1. 文件名中的 `stepN`
2. `sample_timestamp`
3. `relative_timestamp`

建议实际实现时同时校验 `scene_token`，如果同目录内出现多个 `scene_token`，再按 `scene_token` 二次分组。

### 时序窗口

每个样本除自身外，读取前后 `K` 个 `.pkl` 做上下文，建议：

```text
K = 2
```

2Hz 采样下，前后 2 个样本约等于前后 1 秒，总窗口约 2 秒，足够抑制噪声，又不会过度平滑短动作。

### 短脉冲过滤

如果某个非中心保持标签只持续 1 帧，并且前后都是 `lane_keeping_and_centering`，则回退为
`lane_keeping_and_centering`。

示例：

```text
keep, keep, in_lane_left_nudge, keep, keep
```

修正为：

```text
keep, keep, keep, keep, keep
```

理由：单帧 nudge 更可能是 p2lane 抖动。

### 短洞填补

如果同方向标签被 1 帧 `lane_keeping_and_centering` 打断，则填补中间帧。

示例：

```text
in_lane_left_nudge, lane_keeping_and_centering, in_lane_left_nudge
```

修正为：

```text
in_lane_left_nudge, in_lane_left_nudge, in_lane_left_nudge
```

`out_of_lane_left/right` 同理，但左右方向必须一致。

### nudge 到 out_of_lane 升级

如果连续多个同方向 `in_lane_*_nudge` 样本中，后续样本出现同方向 `out_of_lane_*`，则可以把靠近越界的前 1 到 2 帧升级或标记为高置信 nudge。

保守建议：不要直接把所有前序 nudge 升级为 out_of_lane，只在满足下面条件时升级：

```text
当前样本 max(abs_d) >= 0.8
后 1 到 2 帧存在同方向 out_of_lane
当前样本 direction 与后续 out_of_lane 一致
```

这样能更早识别出即将出车道的样本，但不会把正常车道内轻微偏移过度升级。

### out_of_lane 到 nudge 降级

如果某一帧被初判为 `out_of_lane_*`，但前后 2 帧都没有同方向 out_of_lane，且本帧只有单个点超过 0.9，
建议降级为同方向 `in_lane_*_nudge`。

理由：单个 future/history 点异常超过阈值，可能是地图匹配抖动。

### 左右冲突处理

如果一个时序窗口内出现左右交替，例如：

```text
in_lane_left_nudge, in_lane_right_nudge, in_lane_left_nudge
```

处理策略：

- 如果当前样本自身置信度低，回退为 `lane_keeping_and_centering`
- 如果当前样本 `abs(median_d)` 明显较大，比如 `> 0.5`，保留当前样本方向
- 不允许直接用窗口多数方向覆盖高置信单样本，避免真实 S 型轨迹被抹平

## 置信度设计

建议每个样本内部临时计算 `confidence`，不一定写回 pkl，但用于时序修正。

参考分数：

```text
valid_ratio = 有效点数 / 总点数
offset_strength = min(max(abs_ds) / 0.9, 1.0)
direction_consistency = 同方向点数 / 有效点数
relation_consistency = 明确 relation 是否单方向
```

建议：

- `lane_keeping_and_centering`: `abs(d) < 0.3` 占比越高，置信度越高
- `in_lane_nudge`: `0.3 < abs(d) <= 0.9` 占比越高，方向越一致，置信度越高
- `out_of_lane`: `max(abs_d)` 越高，且 relation 单方向越明确，置信度越高

低置信样本优先接受时序平滑，高置信样本尽量保留。

## 脚本接口建议

建议新增脚本：

```text
mogo-dataset-tools/update_meta_action_by_p2lane.py
```

参数：

```bash
python mogo-dataset-tools/update_meta_action_by_p2lane.py \
  --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
  --dry-run
```

正式写入：

```bash
python mogo-dataset-tools/update_meta_action_by_p2lane.py \
  --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
  --inplace \
  --backup
```

可选输出到新目录：

```bash
python mogo-dataset-tools/update_meta_action_by_p2lane.py \
  --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
  --output-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane_refined
```

建议参数：

- `--input-dir`: 输入 pkl 根目录
- `--output-dir`: 输出根目录，不指定时配合 `--inplace` 原地修改
- `--inplace`: 原地覆盖
- `--backup`: 原地覆盖前生成 `.bak`
- `--dry-run`: 只统计不写文件
- `--context-k`: 前后窗口大小，默认 2
- `--min-valid-points`: 最少有效 p2lane 点，默认 5
- `--center-threshold`: 中心保持阈值，默认 0.3
- `--out-threshold`: 出车道阈值，默认 0.9
- `--write-report`: 输出 CSV/JSON 统计报告

## 写回策略

每个样本写回：

```python
sample["raw_meta_action"] = old_meta_action
sample["meta_action"] = refined_meta_action
```

建议可选写入调试字段，默认不写，避免污染训练数据：

```python
sample["_meta_action_refine_debug"] = {
    "single_frame_action": "...",
    "final_action": "...",
    "confidence": 0.0,
    "reason": "...",
}
```

如果需要保留可审计性，更推荐把 debug 信息写到独立报告文件，而不是写入 pkl。

## 统计报告

脚本运行后输出汇总：

- 总 pkl 数
- 缺少 `traj_p2lane_stable` 数
- `traj_p2lane_stable` 有效点不足数
- 原始 `lane_keeping_and_centering` 数
- 因 `navigation_command` 规则而保留原值的样本数
- 保持 `lane_keeping_and_centering` 数
- 修正为 `in_lane_left_nudge` 数
- 修正为 `in_lane_right_nudge` 数
- 修正为 `out_of_lane_left` 数
- 修正为 `out_of_lane_right` 数
- 非 `lane_keeping_and_centering` 跳过数
- 时序平滑修改数
- 短脉冲过滤数
- 短洞填补数
- out_of_lane 降级数

建议额外输出 `changes.csv`：

```text
pkl_path,old_meta_action,new_meta_action,single_frame_action,confidence,reason,max_abs_d,median_d,relations
```

用于抽样复核。

## 推荐验证方式

### 抽样检查

每类新标签随机抽样 50 到 100 个样本，使用已有可视化工具查看：

```text
mogo-dataset-tools/visualize_pkl.py
```

重点检查：

- `in_lane_left_nudge` 是否确实偏左但未明显跨车道
- `in_lane_right_nudge` 是否确实偏右但未明显跨车道
- `out_of_lane_left/right` 是否存在明确跨出或切换车道趋势
- 是否误改了原本非 `lane_keeping_and_centering` 的样本

### 时序检查

对每个 scene 输出动作序列，检查是否存在高频跳变：

```text
keep -> left_nudge -> keep -> right_nudge -> keep
```

这类序列应明显减少。

### 阈值回归

先用 `--dry-run` 对比不同阈值：

```text
center-threshold: 0.25, 0.30, 0.35
out-threshold: 0.85, 0.90, 1.00
context-k: 1, 2, 3
```

优先选择“误报较少”的配置。标签修正用于训练数据时，保守比激进更安全。

## 风险与边界

- `stable_d` 的正负方向依赖当前 `lane_frenet.py` 的约定，如果后续 p2lane 生成逻辑变更，需要重新确认左右语义。
- 地图匹配失败或车道拓扑异常时，`traj_p2lane_stable` 可能包含空 lane 或 `nan`，必须过滤。
- `out_of_lane` 不应只依赖单个点越过 0.9，建议结合 relation 和时序上下文。
- 过强时序平滑可能抹掉真实短时避障动作，因此只过滤 1 帧短脉冲，不建议过滤 2 帧以上连续动作。

## 结论

相比只看单个 `.pkl` 的方案，加入同目录前后样本信息后，可以显著降低 p2lane 抖动导致的误标。推荐实现为：

1. 单样本基于 `stable_d` 和 `lane_relation` 产生候选动作。
2. 同 scene 内按 step 或 timestamp 排序，使用前后 2 帧做平滑。
3. 所有修改都保留 `raw_meta_action`，并输出独立报告供抽样复核。

默认策略应偏保守：证据不足时保持 `lane_keeping_and_centering`，只在偏移、方向和时序都相对一致时改为 nudge 或 out_of_lane。
