# `p2lane + stable Frenet` 使用说明

本文档面向实际跑数和验数，说明如何在：

- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py`

中启用并使用新增的：

- `traj_p2lane`
- `traj_p2lane_stable`

配套实现说明见：

- [p2lane_frenet_technical_note.md](/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/doc/p2lane_frenet_technical_note.md)

## 1. 功能概览

启用后，脚本会基于 `sampled_2hz_locations` 生成两条逐点车道匹配结果：

1. 原始 `traj_p2lane`
   - 每个点结构为 `(matched_lane_id, s, d)`
   - `matched_lane_id` 是地图原始 lane / lane_link ID
2. 稳定版 `traj_p2lane_stable`
   - 每个点结构为 `(stable_lane_id, s, d, lane_relation)`
   - `stable_lane_id` 会把一对一连续的 lane segment 合并成稳定 ID

符号约定：

- `d > 0` 表示点在参考中心线左侧
- `d < 0` 表示点在参考中心线右侧

`lane_relation` 对有效点只会取以下值：

- `keep`
- `toleft`
- `toright`

无效点或补齐点会写为：

```python
("", nan, nan, "")
```

## 2. 运行前准备

## 2.1 必需输入

运行脚本至少需要三类输入：

1. `rosbags-root`
2. `images-root`
3. `output-root`

如果需要生成 `hdmap_bev`、`traj_p2lane` 和 `traj_p2lane_stable`，还需要：

4. `hdmap-sql`

## 2.2 依赖

需要确保以下 Python 依赖已可用：

- `pyproj`
- `shapely`

还需要系统可加载 Spatialite 扩展。

如果系统不能自动加载，可以手动指定：

```bash
export SPATIALITE_EXTENSION=/path/to/mod_spatialite.so
```

## 3. 最常用运行命令

## 3.1 基本跑数

```bash
python3 /pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py \
  --rosbags-root /path/to/rosbags \
  --images-root /path/to/images \
  --output-root /path/to/output \
  --hdmap-sql /D2/NXT/aip/maps/beijing/4.1.3_20251222/map.sqlite \
  --hdmap-utm-zone 50
```

这会同时生成：

- 原有 pkl sample
- `hdmap_bev`
- `traj_p2lane`
- `traj_p2lane_stable`

## 3.2 只跑少量切片做 smoke test

```bash
python3 /pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py \
  --rosbags-root /path/to/rosbags \
  --images-root /path/to/images \
  --output-root /path/to/output_test \
  --hdmap-sql /D2/NXT/aip/maps/beijing/4.1.3_20251222/map.sqlite \
  --hdmap-utm-zone 50 \
  --limit 1
```

## 3.3 需要额外输出 JSON 调试文件

```bash
python3 /pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py \
  --rosbags-root /path/to/rosbags \
  --images-root /path/to/images \
  --output-root /path/to/output \
  --hdmap-sql /D2/NXT/aip/maps/beijing/4.1.3_20251222/map.sqlite \
  --write-json
```

## 4. 常用参数说明

- `--rosbags-root`
  - 原始 bag 根目录
- `--images-root`
  - 已拆图像根目录
- `--output-root`
  - 输出 pkl 根目录
- `--labels-root`
  - 标签根目录，可选
- `--write-json`
  - 是否输出 JSON 调试文件，可选
- `--limit`
  - 仅处理前 N 个切片，适合调试
- `--hdmap-sql`
  - 地图 sqlite 路径
- `--hdmap-utm-zone`
  - 定位 UTM zone，默认 `50`
- `--hdmap-utm-south`
  - 南半球时开启

说明：

- 如果不传 `--hdmap-sql`，`hdmap_bev`、`traj_p2lane`、`traj_p2lane_stable` 都不会生成。
- `traj_p2lane` 和 `traj_p2lane_stable` 与 `hdmap_bev` 共享同一个地图文件输入。

## 5. 输出内容

## 5.1 sample 中新增字段

每个 pkl sample 新增：

```python
sample_data["traj_p2lane"]
sample_data["traj_p2lane_stable"]
sample_data["ego_utm_position"]
sample_data["ego_yaw"]
```

其中：

- `ego_utm_position`
  - 当前 sample 对应自车的全局 UTM 位置，格式为 `np.ndarray(shape=(3,))`
- `ego_yaw`
  - 当前 sample 对应自车航向角，单位为弧度
  - 可用于后续把每帧 `hdmap_bev` 从当前 ego 局部系重新投影到统一参考系

这两个字段主要用于：

- HDMap 可视化调试
- world-stabilized 导出

相关说明见：

- [hdmap_bev_visualization_debug_guide.md](/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/doc/hdmap_bev_visualization_debug_guide.md)

## 5.2 `traj_p2lane`

格式为：

```python
[
    (matched_lane_id, s, d),
    ...
]
```

其中：

- `matched_lane_id` 为地图原始 lane / lane_link ID
- `s` 为相对当前匹配对象中心线的弧长坐标
- `d` 为相对当前匹配对象中心线的有符号横向偏移

### `matched_lane_id` 编码规则

对于 `lane`：

```text
road_pkid_section_id_lane_id
```

对于 `lane_link`：

```text
from_road_from_section_from_lane_to_road_to_section_to_lane
```

失败时写为：

```python
("", nan, nan)
```

## 5.3 `traj_p2lane_stable`

格式为：

```python
[
    (stable_lane_id, s, d, lane_relation),
    ...
]
```

其中：

- `stable_lane_id`
  - 对一对一连续的原始 lane segment 做归并后的稳定 ID
  - 当前命名为该 stable lane chain 的起点 `lane_key`
- `s`
  - 相对 merged stable lane 的累计弧长坐标
  - 已包含各子 segment 的累计 `s offset`
- `d`
  - 相对当前 stable lane segment 中心线的有符号横向偏移
- `lane_relation`
  - `keep` / `toleft` / `toright`

### `stable_lane_id` 合并条件

当前实现会在以下条件同时满足时，把两段原始 lane 合并为同一个稳定 ID：

1. 两段 lane 通过 `TX_LANE_LINK` 连通
2. 是一对一连续关系
3. 不处在分叉或汇合点
4. 两段端点距离很小
5. 两段 heading 基本一致
6. `lane_id` 相同

当前默认阈值为：

- 端点距离小于等于 `2.0m`
- heading 夹角小于等于 `15deg`

### `lane_relation` 语义

当前实现的行为是：

1. 第一个有效点写 `keep`
2. 若当前点与前一个有效 stable 点的 `stable_lane_id` 相同，写 `keep`
3. 若 `stable_lane_id` 不同，则用“当前点相对前一个有效 stable lane 中心线”的有符号横向偏移判断：
   - `d >= 0` 写 `toleft`
   - `d < 0` 写 `toright`

注意：

- 这里的基准是“前一个有效 stable lane”
- 不是严格的“前一个原始 matched lane / lane_link”

### 失败时

```python
("", nan, nan, "")
```

## 5.4 长度和时序

`traj_p2lane` 和 `traj_p2lane_stable` 长度都固定为 `11`，对应：

- 4 帧历史
- 1 帧当前
- 6 帧未来

顺序与 sample 构造窗口一致。

## 6. 跑完后怎么检查

## 6.1 检查 sample 是否带两个字段

```bash
python3 - <<'PY'
import pickle

sample_path = "/path/to/one_sample.pkl"
with open(sample_path, "rb") as f:
    data = pickle.load(f)

print("keys:", data.keys())
print("traj_p2lane len:", len(data.get("traj_p2lane", [])))
print("traj_p2lane_stable len:", len(data.get("traj_p2lane_stable", [])))
print("traj_p2lane:", data.get("traj_p2lane"))
print("traj_p2lane_stable:", data.get("traj_p2lane_stable"))
PY
```

预期：

- 能看到 `traj_p2lane`
- 能看到 `traj_p2lane_stable`
- 两者长度都为 `11`

## 6.2 快速检查匹配成功率

```bash
python3 - <<'PY'
import pickle

sample_path = "/path/to/one_sample.pkl"
with open(sample_path, "rb") as f:
    data = pickle.load(f)

traj_raw = data["traj_p2lane"]
traj_stable = data["traj_p2lane_stable"]

raw_matched = sum(1 for lane_id, s, d in traj_raw if lane_id)
stable_matched = sum(1 for lane_id, s, d, rel in traj_stable if lane_id)

print("raw matched / total =", raw_matched, "/", len(traj_raw))
print("stable matched / total =", stable_matched, "/", len(traj_stable))
print("stable values =")
for item in traj_stable:
    print(item)
PY
```

## 6.3 检查 stable `s` 是否连续

对于被合并成同一个 `stable_lane_id` 的连续 lane segment，预期：

- `stable_lane_id` 不变
- `traj_p2lane_stable` 中的 `s` 单调前进
- 切到下一原始 segment 时，`s` 不会回到 `0` 附近

可用下面脚本快速查看：

```bash
python3 - <<'PY'
import pickle

sample_path = "/path/to/one_sample.pkl"
with open(sample_path, "rb") as f:
    data = pickle.load(f)

for idx, item in enumerate(data["traj_p2lane_stable"]):
    print(idx, item)
PY
```

## 6.4 检查 `d` 和 `lane_relation` 是否合理

经验上：

- `d` 接近 `0` 时，点应在车道中心附近
- `d > 0` 表示点偏左
- `d < 0` 表示点偏右
- `lane_relation == keep` 时，当前点与前一个有效 stable 点应处在同一 stable lane
- `lane_relation == toleft / toright` 时，通常对应 stable lane 发生变化

## 7. JSON 调试建议

如果怀疑 pkl 不方便观察，建议加 `--write-json`。

可以直接 grep：

```bash
grep -R "\"traj_p2lane\"" /path/to/json_debug | head
grep -R "\"traj_p2lane_stable\"" /path/to/json_debug | head
```

重点查看：

- `traj_p2lane`
- `traj_p2lane_stable`
- `his_trajectory`
- `fur_trajectory`
- `navigation_command`

## 8. 常见问题

## 8.1 `traj_p2lane` / `traj_p2lane_stable` 全是空值

先检查：

1. 是否传了 `--hdmap-sql`
2. `hdmap-sql` 文件是否存在
3. `hdmap-utm-zone` 是否和定位坐标一致
4. `SPATIALITE_EXTENSION` 是否能正确加载
5. sampled localization 的 `position.x/y` 是否真的是 UTM

## 8.2 `traj_p2lane` 有跳变，但 `traj_p2lane_stable` 更平滑

这是正常现象。

原因是：

- `traj_p2lane` 输出的是地图原始分段 ID
- `traj_p2lane_stable` 会把一对一连续 segment 合并成稳定 ID

因此在直行连续段里，`traj_p2lane_stable` 的 ID 通常更稳定。

## 8.3 `traj_p2lane_stable` 的 `s` 为什么比原始 `s` 大很多

因为它不是当前 segment 的局部 `s`，而是 merged stable lane 的累计 `s`：

- 当前 stable segment 的局部投影 `s`
- 再加上前面已合并 segment 的累计长度 offset

这正是为了避免跨 segment 后 `s` 回到 0。

## 8.4 `lane_relation` 为什么和“前一个原始 lane”不完全一致

当前实现的基准是前一个有效 stable lane，不是前一个原始 matched lane。

这样做的好处是：

- 对 lane_link 抖动更稳
- 对连续分段 lane 更稳

如果后续需要严格按“前一个原始 lane”定义，可再单独调整实现。

## 9. 推荐使用流程

推荐按下面顺序使用：

1. 先用 `--limit 1` 跑单个切片
2. 打开一个 pkl 或 JSON，确认两个字段都存在且长度为 11
3. 抽几条 sample，看 `traj_p2lane_stable` 的 `stable_lane_id` 是否更平滑
4. 再检查 `stable s` 是否跨 segment 连续
5. 最后再跑全量

## 10. 结论

当前版本同时提供：

- 原始 `traj_p2lane`
- 稳定版 `traj_p2lane_stable`

推荐使用方式是：

- 保留 `traj_p2lane` 作为原始地图对象匹配结果
- 使用 `traj_p2lane_stable` 做更稳定的车道级特征或监督

日常使用时，只要：

- 正确提供 `--hdmap-sql`
- 确认 UTM zone 正确
- 跑完后同时检查 `traj_p2lane` 和 `traj_p2lane_stable`

就可以把这两条结果稳定用起来。
