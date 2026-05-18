# `parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py` 中 `p2lane + stable Frenet` 技术说明

## 1. 背景与目标

本次实现的目标是在尽量不影响原脚本既有逻辑的前提下，为脚本：

- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py`

新增两条基于 **2Hz sampled localization** 的车道匹配数据流：

1. 原始 `traj_p2lane`
   - 输出 `(matched_lane_id, s, d)`
   - 保留地图原始 lane / lane_link 分段 ID
2. 稳定版 `traj_p2lane_stable`
   - 输出 `(stable_lane_id, s, d, lane_relation)`
   - 对一对一连续的 lane segment 做 stable merge

这样做的目的有两个：

- 保留原始地图对象级匹配结果，便于调试和追查
- 提供一份更稳定的 lane 级结果，减少“同一条物理车道被地图切成多段”带来的 ID 跳变

## 2. 改动范围

## 2.1 新增文件

- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/map/lane_frenet.py`
- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/map/__init__.py`

## 2.2 修改文件

- `/pcpt/pcpt/project/lijun/.code/Drive_pi/mogo-dataset-tools/parseRosBag-2O-Poutine_junv2_50f_merge_hdmap_p2lane.py`

接线改动主要包括：

- 初始化 `LaneFrenetMatcher`
- 基于 `sampled_2hz_locations` 预计算原始和稳定版 `p2lane`
- 在样本输出中同时新增：
  - `traj_p2lane`
  - `traj_p2lane_stable`

未改动原有：

- 图像对齐逻辑
- 2Hz 采样逻辑
- `his_trajectory` / `fur_trajectory` 生成逻辑
- 导航解析逻辑
- HDMap BEV 生成逻辑

## 3. 数据流设计

## 3.1 原有主路径

原脚本主路径为：

1. 加载图像目录
2. 按 `front` 相机对齐多相机
3. 解析 `/localization/global`
4. 做 2Hz 图像驱动采样，得到：
   - `self.sampled_2hz_timestamps`
   - `self.sampled_2hz_locations`
   - `self.sampled_2hz_indices`
5. 基于 sampled localization 生成 sample

## 3.2 新增数据流

在第 4 步和第 5 步之间增加：

1. 提取全部 sampled UTM 点
2. 一次性加载轨迹附近地图
3. 逐点计算原始匹配结果：
   - `matched_lane_id`
   - `s`
   - `d`
4. 逐点计算稳定版匹配结果：
   - `stable_lane_id`
   - `stable s`
   - `stable d`
   - `lane_relation`
5. 结果缓存到：
   - `self.sampled_2hz_p2lane`
   - `self.sampled_2hz_p2lane_stable`

生成每个 sample 时，再从上述两个缓存中切出当前窗口对应的结果，写入：

- `sample_data["traj_p2lane"]`
- `sample_data["traj_p2lane_stable"]`
- `sample_data["ego_utm_position"]`
- `sample_data["ego_yaw"]`

## 4. 输出字段说明

## 4.1 `self.sampled_2hz_p2lane`

类型：

```python
List[Tuple[str, float, float]]
```

每个元素为：

```python
(matched_lane_id, s, d)
```

## 4.2 `self.sampled_2hz_p2lane_stable`

类型：

```python
List[Tuple[str, float, float, str]]
```

每个元素为：

```python
(stable_lane_id, s, d, lane_relation)
```

## 4.3 `sample_data["traj_p2lane"]`

类型：

```python
List[Tuple[str, float, float]]
```

## 4.4 `sample_data["traj_p2lane_stable"]`

类型：

```python
List[Tuple[str, float, float, str]]
```

此外，sample 还会补充写入：

```python
sample_data["ego_utm_position"]  # np.ndarray(shape=(3,))
sample_data["ego_yaw"]           # float, rad
```

这两个字段不参与 `p2lane` 匹配本身，但用于后续：

- world-stabilized HDMap 可视化
- 将不同 sample 的 ego-centric 地图重新投影到统一参考系

`traj_p2lane` 和 `traj_p2lane_stable` 都与当前 sample 的时序窗口对齐，长度固定为：

- `4 history + 1 current + 6 future = 11`

索引顺序为：

1. 历史第 4 帧
2. 历史第 3 帧
3. 历史第 2 帧
4. 历史第 1 帧
5. 当前帧
6. 未来第 1 帧
7. 未来第 2 帧
8. 未来第 3 帧
9. 未来第 4 帧
10. 未来第 5 帧
11. 未来第 6 帧

## 4.5 ID 编码规则

### 原始 `matched_lane_id`

对于 `lane`：

```text
road_pkid_section_id_lane_id
```

对于 `lane_link`：

```text
from_road_from_section_from_lane_to_road_to_section_to_lane
```

### 稳定版 `stable_lane_id`

`stable_lane_id` 当前使用 merge 后 stable lane chain 的起点 `lane_key`，格式仍为：

```text
road_pkid_section_id_lane_id
```

它只对应稳定 lane group，不再直接表示单一原始地图 segment。

### 失败时

原始结果：

```python
("", nan, nan)
```

稳定结果：

```python
("", nan, nan, "")
```

## 5. 核心实现说明

## 5.1 地图加载

`LaneFrenetMatcher` 直接读取 Spatialite 地图中的：

- `TX_LANE`
- `TX_LANE_LINK`
- `TX_LANE_BOUNDARY`

并在轨迹包围盒外扩一定 margin 后，只加载附近数据，避免全图扫描。

读取内容如下。

`TX_LANE`：

- `road_pkid`
- `section_id`
- `lane_id`
- `left_boundary`
- `right_boundary`
- `geom`

`TX_LANE_LINK`：

- `pkid`
- `from_road_pkid`
- `from_section_id`
- `from_lane_id`
- `to_road_pkid`
- `to_section_id`
- `to_lane_id`
- `geom`

`TX_LANE_BOUNDARY`：

- `pkid`
- `geom`

## 5.2 坐标系处理

地图库中的几何为 WGS84，经 `pyproj` 转到 UTM 后参与匹配和 Frenet 计算。

脚本中的定位点已经是 UTM，因此匹配过程统一在 UTM 平面进行。

## 5.3 lane polygon 构造

参考 `DataMining_BassetHound` 的做法，lane 的判定不是直接靠中心线最近距离，而是优先构造 lane polygon：

1. 根据 `left_boundary` / `right_boundary` 查询边界几何
2. 左边界正向 + 右边界逆向，拼成封闭 polygon
3. 若 polygon 无效，则尝试 `buffer(0)` 修复
4. 若仍失败，则退化为中心线 `buffer`

这样可以更稳定地判断点是否落在 lane 内。

## 5.4 lane_link polygon 构造

`TX_LANE_LINK` 有时几何不完整，因此使用近似方法构造 lane_link polygon：

1. 若 `geom` 存在，则用其作为中心线
2. 若 `geom` 缺失，则退化为：
   - `from_lane` 中心线末点
   - 到 `to_lane` 中心线起点
   的直线
3. 根据 `from_lane` 末端宽度和 `to_lane` 起始宽度沿中心线插值
4. 左右偏置生成左右边界
5. 闭合为 polygon

## 5.5 逐点匹配策略

逐点匹配采用“连续性优先、全局回退”的策略。

### 第一步：优先尝试与前一时刻连续

对当前点，优先检查：

1. 是否仍落在上一时刻匹配对象的 polygon 内
2. 若不在，则检查上一对象的 `outgoing_ids` 对应对象及其下游 `to_lane` 是否包含当前点
3. 若仍未命中，则在上一对象、后继 lane_link 及其下游 `to_lane` 中做小范围最近邻回退

这里特意把 lane_link 的 `to_lane` 加入连续性候选，是为了避免在相邻车道连接关系很近时只按短 lane_link 中心线距离误选目标车道。

已验证过的典型问题场景：

- case：`0313-1/2026-03-13-14-24-09/20260313142634`
- previous lane：`32664_1_-2`
- outgoing lane_link：
  - `32664_1_-2_32580_1_-2`
  - `32664_1_-2_32580_1_-3`

旧逻辑只在两个 outgoing lane_link 中做包含或最近邻匹配。车辆实际已经进入 `32580_1_-2` 的 lane polygon，但两个短 lane_link 到当前点距离非常接近，可能误选 `32664_1_-2_32580_1_-3`，随后由于连续性约束持续匹配到 `32580_1_-3`。

修复后，`_match_with_previous()` 会把每个 outgoing lane_link 的 `to_lane` 一并纳入 transition candidates。当前点已经落入下游 lane polygon 时，会优先匹配真实 lane：

```text
previous_obj
  -> outgoing lane_link
  -> outgoing lane_link.to_lane
```

该 case 回归结果：

- 重新生成样本数：`50`
- 重新生成 `hdmap_bev_npz`：`50`
- `20260313142634_step16.pkl` 到 `20260313142634_step36.pkl` 的 `traj_p2lane` 中不再出现 `32580_1_-3` 或指向 `32580_1_-3` 的 lane_link
- 对应区间改为匹配 `32580_1_-2`，后续再自然进入 `32585_1_-2`

同一 case 还暴露了 stable 结果的后续问题：`traj_p2lane` 已正确从 `32664_1_-2` 延续到 `32580_1_-2`，但地图级 stable merge 因为 `32664_1_-2` 存在两个 outgoing 而没有合并 `32580_1_-2`，导致 `traj_p2lane_stable` 从 `32417_1_-2` 跳到 `32580_1_-2`，并生成 `toleft`。

这不符合车辆实际没有变道的语义。因此当前实现进一步加入“轨迹感知 stable continuity”：

1. 先按逐点匹配结果确定实际走到哪个下游 lane
2. 若前一 stable lane 和当前 lane 的 `lane_id` 相同
3. 且二者通过当前匹配链路连通
4. 且端点距离和 heading 连续性满足 stable merge 阈值
5. 则把当前 stable group 动态 alias 到前一 stable ID，并补上连续的 stable `s` offset

这层逻辑不会改变全局地图拓扑 merge，只在当前匹配轨迹上消除“分叉地图拓扑但实际直行保持车道”带来的 stable ID 跳变。

该 case 回归后：

- `20260313142634_step16.pkl` 到 `20260313142634_step36.pkl` 的 `traj_p2lane_stable` 保持 `32417_1_-2`
- 对应 `lane_relation` 全部为 `keep`
- stable `s` 在 `32664_1_-2 -> 32580_1_-2 -> 32585_1_-2` 上连续增长

### 第二步：全局匹配

若连续性匹配失败，则：

1. 在所有 `lane` 中查 polygon 命中
2. 若没有，再查所有 `lane_link`
3. 若仍没有，则分别在 lane / lane_link 中查中心线最近对象

## 5.6 原始 Frenet 计算

对于已匹配对象：

### `s`

直接使用中心线投影弧长：

```python
s = centerline.project(point)
```

### `d`

先得到投影点和该点处中心线航向 `heading`，再计算横向偏移：

```python
dx = x - ref_x
dy = y - ref_y
d = -sin(heading) * dx + cos(heading) * dy
```

符号约定：

- `d > 0` 表示点在中心线左侧
- `d < 0` 表示点在中心线右侧

## 5.7 stable lane merge 规则

当前实现会遍历所有轨迹附近的 `TX_LANE_LINK`，在以下条件同时满足时，把 `from_lane` 和 `to_lane` 合并到同一个 stable group：

1. 两段 lane 通过 `TX_LANE_LINK` 连通
2. `from_lane -> to_lane` 是一对一连续关系
3. `to_lane <- from_lane` 也是一对一连续关系
4. `lane_id` 相同
5. `from_lane` 末点到 `to_lane` 起点距离很小
6. 两段 centerline 端点 heading 基本一致

当前默认阈值为：

- 端点距离 `<= 2.0m`
- heading 差值 `<= 15deg`

这组规则的目标是：

- 合并“地图把同一条物理直行 lane 切成多个连续 segment”的情况
- 不穿过明显的分叉、汇合或复杂 lane topology

## 5.8 stable `s offset` 计算

仅合并 stable ID 还不够，因为如果直接复用原始 segment 的局部 `s`，车辆跨到下一段时 `s` 会回到 `0` 附近。

当前实现额外为每个 stable group 建立了累计 `s offset`：

1. 先记录所有成功 merge 的 `from_lane -> to_lane` 关系
2. 在每个 stable group 内恢复一条有序 lane chain
3. 为 chain 上每个 lane 记录其起始 `stable_s_offset`
4. 输出 stable Frenet 时使用：

```python
stable_s = stable_s_offset + local_s
```

因此：

- 同一 stable lane 内跨原始 segment 时，`stable s` 会连续增长
- 不会因为切到下一段而重新从 `0` 开始

## 5.9 stable `d` 计算

当前实现的 stable `d` 仍然是相对“当前选中的 stable lane segment 中心线”计算，而不是对整条 merge 后 polyline 重建一条全局中心线。

也就是说：

- `stable s` 是 merged stable lane 的累计坐标
- `stable d` 仍基于当前 lane segment 的局部中心线

这种做法的优点是实现简单、稳定，且与当前地图组织结构兼容。

## 5.10 `lane_relation` 定义

当前实现的 `lane_relation` 规则如下：

1. 若当前点无有效 stable lane，则写空字符串
2. 第一个有效 stable 点写 `keep`
3. 若当前点与前一个有效 stable 点的 `stable_lane_id` 相同，则写 `keep`
4. 若 `stable_lane_id` 不同，则计算“当前点相对前一个有效 stable lane 中心线”的有符号横向偏移：
   - `d >= 0` 写 `toleft`
   - `d < 0` 写 `toright`

注意这里的基准是：

- 前一个有效 stable lane

不是：

- 前一个原始 matched lane / lane_link

这样做的目的，是让关系判断更偏向稳定 lane 级语义，减少 lane_link 和原始分段带来的抖动。

## 5.11 lane_link 的 stable 归属

若当前点匹配到的是 `lane_link`：

- 若 `from_lane` 和 `to_lane` 属于同一个 stable group，则它们共享同一个 `stable_lane_id`
- 若它们不在同一个 stable group，当前实现默认把该 `lane_link` 归到 `to_lane` 的 stable ID

这意味着在跨 stable group 连接段上，稳定版结果会更偏向“即将进入的目标 stable lane”。

## 6. 原脚本中的接入点

## 6.1 初始化

在 `BagDataHandler.__init__` 中新增：

- `self.sampled_2hz_p2lane`
- `self.sampled_2hz_p2lane_stable`
- `self.lane_frenet_matcher`

当 `--hdmap-sql` 提供时：

- 原逻辑继续初始化 `HDMapBEVGenerator`
- 新逻辑额外初始化 `LaneFrenetMatcher`

若初始化失败：

- 只打印 warning
- 不影响原有样本生成

## 6.2 预计算 sampled p2lane

新增方法：

- `build_sampled_p2lane()`

在 `run()` 中调用顺序为：

1. `parse_localization()`
2. `sample_localization_2hz()`
3. `build_sampled_p2lane()`
4. `parse_routing()`
5. `generate_pkl_samples()`

## 6.3 样本输出

在 `_create_sample_data()` 中，按当前 `loc_idx` 切取：

- `loc_idx - 4`
- 到 `loc_idx + 6`

共 11 个项，分别写入：

```python
sample_data["traj_p2lane"]
sample_data["traj_p2lane_stable"]
```

## 7. 依赖项

新增逻辑依赖：

- `pyproj`
- `shapely`
- Spatialite 扩展

若系统未能自动加载 Spatialite，可通过环境变量指定：

```bash
export SPATIALITE_EXTENSION=/path/to/mod_spatialite.so
```

## 8. 验证建议

当前建议最少完成以下验证：

1. 对 `map/lane_frenet.py` 和主脚本执行 `py_compile`
2. 用 `--limit 1` 跑单切片 smoke test
3. 打开一个 sample，检查：
   - `traj_p2lane` 是否存在
   - `traj_p2lane_stable` 是否存在
   - 两者长度是否都是 `11`
4. 抽样检查 stable 结果：
   - `stable_lane_id` 是否比原始 ID 更平滑
   - `stable s` 是否跨连续 segment 单调前进
   - `lane_relation` 是否与直观左右关系大致一致

## 9. 已知限制

## 9.1 `traj_p2lane` 仍然是原始地图对象级结果

`traj_p2lane` 描述的是：

- 原始 lane / lane_link 匹配结果

因此在直行连续段里，若地图本身切成多个原始 segment，`matched_lane_id` 仍可能跳变。

## 9.2 stable merge 只覆盖一对一连续链

当前 stable merge 故意较保守：

- 不尝试穿过分叉或汇合点
- 不尝试合并复杂 topology

因此某些肉眼看起来“像一条路”的对象，可能不会被合并到同一个 `stable_lane_id`。

## 9.3 stable `d` 仍是局部 segment 中心线定义

当前没有为 merged stable lane 重建一条全局中心线。

因此：

- `stable s` 是累计坐标
- `stable d` 是局部 segment 参考系下的横向偏移

## 9.4 `lane_relation` 是 stable lane 级语义

当前 `lane_relation` 是相对前一个有效 stable lane 定义的。

如果后续需要严格的“相对前一个原始 matched lane”定义，需要单独调整实现。

## 9.5 匹配失败时使用空值

当前实现不会因为单点匹配失败中断整段 sample 生成，而是返回：

原始：

```python
("", nan, nan)
```

稳定版：

```python
("", nan, nan, "")
```

## 10. 后续可扩展方向

如后续需要更强可解释性或更细粒度监督，可继续扩展：

1. 输出对象类型：
   - `lane`
   - `lane_link`
2. 输出 `from_lane_id / to_lane_id`
3. 输出左 / 右边界距离
4. 输出归一化横向偏移
5. 在 sample 中拆分为：
   - `his_traj_p2lane`
   - `cur_p2lane`
   - `fur_traj_p2lane`
6. 若需要严格遵守“前一个原始 lane”语义，可单独重写 `lane_relation`
7. 若需要更物理一致的 stable Frenet，可为 merged stable lane 重建统一中心线

## 11. 结论

当前版本实现了以下目标：

- 保留原始 `traj_p2lane`
- 新增稳定版 `traj_p2lane_stable`
- 对一对一连续 lane segment 做 stable merge
- 为 merged stable lane 建立累计 `s offset`
- 输出 `lane_relation` 作为稳定 lane 级左右关系标签

整体上，这是一条附加型数据流：

- 不改变原始样本主结构
- 保留对象级原始匹配结果
- 额外补充一份更稳定的 lane 级监督特征
