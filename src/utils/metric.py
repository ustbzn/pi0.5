from typing import Dict, List, Optional

import torch


# ===== 方案 D: 分桶名称映射 =====
BUCKET_NAMES = {
    0: "straight",
    1: "turn",
    2: "static",
    3: "high_curvature",
}


def get_action_accuracy(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds: List[float] = [0.1, 0.2],
) -> torch.FloatTensor:
    device = gt.device
    diff = torch.abs(gt - pred).reshape(-1, gt.shape[-1])

    # get the percentage of diff lower than threshold for all action dimensions
    accuracies = torch.zeros(len(thresholds), device=device)
    for idx, threshold in enumerate(thresholds):
        accuracy = torch.mean(
            (torch.mean((diff < threshold).float(), dim=1) >= 1.0).float()
        )
        accuracies[idx] = accuracy
    return accuracies


def get_action_accuracy_counts(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds: List[float] = [0.1, 0.2],
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """返回 Acc@threshold 的正确 waypoint 数和总 waypoint 数，用于跨 batch/rank 精确聚合。"""
    device = gt.device
    diff = torch.abs(gt.float() - pred.float()).reshape(-1, gt.shape[-1])
    correct_counts = torch.zeros(len(thresholds), device=device)
    for idx, threshold in enumerate(thresholds):
        correct_counts[idx] = (
            torch.mean((diff < threshold).float(), dim=1) >= 1.0
        ).float().sum()
    total_count = torch.tensor(float(diff.shape[0]), device=device)
    return correct_counts, total_count


def get_action_l1_sum_count(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """返回 L1 误差绝对值总和和元素总数，用于跨 batch/rank 精确聚合。"""
    diff = torch.abs(pred.float() - gt.float())
    return diff.sum(), torch.tensor(float(diff.numel()), device=gt.device)


def get_action_avg_l2(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
) -> torch.FloatTensor:
    """计算 Avg. L2：每个 future waypoint 的欧氏距离，再对 batch 和 horizon 求均值。"""
    return torch.linalg.vector_norm(pred.float() - gt.float(), ord=2, dim=-1).mean()


def get_action_avg_l2_sum_count(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """返回 waypoint L2 距离总和和 waypoint 总数，用于跨 batch/rank 精确聚合。"""
    l2 = torch.linalg.vector_norm(pred.float() - gt.float(), ord=2, dim=-1)
    return l2.sum(), torch.tensor(float(l2.numel()), device=gt.device)


def _denormalize_xy(
    trajectory: torch.FloatTensor,
    normalizer,
) -> torch.FloatTensor:
    """将归一化轨迹的前两维 [x, y] 反归一化到真实坐标系。"""
    trajectory = trajectory.float()
    if trajectory.shape[-1] < 2:
        raise ValueError(
            f"Avg. L2 requires at least 2 action dims, got {trajectory.shape[-1]}"
        )

    def _range(p01, p99, vmin, vmax):
        if normalizer.norm_mode == "minmax" and abs(float(vmax) - float(vmin)) > 1e-8:
            return vmin, vmax
        return p01, p99

    x_low, x_high = _range(
        normalizer.fur_x_01,
        normalizer.fur_x_99,
        normalizer.fur_x_min,
        normalizer.fur_x_max,
    )
    y_low, y_high = _range(
        normalizer.fur_y_01,
        normalizer.fur_y_99,
        normalizer.fur_y_min,
        normalizer.fur_y_max,
    )

    lows = torch.tensor(
        [x_low, y_low],
        device=trajectory.device,
        dtype=trajectory.dtype,
    )
    highs = torch.tensor(
        [x_high, y_high],
        device=trajectory.device,
        dtype=trajectory.dtype,
    )
    return (trajectory[..., :2] + 1.0) * (highs - lows) / 2.0 + lows


def get_denormalized_action_avg_l2(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    normalizer,
) -> torch.FloatTensor:
    """计算真实坐标系 Avg. L2，与开环轨迹评测口径一致。"""
    gt_xy = _denormalize_xy(gt, normalizer)
    pred_xy = _denormalize_xy(pred, normalizer)
    return get_action_avg_l2(gt_xy, pred_xy)


def get_denormalized_action_avg_l2_sum_count(
    gt: torch.FloatTensor,
    pred: torch.FloatTensor,
    normalizer,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """返回真实坐标系 waypoint L2 总和和 waypoint 总数。"""
    gt_xy = _denormalize_xy(gt, normalizer)
    pred_xy = _denormalize_xy(pred, normalizer)
    return get_action_avg_l2_sum_count(gt_xy, pred_xy)


def get_bucketed_action_accuracy(
    gt: torch.FloatTensor,        # [B, H, A]
    pred: torch.FloatTensor,      # [B, H, A]
    bucket_ids: torch.LongTensor, # [B]
    thresholds: List[float] = [0.05, 0.1],
    avg_l2_normalizer: Optional[object] = None,
) -> Dict[str, Dict[str, float]]:
    """按 trajectory_bucket 分桶计算 L1 和 accuracy。

    方案 D 核心函数：让 VLM 收益不被 easy straight 样本的均值掩盖，
    分别报告 straight / turn / static / high_curvature 桶上的指标。

    Args:
        gt:         ground-truth trajectory, [B, H, A]
        pred:       predicted trajectory, [B, H, A]
        bucket_ids: 每个样本的分桶 id (0=straight,1=turn,2=static,3=high_curvature)
        thresholds: accuracy 阈值列表

    Returns:
        dict: {bucket_name: {"l1": float, "acc@0.05": float, ..., "count": int}}
    """
    device = gt.device
    results = {}

    for bucket_id, bucket_name in BUCKET_NAMES.items():
        mask = (bucket_ids == bucket_id)
        count = mask.sum().item()
        if count == 0:
            results[bucket_name] = {"l1": 0.0, "avg_l2": 0.0, "count": 0}
            for t in thresholds:
                results[bucket_name][f"acc@{t}"] = 0.0
            continue

        gt_b = gt[mask]
        pred_b = pred[mask]
        l1_sum, l1_count = get_action_l1_sum_count(gt_b, pred_b)
        l1 = (l1_sum / torch.clamp(l1_count, min=1.0)).item()
        if avg_l2_normalizer is None:
            avg_l2 = get_action_avg_l2(gt_b, pred_b).item()
        else:
            avg_l2 = get_denormalized_action_avg_l2(
                gt_b,
            pred_b,
            avg_l2_normalizer,
        ).item()

        diff = torch.abs(gt_b.float() - pred_b.float()).reshape(-1, gt_b.shape[-1])
        accs = {}
        for t in thresholds:
            acc = torch.mean(
                (torch.mean((diff < t).float(), dim=1) >= 1.0).float()
            ).item()
            accs[f"acc@{t}"] = acc

        results[bucket_name] = {"l1": l1, "avg_l2": avg_l2, "count": count, **accs}

    return results
