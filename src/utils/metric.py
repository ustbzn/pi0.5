from typing import List, Dict

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


def get_bucketed_action_accuracy(
    gt: torch.FloatTensor,        # [B, H, A]
    pred: torch.FloatTensor,      # [B, H, A]
    bucket_ids: torch.LongTensor, # [B]
    thresholds: List[float] = [0.05, 0.1],
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
            results[bucket_name] = {"l1": 0.0, "count": 0}
            for t in thresholds:
                results[bucket_name][f"acc@{t}"] = 0.0
            continue

        gt_b = gt[mask]
        pred_b = pred[mask]
        l1 = torch.nn.functional.l1_loss(pred_b, gt_b).item()

        diff = torch.abs(gt_b - pred_b).reshape(-1, gt_b.shape[-1])
        accs = {}
        for t in thresholds:
            acc = torch.mean(
                (torch.mean((diff < t).float(), dim=1) >= 1.0).float()
            ).item()
            accs[f"acc@{t}"] = acc

        results[bucket_name] = {"l1": l1, "count": count, **accs}

    return results
