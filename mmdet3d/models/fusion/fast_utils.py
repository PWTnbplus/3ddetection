import math
from typing import Optional, Sequence

import torch
from torch import Tensor


def build_image_stats(imgs,
                      batch_size: int,
                      device: torch.device,
                      dtype: torch.dtype) -> Tensor:
    if imgs is None:
        return torch.zeros((batch_size, 2), device=device, dtype=dtype)
    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(imgs)
    if imgs.device != device or imgs.dtype != dtype:
        imgs = imgs.to(device=device, dtype=dtype, non_blocking=True)
    dims = (1, 2, 3, 4) if imgs.dim() == 5 else (1, 2, 3)
    imgs_float = imgs.float()
    brightness = imgs_float.mean(dim=dims).sigmoid().to(dtype=dtype)
    variance = imgs_float.var(dim=dims, unbiased=False).clamp_min(0).sqrt()
    variance = (variance / (variance + 1.0)).clamp(0, 1).to(dtype=dtype)
    return torch.stack([brightness, variance], dim=1)


def _unwrap_points(points) -> Optional[Tensor]:
    if points is None:
        return None
    tensor = getattr(points, 'tensor', points)
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return tensor


def _point_xy_entropy(points: Tensor,
                      point_cloud_range: Optional[Sequence[float]],
                      bins: int = 8) -> float:
    if points.numel() == 0 or points.shape[1] < 2:
        return 0.0
    xy = points[:, :2].float()
    if point_cloud_range is None:
        xy_min = xy.min(dim=0).values
        xy_max = xy.max(dim=0).values
    else:
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        xy_min = xy.new_tensor([x_min, y_min])
        xy_max = xy.new_tensor([x_max, y_max])
    xy = (xy - xy_min) / (xy_max - xy_min).clamp_min(1e-6)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0) & (xy[:, 1] < 1)
    xy = xy[valid]
    if xy.numel() == 0:
        return 0.0
    xy_idx = (xy * bins).long().clamp(0, bins - 1)
    linear_idx = xy_idx[:, 1] * bins + xy_idx[:, 0]
    hist = torch.bincount(linear_idx, minlength=bins * bins).float()
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = -(probs * (probs + 1e-6).log()).sum()
    return float((entropy / math.log(bins * bins)).clamp(0, 1).item())


def build_point_stats(points_batch,
                      batch_size: int,
                      device: torch.device,
                      dtype: torch.dtype,
                      point_cloud_range: Optional[Sequence[float]]) -> Tensor:
    if points_batch is None:
        return torch.zeros((batch_size, 2), device=device, dtype=dtype)

    stats = []
    for points in points_batch[:batch_size]:
        tensor = _unwrap_points(points)
        if tensor is None or tensor.numel() == 0:
            stats.append((0.0, 0.0))
            continue
        count_score = min(math.log1p(int(tensor.shape[0])) / 10.0, 1.0)
        entropy = _point_xy_entropy(tensor, point_cloud_range)
        stats.append((count_score, entropy))

    if len(stats) < batch_size:
        stats.extend([(0.0, 0.0)] * (batch_size - len(stats)))

    return torch.as_tensor(stats, device=device, dtype=dtype)


def build_text_confidence(batch_data_samples,
                          batch_size: int,
                          device: torch.device,
                          dtype: torch.dtype) -> Tensor:
    if not batch_data_samples:
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    values = []
    for data_sample in batch_data_samples[:batch_size]:
        metainfo = data_sample.metainfo
        confidence = metainfo.get(
            'text_confidence',
            metainfo.get('text_conf', metainfo.get('llm_confidence', None)))
        if confidence is None:
            confidence = 1.0 if metainfo.get('text', '') else 0.0
        values.append(float(confidence))

    if len(values) < batch_size:
        values.extend([0.0] * (batch_size - len(values)))

    return torch.as_tensor(
        values[:batch_size], device=device, dtype=dtype).clamp(0, 1)[:, None]
