# Copyright (c) OpenMMLab. All rights reserved.
"""Rotated IoU helpers for KITTI/VOD evaluation.

This module intentionally avoids a hard dependency on ``numba.cuda`` so the
KITTI metric can run on Windows-local environments that do not ship
``nvvm.dll`` / ``libNVVM``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

import numpy as np


EPS = 1e-7


def _normalize_backend_name(backend: Optional[str]) -> str:
    backend = (backend or os.getenv('MMD3D_ROTATE_IOU_BACKEND', 'auto'))
    backend = backend.strip().lower()
    if backend in ('', 'default'):
        backend = 'auto'
    if backend not in ('auto', 'mmcv', 'python'):
        raise ValueError(
            'MMD3D_ROTATE_IOU_BACKEND must be one of '
            '"auto", "mmcv", or "python".')
    return backend


def _as_boxes_array(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f'Expected boxes with shape [N, 5], got {boxes.shape}')
    return boxes


@lru_cache(maxsize=1)
def _load_mmcv_box_iou_rotated():
    try:
        from mmcv.ops import box_iou_rotated
    except Exception:
        return None
    return box_iou_rotated


def _probe_numba_cuda_nvvm() -> Dict[str, Optional[str]]:
    status: Dict[str, Optional[str]] = dict(
        installed=False, cuda_available=False, nvvm_available=False, error=None)
    try:
        from numba import cuda
    except Exception as exc:
        status['error'] = f'numba.cuda import failed: {exc}'
        return status

    status['installed'] = True
    try:
        status['cuda_available'] = bool(cuda.is_available())
    except Exception as exc:
        status['error'] = f'numba.cuda.is_available failed: {exc}'
        return status

    try:
        import numba.cuda.cudadrv.nvvm as nvvm

        nvvm.NVVM()
        status['nvvm_available'] = True
    except Exception as exc:
        status['error'] = str(exc)
    return status


def probe_rotate_iou_backends(
        backend: Optional[str] = None) -> Dict[str, Optional[str]]:
    requested = _normalize_backend_name(backend)
    mmcv_available = _load_mmcv_box_iou_rotated() is not None
    numba_status = _probe_numba_cuda_nvvm()

    if requested == 'mmcv':
        selected = 'mmcv' if mmcv_available else 'python'
    elif requested == 'python':
        selected = 'python'
    else:
        selected = 'mmcv' if mmcv_available else 'python'

    reason_parts = []
    if selected == 'mmcv':
        if requested in ('auto', 'mmcv'):
            reason_parts.append('using mmcv.ops.box_iou_rotated')
        else:
            reason_parts.append('explicit python backend requested')
    else:
        if requested == 'mmcv' and not mmcv_available:
            reason_parts.append('mmcv rotated IoU op unavailable, falling back')
        else:
            reason_parts.append('pure Python fallback selected')

    if numba_status['installed'] and not numba_status['nvvm_available']:
        reason_parts.append('numba CUDA NVVM unavailable')

    return dict(
        requested_backend=requested,
        selected_backend=selected,
        mmcv_available=mmcv_available,
        python_available=True,
        numba_cuda_installed=numba_status['installed'],
        numba_cuda_available=numba_status['cuda_available'],
        numba_nvvm_available=numba_status['nvvm_available'],
        numba_error=numba_status['error'],
        reason='; '.join(reason_parts))


def _rbbox_to_corners(rbbox: np.ndarray) -> np.ndarray:
    """Convert one rotated box to four corners.

    The formula matches the previous numba CUDA implementation exactly.
    """

    center_x, center_y, width, height, angle = map(float, rbbox)
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)

    local_corners = np.array([
        [-width / 2.0, -height / 2.0],
        [-width / 2.0, height / 2.0],
        [width / 2.0, height / 2.0],
        [width / 2.0, -height / 2.0],
    ],
                             dtype=np.float64)
    rotation = np.array([[a_cos, a_sin], [-a_sin, a_cos]], dtype=np.float64)
    corners = local_corners @ rotation.T
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    return corners


def _signed_polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_area(points: np.ndarray) -> float:
    return abs(_signed_polygon_area(points))


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray,
                       q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-12:
        return p2.copy()
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denominator
    return p1 + t * r


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray,
            orientation: float) -> bool:
    cross = ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
             (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]))
    return cross * orientation >= -1e-9


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if len(output) == 0:
        return output

    orientation = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0
    edge_start = clip[-1]
    for edge_end in clip:
        input_points = output
        if len(input_points) == 0:
            break
        clipped_points = []
        start_point = input_points[-1]
        for end_point in input_points:
            end_inside = _inside(end_point, edge_start, edge_end, orientation)
            start_inside = _inside(start_point, edge_start, edge_end,
                                   orientation)
            if end_inside:
                if not start_inside:
                    clipped_points.append(
                        _line_intersection(start_point, end_point, edge_start,
                                           edge_end))
                clipped_points.append(end_point)
            elif start_inside:
                clipped_points.append(
                    _line_intersection(start_point, end_point, edge_start,
                                       edge_end))
            start_point = end_point
        edge_start = edge_end
        output = np.asarray(clipped_points, dtype=np.float64)
    return output


def _intersection_area(box: np.ndarray, query_box: np.ndarray) -> float:
    clipped = _clip_polygon(_rbbox_to_corners(box), _rbbox_to_corners(query_box))
    if len(clipped) < 3:
        return 0.0
    return _polygon_area(clipped)


def _rotate_iou_python_eval(boxes: np.ndarray,
                            query_boxes: np.ndarray,
                            criterion: int = -1) -> np.ndarray:
    boxes = _as_boxes_array(boxes)
    query_boxes = _as_boxes_array(query_boxes)
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return overlaps.astype(boxes.dtype)

    box_areas = boxes[:, 2] * boxes[:, 3]
    query_box_areas = query_boxes[:, 2] * query_boxes[:, 3]
    for i, box in enumerate(boxes):
        for j, query_box in enumerate(query_boxes):
            inter_area = _intersection_area(box, query_box)
            if inter_area <= 0:
                continue
            if criterion == -1:
                denominator = box_areas[i] + query_box_areas[j] - inter_area
                overlaps[i, j] = inter_area / max(denominator, EPS)
            elif criterion == 0:
                overlaps[i, j] = inter_area / max(box_areas[i], EPS)
            elif criterion == 1:
                overlaps[i, j] = inter_area / max(query_box_areas[j], EPS)
            else:
                overlaps[i, j] = inter_area
    return overlaps.astype(boxes.dtype, copy=False)


def _rotate_iou_mmcv_eval(boxes: np.ndarray,
                          query_boxes: np.ndarray,
                          criterion: int = -1) -> np.ndarray:
    box_iou_rotated = _load_mmcv_box_iou_rotated()
    if box_iou_rotated is None:
        raise RuntimeError('mmcv.ops.box_iou_rotated is unavailable')

    import torch

    boxes = _as_boxes_array(boxes)
    query_boxes = _as_boxes_array(query_boxes)
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return np.zeros((boxes.shape[0], query_boxes.shape[0]),
                        dtype=boxes.dtype)

    boxes_tensor = torch.from_numpy(boxes)
    query_boxes_tensor = torch.from_numpy(query_boxes)
    with torch.no_grad():
        if criterion == -1:
            overlaps = box_iou_rotated(
                boxes_tensor,
                query_boxes_tensor,
                mode='iou',
                aligned=False,
                clockwise=False)
        elif criterion == 0:
            overlaps = box_iou_rotated(
                boxes_tensor,
                query_boxes_tensor,
                mode='iof',
                aligned=False,
                clockwise=False)
        elif criterion == 1:
            overlaps = box_iou_rotated(
                query_boxes_tensor,
                boxes_tensor,
                mode='iof',
                aligned=False,
                clockwise=False).t()
        else:
            iou = box_iou_rotated(
                boxes_tensor,
                query_boxes_tensor,
                mode='iou',
                aligned=False,
                clockwise=False)
            box_areas = boxes_tensor[:, 2] * boxes_tensor[:, 3]
            query_box_areas = query_boxes_tensor[:, 2] * query_boxes_tensor[:,
                                                                             3]
            overlaps = iou * (box_areas[:, None] + query_box_areas[None, :]) / (
                1.0 + iou).clamp_min(EPS)

    return overlaps.cpu().numpy().astype(boxes.dtype, copy=False)


def rotate_iou_eval(boxes: np.ndarray,
                    query_boxes: np.ndarray,
                    criterion: int = -1,
                    backend: Optional[str] = None) -> np.ndarray:
    backend_info = probe_rotate_iou_backends(backend)
    selected_backend = backend_info['selected_backend']
    if selected_backend == 'mmcv':
        return _rotate_iou_mmcv_eval(boxes, query_boxes, criterion)
    return _rotate_iou_python_eval(boxes, query_boxes, criterion)


def rotate_iou_gpu_eval(boxes: np.ndarray,
                        query_boxes: np.ndarray,
                        criterion: int = -1,
                        device_id: int = 0) -> np.ndarray:
    """Backward-compatible wrapper used by KITTI eval.

    ``device_id`` is kept for API compatibility but is no longer required by
    the Windows-local evaluation path.
    """

    del device_id
    return rotate_iou_eval(boxes, query_boxes, criterion)
