#!/usr/bin/env python
"""Analyze KITTI/VoD 3D box regression errors.

This script is independent from training/evaluation code. It parses KITTI-style
label txt files, matches predictions to GT boxes within each frame and class,
then reports size, center, yaw, BEV IoU, and 3D IoU statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


EPS = 1e-9


@dataclass
class BoxRecord:
    frame_id: str
    cls_name: str
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    yaw: float
    score: Optional[float] = None


@dataclass
class MatchRecord:
    frame_id: str
    cls_name: str
    gt: BoxRecord
    pred: BoxRecord
    bev_iou: float
    iou3d: float
    box_mode: str = 'camera'

    @property
    def dx(self) -> float:
        return self.pred.x - self.gt.x

    @property
    def dy(self) -> float:
        return self.pred.y - self.gt.y

    @property
    def dz(self) -> float:
        return self.pred.z - self.gt.z

    @property
    def bev_center_error(self) -> float:
        if self.box_mode == 'lidar':
            return math.sqrt(self.dx * self.dx + self.dy * self.dy)
        return math.sqrt(self.dx * self.dx + self.dz * self.dz)

    @property
    def dl(self) -> float:
        return self.pred.l - self.gt.l

    @property
    def dw(self) -> float:
        return self.pred.w - self.gt.w

    @property
    def dh(self) -> float:
        return self.pred.h - self.gt.h

    @property
    def yaw_error(self) -> float:
        return normalize_angle(self.pred.yaw - self.gt.yaw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Diagnose KITTI/VoD 3D box regression errors.')
    parser.add_argument('--gt-dir', required=True, help='GT label txt directory.')
    parser.add_argument('--pred-dir', required=True, help='Prediction txt directory.')
    parser.add_argument(
        '--output-dir',
        default='work_dirs/analysis_vod_box_errors',
        help='Directory for CSV summaries and plots.')
    parser.add_argument(
        '--classes',
        nargs='+',
        default=['Car', 'Pedestrian', 'Cyclist'],
        help='Class names to analyze.')
    parser.add_argument(
        '--match-iou-thr',
        type=float,
        default=0.1,
        help='Minimum IoU for a GT/pred pair to be considered matched.')
    parser.add_argument(
        '--use-bev-match',
        action='store_true',
        help='Use BEV IoU instead of 3D IoU for greedy matching.')
    parser.add_argument(
        '--box-mode',
        choices=['camera', 'lidar'],
        default='camera',
        help=('Coordinate convention. KITTI/VoD txt is usually camera: '
              'BEV=(x,z), vertical=y, location is box bottom center. '
              'Lidar mode uses BEV=(x,y), vertical=z, location is box center.'))
    parser.add_argument(
        '--score-thr',
        type=float,
        default=None,
        help='Optional prediction score threshold. GT files ignore scores.')
    parser.add_argument('--no-plots', action='store_true', help='Skip plots.')
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_kitti_file(path: Path, frame_id: str, classes: Sequence[str],
                    is_pred: bool, score_thr: Optional[float]) -> List[BoxRecord]:
    """Read one KITTI-style label file.

    Expected fields:
    type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom
    height width length x y z rotation_y [score]
    """
    if not path.exists() or path.stat().st_size == 0:
        return []

    records: List[BoxRecord] = []
    with path.open('r', encoding='utf-8') as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                raise ValueError(
                    f'{path}:{line_idx} has {len(parts)} fields, expected at '
                    'least 15 KITTI fields.')

            cls_name = parts[0]
            if cls_name in ('DontCare', 'ignore') or cls_name not in classes:
                continue

            try:
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                yaw = float(parts[14])
                score = float(parts[15]) if len(parts) > 15 else None
            except ValueError as exc:
                raise ValueError(
                    f'{path}:{line_idx} cannot parse KITTI 3D fields: '
                    f'{raw_line.rstrip()}') from exc

            if is_pred and score_thr is not None:
                if score is None:
                    raise ValueError(
                        f'{path}:{line_idx} has no score, but --score-thr was set.')
                if score < score_thr:
                    continue

            if min(h, w, l) <= 0:
                raise ValueError(
                    f'{path}:{line_idx} has non-positive dimensions '
                    f'h={h}, w={w}, l={l}.')

            records.append(
                BoxRecord(
                    frame_id=frame_id,
                    cls_name=cls_name,
                    h=h,
                    w=w,
                    l=l,
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    score=score))
    return records


def load_label_dir(label_dir: Path, classes: Sequence[str], is_pred: bool,
                   score_thr: Optional[float]) -> Dict[str, List[BoxRecord]]:
    if not label_dir.exists():
        raise FileNotFoundError(f'Label directory does not exist: {label_dir}')
    frames: Dict[str, List[BoxRecord]] = {}
    for path in sorted(label_dir.glob('*.txt')):
        frame_id = path.stem
        frames[frame_id] = read_kitti_file(path, frame_id, classes, is_pred,
                                           score_thr)
    return frames


def rect_corners_bev(box: BoxRecord, box_mode: str) -> np.ndarray:
    """Return 4 rotated BEV rectangle corners."""
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    local = np.array(
        [[box.l / 2, box.w / 2], [box.l / 2, -box.w / 2],
         [-box.l / 2, -box.w / 2], [-box.l / 2, box.w / 2]],
        dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    corners = local @ rot.T
    center = np.array([box.x, box.z] if box_mode == 'camera' else [box.x, box.y],
                      dtype=np.float64)
    return corners + center


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    arr = np.asarray(poly, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float((np.dot(x, np.roll(y, -1)) -
                  np.dot(y, np.roll(x, -1))) * 0.5)


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_signed_area(poly) >= 0 else poly[::-1]


def line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray,
                      q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < EPS:
        return p2.copy()
    t = ((q1[0] - p1[0]) * s[1] - (q1[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def is_inside(point: np.ndarray, edge_start: np.ndarray,
              edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    rel = point - edge_start
    return edge[0] * rel[1] - edge[1] * rel[0] >= -EPS


def convex_polygon_intersection(subject: np.ndarray,
                                clipper: np.ndarray) -> List[np.ndarray]:
    """Sutherland-Hodgman clipping for convex polygons."""
    subject = ensure_ccw(subject)
    clipper = ensure_ccw(clipper)
    output: List[np.ndarray] = [p for p in subject]
    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        for curr in input_list:
            curr_inside = is_inside(curr, edge_start, edge_end)
            prev_inside = is_inside(prev, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(
                        line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(
                    line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
    return output


def bev_intersection_area(a: BoxRecord, b: BoxRecord, box_mode: str) -> float:
    poly_a = rect_corners_bev(a, box_mode)
    poly_b = rect_corners_bev(b, box_mode)
    return polygon_area(convex_polygon_intersection(poly_a, poly_b))


def vertical_interval(box: BoxRecord, box_mode: str) -> Tuple[float, float]:
    if box_mode == 'camera':
        # KITTI/VoD camera labels use y at the bottom face of the 3D box.
        return box.y - box.h, box.y
    # Most LiDAR boxes use z at the geometric center.
    return box.z - box.h / 2, box.z + box.h / 2


def box_iou_pair(a: BoxRecord, b: BoxRecord,
                 box_mode: str) -> Tuple[float, float]:
    inter_bev = bev_intersection_area(a, b, box_mode)
    area_a = a.l * a.w
    area_b = b.l * b.w
    bev_union = area_a + area_b - inter_bev
    bev_iou = inter_bev / bev_union if bev_union > EPS else 0.0

    a_min, a_max = vertical_interval(a, box_mode)
    b_min, b_max = vertical_interval(b, box_mode)
    inter_h = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    inter_vol = inter_bev * inter_h
    vol_a = area_a * a.h
    vol_b = area_b * b.h
    union_vol = vol_a + vol_b - inter_vol
    iou3d = inter_vol / union_vol if union_vol > EPS else 0.0
    return bev_iou, iou3d


def flatten_by_class(frames: Dict[str, List[BoxRecord]],
                     cls_name: str) -> List[BoxRecord]:
    return [
        box for boxes in frames.values() for box in boxes
        if box.cls_name == cls_name
    ]


def greedy_match_frame(gt_boxes: List[BoxRecord], pred_boxes: List[BoxRecord],
                       match_iou_thr: float, use_bev_match: bool,
                       box_mode: str) -> List[MatchRecord]:
    candidates: List[Tuple[float, int, int, float, float]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            bev_iou, iou3d = box_iou_pair(gt, pred, box_mode)
            match_iou = bev_iou if use_bev_match else iou3d
            if match_iou >= match_iou_thr:
                candidates.append((match_iou, gi, pi, bev_iou, iou3d))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches: List[MatchRecord] = []
    for _, gi, pi, bev_iou, iou3d in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt = gt_boxes[gi]
        pred = pred_boxes[pi]
        matches.append(
            MatchRecord(gt.frame_id, gt.cls_name, gt, pred, bev_iou, iou3d,
                        box_mode))
    return matches


def match_all(gt_frames: Dict[str, List[BoxRecord]],
              pred_frames: Dict[str, List[BoxRecord]], classes: Sequence[str],
              match_iou_thr: float, use_bev_match: bool,
              box_mode: str) -> Dict[str, List[MatchRecord]]:
    frame_ids = sorted(set(gt_frames) | set(pred_frames))
    matches_by_class: Dict[str, List[MatchRecord]] = {c: [] for c in classes}
    for frame_id in frame_ids:
        gt_boxes = gt_frames.get(frame_id, [])
        pred_boxes = pred_frames.get(frame_id, [])
        for cls_name in classes:
            cls_gt = [b for b in gt_boxes if b.cls_name == cls_name]
            cls_pred = [b for b in pred_boxes if b.cls_name == cls_name]
            matches_by_class[cls_name].extend(
                greedy_match_frame(cls_gt, cls_pred, match_iou_thr,
                                   use_bev_match, box_mode))
    return matches_by_class


def safe_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = safe_array(values)
    if arr.size == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'p90': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'p90': float(np.percentile(arr, 90)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'nan'
    return f'{value:.{digits}f}'


def size_bias_label(mean_diff: float, gt_mean: float) -> str:
    if np.isnan(mean_diff) or np.isnan(gt_mean) or gt_mean <= EPS:
        return 'unknown'
    rel = mean_diff / gt_mean
    if rel > 0.05:
        return 'pred larger'
    if rel < -0.05:
        return 'pred smaller'
    return 'close'


def collect_size_rows(gt_frames: Dict[str, List[BoxRecord]],
                      pred_frames: Dict[str, List[BoxRecord]],
                      matches_by_class: Dict[str, List[MatchRecord]],
                      classes: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cls_name in classes:
        gt_boxes = flatten_by_class(gt_frames, cls_name)
        pred_boxes = flatten_by_class(pred_frames, cls_name)
        matches = matches_by_class[cls_name]
        for dim in ('l', 'w', 'h'):
            gt_stat = stats(getattr(b, dim) for b in gt_boxes)
            pred_stat = stats(getattr(b, dim) for b in pred_boxes)
            diff_stat = stats(
                getattr(m.pred, dim) - getattr(m.gt, dim) for m in matches)
            rows.append({
                'class': cls_name,
                'dim': dim,
                'gt_count': gt_stat['count'],
                'gt_mean': gt_stat['mean'],
                'gt_std': gt_stat['std'],
                'gt_median': gt_stat['median'],
                'pred_count': pred_stat['count'],
                'pred_mean': pred_stat['mean'],
                'pred_std': pred_stat['std'],
                'pred_median': pred_stat['median'],
                'matched_pred_minus_gt_mean': diff_stat['mean'],
                'matched_pred_minus_gt_median': diff_stat['median'],
                'bias': size_bias_label(diff_stat['mean'], gt_stat['mean']),
            })
    return rows


def collect_match_rows(
        matches_by_class: Dict[str, List[MatchRecord]],
        classes: Sequence[str]) -> Tuple[List[Dict[str, object]],
                                         List[Dict[str, object]]]:
    sample_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    metrics = {
        'abs_dx': lambda m: abs(m.dx),
        'abs_dy': lambda m: abs(m.dy),
        'abs_dz': lambda m: abs(m.dz),
        'bev_center_error': lambda m: m.bev_center_error,
        'abs_dl': lambda m: abs(m.dl),
        'abs_dw': lambda m: abs(m.dw),
        'abs_dh': lambda m: abs(m.dh),
        'rel_l_error': lambda m: abs(m.dl) / max(m.gt.l, EPS),
        'rel_w_error': lambda m: abs(m.dw) / max(m.gt.w, EPS),
        'rel_h_error': lambda m: abs(m.dh) / max(m.gt.h, EPS),
        'abs_yaw_error_rad': lambda m: abs(m.yaw_error),
        'bev_iou': lambda m: m.bev_iou,
        'iou3d': lambda m: m.iou3d,
        'bev_minus_3d_iou': lambda m: m.bev_iou - m.iou3d,
    }

    for cls_name in classes:
        matches = matches_by_class[cls_name]
        for m in matches:
            sample_rows.append({
                'frame_id': m.frame_id,
                'class': cls_name,
                'gt_x': m.gt.x,
                'gt_y': m.gt.y,
                'gt_z': m.gt.z,
                'pred_x': m.pred.x,
                'pred_y': m.pred.y,
                'pred_z': m.pred.z,
                'dx': m.dx,
                'dy': m.dy,
                'dz': m.dz,
                'gt_l': m.gt.l,
                'gt_w': m.gt.w,
                'gt_h': m.gt.h,
                'pred_l': m.pred.l,
                'pred_w': m.pred.w,
                'pred_h': m.pred.h,
                'dl': m.dl,
                'dw': m.dw,
                'dh': m.dh,
                'yaw_error_rad': m.yaw_error,
                'yaw_error_deg': math.degrees(m.yaw_error),
                'bev_iou': m.bev_iou,
                'iou3d': m.iou3d,
                'score': m.pred.score,
            })
        for metric_name, getter in metrics.items():
            summary_rows.append({
                'class': cls_name,
                'metric': metric_name,
                **stats(getter(m) for m in matches),
            })
    return sample_rows, summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(summary_rows: List[Dict[str, object]], cls_name: str,
                   metric: str, key: str = 'mean') -> float:
    for row in summary_rows:
        if row['class'] == cls_name and row['metric'] == metric:
            return float(row[key])
    return np.nan


def max_finite(values: Sequence[float], default: float = np.nan) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return max(finite) if finite else default


def make_diagnosis(cls_name: str, size_rows: List[Dict[str, object]],
                   summary_rows: List[Dict[str, object]]) -> str:
    size_parts = []
    for dim in ('l', 'w', 'h'):
        row = next((r for r in size_rows
                    if r['class'] == cls_name and r['dim'] == dim), None)
        if row is None:
            continue
        mean_diff = float(row['matched_pred_minus_gt_mean'])
        gt_mean = float(row['gt_mean'])
        if np.isnan(mean_diff) or np.isnan(gt_mean) or gt_mean <= EPS:
            continue
        rel = mean_diff / gt_mean
        if abs(rel) > 0.05:
            direction = 'larger' if rel > 0 else 'smaller'
            size_parts.append(f'{dim} {direction} by {abs(rel) * 100:.1f}%')

    center = metric_summary(summary_rows, cls_name, 'bev_center_error', 'median')
    yaw = metric_summary(summary_rows, cls_name, 'abs_yaw_error_rad', 'median')
    bev_iou = metric_summary(summary_rows, cls_name, 'bev_iou', 'median')
    iou3d = metric_summary(summary_rows, cls_name, 'iou3d', 'median')
    gap = metric_summary(summary_rows, cls_name, 'bev_minus_3d_iou', 'median')
    rel_l = metric_summary(summary_rows, cls_name, 'rel_l_error', 'median')
    rel_w = metric_summary(summary_rows, cls_name, 'rel_w_error', 'median')
    rel_h = metric_summary(summary_rows, cls_name, 'rel_h_error', 'median')

    reasons = []
    if not np.isnan(center):
        if cls_name == 'Pedestrian' and center > 0.35:
            reasons.append('BEV center error is sensitive for small pedestrian boxes')
        elif cls_name != 'Pedestrian' and center > 0.7:
            reasons.append('BEV center error is relatively large')
    if not np.isnan(yaw) and yaw > 0.35:
        reasons.append(f'median yaw error is about {math.degrees(yaw):.1f} deg')
    if max_finite([rel_l, rel_w, rel_h], 0.0) > 0.12:
        reasons.append('matched-box relative size error is noticeable')
    if not np.isnan(gap) and gap > 0.15:
        reasons.append('BEV IoU is much higher than 3D IoU; check height, size, and vertical position')
    if not np.isnan(bev_iou) and not np.isnan(iou3d):
        if bev_iou < 0.25:
            reasons.append('BEV overlap is low, so position, yaw, and planar size may all matter')
        elif iou3d < 0.25:
            reasons.append('BEV is acceptable but 3D IoU is low')

    size_text = ('systematic size bias: ' + ', '.join(size_parts)
                 if size_parts else
                 'no obvious systematic larger/smaller size bias')
    main_text = ('likely bottleneck: ' + '; '.join(reasons)
                 if reasons else
                 'no single dominant regression factor is obvious; recall, classification, score filtering, or mixed factors may contribute')
    return f'{cls_name}: {size_text}. {main_text}.'


def write_markdown_report(path: Path, classes: Sequence[str],
                          size_rows: List[Dict[str, object]],
                          summary_rows: List[Dict[str, object]],
                          gt_frames: Dict[str, List[BoxRecord]],
                          pred_frames: Dict[str, List[BoxRecord]],
                          matches_by_class: Dict[str, List[MatchRecord]],
                          args: argparse.Namespace) -> None:
    lines = [
        '# VoD/KITTI Box Error Analysis',
        '',
        f'- GT dir: `{args.gt_dir}`',
        f'- Pred dir: `{args.pred_dir}`',
        f'- Box mode: `{args.box_mode}`',
        f'- Matching: `{"BEV IoU" if args.use_bev_match else "3D IoU"}` '
        f'>= `{args.match_iou_thr}` with greedy one-to-one matching',
        '',
        '## Per-Class Diagnosis',
        '',
    ]
    for cls_name in classes:
        gt_count = len(flatten_by_class(gt_frames, cls_name))
        pred_count = len(flatten_by_class(pred_frames, cls_name))
        match_count = len(matches_by_class[cls_name])
        lines.append(
            f'- {make_diagnosis(cls_name, size_rows, summary_rows)} '
            f'(GT={gt_count}, Pred={pred_count}, Matched={match_count})')

    lines += ['', '## Key Metrics', '']
    lines += [
        '| class | matched | median BEV IoU | median 3D IoU | '
        'median BEV center err | median yaw err deg | median BEV-3D gap |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for cls_name in classes:
        yaw_rad = metric_summary(summary_rows, cls_name, 'abs_yaw_error_rad',
                                 'median')
        yaw_deg = math.degrees(yaw_rad) if not np.isnan(yaw_rad) else np.nan
        lines.append(
            f'| {cls_name} | {len(matches_by_class[cls_name])} | '
            f'{fmt(metric_summary(summary_rows, cls_name, "bev_iou", "median"))} | '
            f'{fmt(metric_summary(summary_rows, cls_name, "iou3d", "median"))} | '
            f'{fmt(metric_summary(summary_rows, cls_name, "bev_center_error", "median"))} | '
            f'{fmt(yaw_deg)} | '
            f'{fmt(metric_summary(summary_rows, cls_name, "bev_minus_3d_iou", "median"))} |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def values_for_dim(boxes: List[BoxRecord], dim: str) -> np.ndarray:
    return safe_array(getattr(b, dim) for b in boxes)


def plot_hist(ax, values: np.ndarray, label: str, color: str) -> None:
    if values.size == 0:
        return
    ax.hist(values, bins=40, alpha=0.55, label=label, color=color)


def make_plots(output_dir: Path, classes: Sequence[str],
               gt_frames: Dict[str, List[BoxRecord]],
               pred_frames: Dict[str, List[BoxRecord]],
               matches_by_class: Dict[str, List[MatchRecord]],
               box_mode: str) -> None:
    if plt is None:
        print('matplotlib is not installed; skip plots.')
        return
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for cls_name in classes:
        gt_boxes = flatten_by_class(gt_frames, cls_name)
        pred_boxes = flatten_by_class(pred_frames, cls_name)
        matches = matches_by_class[cls_name]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, dim in zip(axes, ('l', 'w', 'h')):
            plot_hist(ax, values_for_dim(gt_boxes, dim), 'GT', '#4c78a8')
            plot_hist(ax, values_for_dim(pred_boxes, dim), 'Pred', '#f58518')
            ax.set_title(f'{cls_name} {dim}')
            ax.set_xlabel(dim)
            ax.set_ylabel('count')
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'{cls_name}_size_lwh_hist.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        center_err = safe_array(m.bev_center_error for m in matches)
        plot_hist(ax, center_err, 'BEV center error', '#54a24b')
        ax.set_title(f'{cls_name} center error')
        ax.set_xlabel('sqrt(dx^2 + dz^2)' if box_mode == 'camera' else
                      'sqrt(dx^2 + dy^2)')
        ax.set_ylabel('count')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'{cls_name}_center_error_hist.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        yaw_err_deg = safe_array(abs(math.degrees(m.yaw_error)) for m in matches)
        plot_hist(ax, yaw_err_deg, 'abs yaw error', '#e45756')
        ax.set_title(f'{cls_name} yaw error')
        ax.set_xlabel('abs yaw error (deg)')
        ax.set_ylabel('count')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'{cls_name}_yaw_error_hist.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        bev = safe_array(m.bev_iou for m in matches)
        iou3d = safe_array(m.iou3d for m in matches)
        if bev.size > 0:
            ax.scatter(bev, iou3d, s=8, alpha=0.5, color='#72b7b2')
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{cls_name} BEV IoU vs 3D IoU')
        ax.set_xlabel('BEV IoU')
        ax.set_ylabel('3D IoU')
        fig.tight_layout()
        fig.savefig(plot_dir / f'{cls_name}_bev_iou_vs_3d_iou.png', dpi=160)
        plt.close(fig)


def print_size_table(size_rows: List[Dict[str, object]],
                     classes: Sequence[str]) -> None:
    print('\n[1] GT/Prediction size distribution and matched pred-GT bias')
    for cls_name in classes:
        print(f'\n{cls_name}')
        for row in [r for r in size_rows if r['class'] == cls_name]:
            print(
                f"  {row['dim']}: "
                f"GT mean/std/median={fmt(row['gt_mean'])}/"
                f"{fmt(row['gt_std'])}/{fmt(row['gt_median'])}, "
                f"Pred mean/std/median={fmt(row['pred_mean'])}/"
                f"{fmt(row['pred_std'])}/{fmt(row['pred_median'])}, "
                f"matched pred-GT mean={fmt(row['matched_pred_minus_gt_mean'])} "
                f"({row['bias']})")


def print_match_table(summary_rows: List[Dict[str, object]],
                      matches_by_class: Dict[str, List[MatchRecord]],
                      classes: Sequence[str]) -> None:
    print('\n[2] Matched-pair error summary')
    metrics = [
        'bev_center_error', 'abs_dx', 'abs_dy', 'abs_dz', 'abs_dl', 'abs_dw',
        'abs_dh', 'rel_l_error', 'rel_w_error', 'rel_h_error',
        'abs_yaw_error_rad', 'bev_iou', 'iou3d', 'bev_minus_3d_iou'
    ]
    for cls_name in classes:
        print(f'\n{cls_name}: matched={len(matches_by_class[cls_name])}')
        for metric in metrics:
            row = next((r for r in summary_rows
                        if r['class'] == cls_name and r['metric'] == metric),
                       None)
            if row is None:
                continue
            suffix = ''
            if metric == 'abs_yaw_error_rad' and not np.isnan(row['median']):
                suffix = f' ({math.degrees(row["median"]):.2f} deg median)'
            print(
                f"  {metric}: mean={fmt(row['mean'])}, "
                f"median={fmt(row['median'])}, p90={fmt(row['p90'])}{suffix}")


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_frames = load_label_dir(gt_dir, args.classes, is_pred=False,
                               score_thr=None)
    pred_frames = load_label_dir(pred_dir, args.classes, is_pred=True,
                                 score_thr=args.score_thr)
    if not gt_frames:
        raise RuntimeError(f'No GT txt files found in {gt_dir}')
    if not pred_frames:
        print(f'Warning: no prediction txt files found in {pred_dir}')

    matches_by_class = match_all(gt_frames, pred_frames, args.classes,
                                 args.match_iou_thr, args.use_bev_match,
                                 args.box_mode)
    size_rows = collect_size_rows(gt_frames, pred_frames, matches_by_class,
                                  args.classes)
    sample_rows, summary_rows = collect_match_rows(matches_by_class,
                                                   args.classes)

    write_csv(output_dir / 'size_distribution_summary.csv', size_rows)
    write_csv(output_dir / 'matched_box_errors.csv', sample_rows)
    write_csv(output_dir / 'matched_error_summary.csv', summary_rows)
    write_markdown_report(output_dir / 'diagnosis_report.md', args.classes,
                          size_rows, summary_rows, gt_frames, pred_frames,
                          matches_by_class, args)
    if not args.no_plots:
        make_plots(output_dir, args.classes, gt_frames, pred_frames,
                   matches_by_class, args.box_mode)

    print(f'Loaded GT frames={len(gt_frames)}, prediction frames={len(pred_frames)}')
    print('Matching rule: same frame + same class + greedy one-to-one by '
          f'{"BEV IoU" if args.use_bev_match else "3D IoU"} '
          f'>= {args.match_iou_thr}')
    print_size_table(size_rows, args.classes)
    print_match_table(summary_rows, matches_by_class, args.classes)
    print('\n[3] Diagnosis')
    for cls_name in args.classes:
        print('  ' + make_diagnosis(cls_name, size_rows, summary_rows))
    print(f'\nSaved analysis to: {output_dir}')


if __name__ == '__main__':
    main()
