from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_converters import radar_converter
from tools.dataset_converters import radar_llm_prompt_converter
from tools.dataset_converters import radar_text_converter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark the optimized VOD/radar data pipeline.')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Dataset root with ImageSets/training/testing.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='work_dirs/perf_benchmark',
        help='Directory to save generated benchmark artifacts and JSON.')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads used by converter/text/prompt stages.')
    parser.add_argument(
        '--text-hash-dim',
        type=int,
        default=768,
        help='Hash dimension used for text-embedding micro benchmark.')
    parser.add_argument(
        '--repeat',
        type=int,
        default=50,
        help='Repeat count used by the text-hash micro benchmark.')
    parser.add_argument(
        '--skip-lidar-check',
        action='store_true',
        help='Skip 7D radar shape checks when benchmarking a lidar-style root.')
    return parser.parse_args()


def load_text_hash_module(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        'text_hash_module', repo_root / 'mmdet3d' / 'utils' / 'text_hash.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def benchmark_radar_converter(data_root: Path,
                              out_dir: Path,
                              workers: int,
                              check_lidar: bool) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    infos = radar_converter.create_vod_infos(
        str(data_root),
        out_dir=str(out_dir),
        pkl_prefix='benchmark',
        splits=['train', 'val', 'test'],
        check_lidar=check_lidar,
        workers=workers)
    elapsed = time.perf_counter() - start
    counts = {split: len(records) for split, records in infos.items()}
    return {
        'elapsed_sec': elapsed,
        'workers': workers,
        'check_lidar': check_lidar,
        'counts': counts,
        'samples_total': sum(counts.values())
    }


def benchmark_radar_text(data_root: Path, out_dir: Path, workers: int) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = data_root / 'training' / 'label_2'
    image_sets_dir = data_root / 'ImageSets'
    split_counts = {}
    start = time.perf_counter()
    for split in ['train', 'val', 'train_val', 'full']:
        sample_ids = radar_text_converter.read_split_ids(
            image_sets_dir / f'{split}.txt')
        records = radar_text_converter.build_records_for_split(
            sample_ids, label_dir, workers=workers)
        radar_text_converter.dump_json(
            records, out_dir / f'radar_texts_{split}.json', pretty=False)
        split_counts[split] = len(records)
    elapsed = time.perf_counter() - start
    return {
        'elapsed_sec': elapsed,
        'workers': workers,
        'counts': split_counts,
        'records_total': sum(split_counts.values())
    }


def benchmark_llm_prompts(input_file: Path, output_file: Path,
                          workers: int) -> dict:
    if output_file.exists():
        output_file.unlink()
    start = time.perf_counter()
    records = radar_llm_prompt_converter.load_json(input_file)
    if workers <= 1:
        prompt_records = [
            radar_llm_prompt_converter.build_prompt_record(record, 30, 0.0, 'en')
            for record in records
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prompt_records = list(
                executor.map(
                    lambda record: radar_llm_prompt_converter.build_prompt_record(
                        record, 30, 0.0, 'en'), records))
    radar_llm_prompt_converter.dump_jsonl(prompt_records, output_file)
    elapsed = time.perf_counter() - start
    return {
        'elapsed_sec': elapsed,
        'workers': workers,
        'records_total': len(prompt_records)
    }


def benchmark_text_hash(input_file: Path, repo_root: Path, text_hash_dim: int,
                        repeat: int) -> dict:
    text_hash = load_text_hash_module(repo_root)
    records = json.loads(input_file.read_text(encoding='utf-8'))
    texts = [record.get('text', '') for record in records]
    precomputed = [
        text_hash.hash_text_to_numpy(text, text_hash_dim) for text in texts
    ]

    start = time.perf_counter()
    for _ in range(repeat):
        tensor = torch.stack([torch.from_numpy(vec) for vec in precomputed], dim=0)
        _ = tensor.float()
    elapsed = time.perf_counter() - start
    return {
        'elapsed_sec': elapsed,
        'repeat': repeat,
        'per_iter_sec': elapsed / max(repeat, 1),
        'records_total': len(texts),
        'text_hash_dim': text_hash_dim
    }


def sync_if_needed(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_callable(fn,
                       device: torch.device,
                       repeat: int = 40,
                       warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_if_needed(device)
    return (time.perf_counter() - start) / max(repeat, 1)


def build_synthetic_samples(batch_size: int, hash_dim: int):
    samples = []
    lidar2img = np.array(
        [[20.0, 0.0, 0.0, 20.0], [0.0, -16.0, 0.0, 128.0],
         [0.0, 0.0, 1.0, 20.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32)
    for idx in range(batch_size):
        text_embedding = None
        if idx % 3 == 0:
            text_embedding = np.random.randn(hash_dim).astype(np.float32)
            text_embedding /= np.linalg.norm(text_embedding) + 1e-6
        samples.append(
            SimpleNamespace(
                metainfo=dict(
                    text=(
                        'car ahead and pedestrian near the right curb with '
                        f'cyclist closing in pattern {idx % 5}'),
                    text_embedding=text_embedding,
                    img_shape=(256, 704),
                    lidar2img=lidar2img,
                    transformation_3d_flow=[])))
    return samples


def legacy_build_hashed_text_tensor(text_hash_module,
                                    texts,
                                    hash_dim: int,
                                    device: torch.device,
                                    dtype: torch.dtype) -> torch.Tensor:
    texts = list(texts)
    text_feats = torch.zeros((len(texts), hash_dim), device=device, dtype=dtype)
    for row, text in enumerate(texts):
        indices, values = text_hash_module._hashed_token_values(text, hash_dim)
        if not indices:
            continue
        index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        value_tensor = torch.tensor(values, device=device, dtype=dtype)
        text_feats[row].index_add_(0, index_tensor, value_tensor)
    return F.normalize(text_feats, dim=1)


def legacy_build_sample_text_features(text_hash_module,
                                      batch_data_samples,
                                      hash_dim: int,
                                      device: torch.device,
                                      dtype: torch.dtype) -> torch.Tensor:
    text_embeds = []
    texts = []
    missing_rows = []
    for row, sample in enumerate(batch_data_samples):
        metainfo = sample.metainfo
        text_embed = metainfo.get('text_embedding',
                                  metainfo.get('text_feat', None))
        if text_embed is None:
            text_embeds.append(None)
            texts.append(str(metainfo.get('text', '')))
            missing_rows.append(row)
            continue
        text_embed = torch.as_tensor(
            text_embed, device=device, dtype=dtype).flatten()
        if text_embed.numel() < hash_dim:
            text_embed = F.pad(text_embed, (0, hash_dim - text_embed.numel()))
        text_embeds.append(text_embed[:hash_dim])
        texts.append('')

    text_feats = torch.zeros(
        (len(batch_data_samples), hash_dim), device=device, dtype=dtype)
    if missing_rows:
        hashed_texts = legacy_build_hashed_text_tensor(
            text_hash_module, [texts[row] for row in missing_rows], hash_dim,
            device, dtype)
        text_feats[torch.tensor(missing_rows, device=device)] = hashed_texts

    for row, text_embed in enumerate(text_embeds):
        if text_embed is not None:
            text_feats[row] = text_embed
    return F.normalize(text_feats, dim=1)


def legacy_point_stats(points_batch, batch_size: int, device: torch.device,
                       dtype: torch.dtype, point_cloud_range) -> torch.Tensor:
    counts = []
    entropies = []
    for points in points_batch:
        points = points.detach().to(device=device, dtype=dtype)
        counts.append(
            torch.as_tensor(
                min(math.log1p(points.shape[0]) / 10.0, 1.0),
                device=device,
                dtype=dtype))
        if points.numel() == 0 or points.shape[1] < 2:
            entropies.append(torch.zeros((), device=device, dtype=dtype))
            continue
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        xy_min = torch.as_tensor([x_min, y_min], device=device, dtype=dtype)
        xy_max = torch.as_tensor([x_max, y_max], device=device, dtype=dtype)
        xy = (points[:, :2] - xy_min) / (xy_max - xy_min).clamp_min(1e-6)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0)
                 & (xy[:, 1] < 1))
        xy = xy[valid]
        if xy.numel() == 0:
            entropies.append(torch.zeros((), device=device, dtype=dtype))
            continue
        bins = 8
        xy_idx = (xy * bins).long().clamp(0, bins - 1)
        linear_idx = xy_idx[:, 1] * bins + xy_idx[:, 0]
        hist = torch.bincount(linear_idx, minlength=bins * bins).to(dtype)
        probs = hist / hist.sum().clamp_min(1.0)
        entropy = -(probs * (probs + 1e-6).log()).sum()
        entropies.append((entropy / math.log(bins * bins)).clamp(0, 1))
    if len(counts) < batch_size:
        pad = batch_size - len(counts)
        counts.extend([torch.zeros((), device=device, dtype=dtype)] * pad)
        entropies.extend([torch.zeros((), device=device, dtype=dtype)] * pad)
    return torch.stack(
        [torch.stack(counts[:batch_size]),
         torch.stack(entropies[:batch_size])],
        dim=1)


def fast_point_stats(points_batch, batch_size: int, device: torch.device,
                     dtype: torch.dtype, point_cloud_range) -> torch.Tensor:
    stats = []
    for points in points_batch[:batch_size]:
        if points.numel() == 0 or points.shape[1] < 2:
            stats.append((0.0, 0.0))
            continue
        count_score = min(math.log1p(int(points.shape[0])) / 10.0, 1.0)
        xy = points[:, :2].float()
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        xy_min = xy.new_tensor([x_min, y_min])
        xy_max = xy.new_tensor([x_max, y_max])
        xy = (xy - xy_min) / (xy_max - xy_min).clamp_min(1e-6)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0)
                 & (xy[:, 1] < 1))
        xy = xy[valid]
        if xy.numel() == 0:
            stats.append((count_score, 0.0))
            continue
        bins = 8
        xy_idx = (xy * bins).long().clamp(0, bins - 1)
        linear_idx = xy_idx[:, 1] * bins + xy_idx[:, 0]
        hist = torch.bincount(linear_idx, minlength=bins * bins).float()
        probs = hist / hist.sum().clamp_min(1.0)
        entropy = -(probs * (probs + 1e-6).log()).sum()
        stats.append((count_score, float(
            (entropy / math.log(bins * bins)).clamp(0, 1).item())))
    if len(stats) < batch_size:
        stats.extend([(0.0, 0.0)] * (batch_size - len(stats)))
    return torch.as_tensor(stats, device=device, dtype=dtype)


class FeatureResizer(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.norm = torch.nn.LayerNorm(output_dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.fc(x)))


class BenchmarkTextToBEV(torch.nn.Module):

    def __init__(self, text_dim: int, bev_channels: int, nhead: int,
                 dropout: float) -> None:
        super().__init__()
        self.value_proj = FeatureResizer(text_dim, bev_channels, dropout)
        self.key_proj = FeatureResizer(text_dim, bev_channels, dropout)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=bev_channels,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False)


def legacy_single_token_attention(module: BenchmarkTextToBEV,
                                  bev_query: torch.Tensor,
                                  text_feat: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = bev_query.shape
    query = bev_query.flatten(2).permute(2, 0, 1)
    key = module.key_proj(text_feat).permute(1, 0, 2)
    value = module.value_proj(text_feat).permute(1, 0, 2)
    aligned = module.attn(
        query=query, key=key, value=value, key_padding_mask=None,
        need_weights=False)[0]
    return aligned.permute(1, 2, 0).reshape(batch_size, channels, height, width)


def fast_single_token_attention(module: BenchmarkTextToBEV,
                                bev_query: torch.Tensor,
                                text_feat: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = bev_query.shape
    value = module.value_proj(text_feat[:, 0, :])
    embed_dim = module.attn.embed_dim
    v_weight = module.attn.in_proj_weight[2 * embed_dim:, :]
    if module.attn.in_proj_bias is not None:
        v_bias = module.attn.in_proj_bias[2 * embed_dim:]
    else:
        v_bias = None
    value = F.linear(value, v_weight, v_bias)
    value = module.attn.out_proj(value)
    return value[:, :, None, None].expand(batch_size, channels, height, width)


def build_projection_cache(point_cloud_range, height_samples, bev_h: int,
                           bev_w: int, device: torch.device,
                           dtype: torch.dtype) -> dict:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    xs = torch.linspace(x_min, x_max, bev_w + 1, device=device, dtype=dtype)[:-1]
    ys = torch.linspace(y_min, y_max, bev_h + 1, device=device, dtype=dtype)[:-1]
    xs = xs + (x_max - x_min) / bev_w * 0.5
    ys = ys + (y_max - y_min) / bev_h * 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    zs = torch.as_tensor(height_samples, device=device, dtype=dtype)
    zs = zs[(zs >= z_min) & (zs <= z_max)]
    ones = torch.ones_like(grid_x)
    bev_samples = []
    for z in zs:
        grid_z = torch.full_like(grid_x, z)
        bev_samples.append(torch.stack([grid_x, grid_y, grid_z], dim=-1))
    return dict(
        points_xyz=torch.stack(bev_samples, dim=0).reshape(-1, 3).contiguous(),
        points_ones=ones.reshape(-1, 1).repeat(zs.numel(), 1).contiguous(),
        points_hom=None,
        zs_num=zs.numel())


def legacy_image_projection(img_feat: torch.Tensor, point_cloud_range,
                            height_samples, target_feat: torch.Tensor,
                            batch_data_samples) -> torch.Tensor:
    device = target_feat.device
    dtype = torch.float32
    sample_img_feat = img_feat.float()
    _, _, bev_h, bev_w = target_feat.shape
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    xs = torch.linspace(x_min, x_max, bev_w + 1, device=device, dtype=dtype)[:-1]
    ys = torch.linspace(y_min, y_max, bev_h + 1, device=device, dtype=dtype)[:-1]
    xs = xs + (x_max - x_min) / bev_w * 0.5
    ys = ys + (y_max - y_min) / bev_h * 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    zs = torch.as_tensor(height_samples, device=device, dtype=dtype)
    zs = zs[(zs >= z_min) & (zs <= z_max)]
    ones = torch.ones_like(grid_x)
    bev_samples = []
    for z in zs:
        grid_z = torch.full_like(grid_x, z)
        bev_samples.append(torch.stack([grid_x, grid_y, grid_z, ones], dim=-1))
    points = torch.stack(bev_samples, dim=0).reshape(-1, 4)

    projected_feats = []
    valid_samples = []
    for batch_idx, data_sample in enumerate(batch_data_samples):
        lidar2img = torch.as_tensor(
            data_sample.metainfo['lidar2img'], device=device,
            dtype=dtype)[:3, :4]
        points_hom = points
        projected = points_hom @ lidar2img.t()
        depth = projected[:, 2].clamp(min=1e-5)
        pixel_x = projected[:, 0] / depth
        pixel_y = projected[:, 1] / depth
        img_shape_h, img_shape_w = data_sample.metainfo['img_shape']
        norm_x = pixel_x / img_shape_w * 2 - 1
        norm_y = pixel_y / img_shape_h * 2 - 1
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)
        sample_grid = sample_grid.view(1, zs.numel(), bev_h * bev_w, 2)
        sampled = F.grid_sample(
            sample_img_feat[batch_idx:batch_idx + 1],
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampled = sampled.view(1, sample_img_feat.shape[1], zs.numel(), bev_h,
                               bev_w)
        valid = ((pixel_x >= 0) & (pixel_x < img_shape_w) &
                 (pixel_y >= 0) & (pixel_y < img_shape_h) &
                 (projected[:, 2] > 1e-5))
        valid = valid.view(1, 1, zs.numel(), bev_h, bev_w).to(dtype=sampled.dtype)
        projected_feats.append((sampled * valid).sum(dim=2))
        valid_samples.append(valid.sum(dim=2).clamp_min(1.0))
    image_bev = torch.cat(projected_feats, dim=0)
    valid_count = torch.cat(valid_samples, dim=0)
    return image_bev / valid_count


def fast_image_projection(img_feat: torch.Tensor, point_cloud_range,
                          height_samples, target_feat: torch.Tensor,
                          batch_data_samples,
                          cache: dict | None = None) -> torch.Tensor:
    device = target_feat.device
    dtype = torch.float32
    sample_img_feat = img_feat.float()
    _, _, bev_h, bev_w = target_feat.shape
    cache = cache or build_projection_cache(point_cloud_range, height_samples,
                                            bev_h, bev_w, device, dtype)
    if cache.get('points_hom', None) is None:
        cache['points_hom'] = torch.cat(
            [cache['points_xyz'], cache['points_ones']], dim=1).contiguous()
    projected_feats = []
    valid_samples = []
    for batch_idx, data_sample in enumerate(batch_data_samples):
        lidar2img = torch.as_tensor(
            data_sample.metainfo['lidar2img'], device=device,
            dtype=dtype)[:3, :4]
        points_hom = cache['points_hom']
        projected = points_hom @ lidar2img.t()
        depth = projected[:, 2].clamp(min=1e-5)
        pixel_x = projected[:, 0] / depth
        pixel_y = projected[:, 1] / depth
        img_shape_h, img_shape_w = data_sample.metainfo['img_shape']
        norm_x = pixel_x / img_shape_w * 2 - 1
        norm_y = pixel_y / img_shape_h * 2 - 1
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).view(
            1, cache['zs_num'], bev_h * bev_w, 2)
        sampled = F.grid_sample(
            sample_img_feat[batch_idx:batch_idx + 1],
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampled = sampled.view(1, sample_img_feat.shape[1], cache['zs_num'],
                               bev_h, bev_w)
        valid = ((pixel_x >= 0) & (pixel_x < img_shape_w) &
                 (pixel_y >= 0) & (pixel_y < img_shape_h) &
                 (projected[:, 2] > 1e-5))
        valid = valid.view(1, 1, cache['zs_num'], bev_h, bev_w).to(
            dtype=sampled.dtype)
        projected_feats.append((sampled * valid).sum(dim=2))
        valid_samples.append(valid.sum(dim=2).clamp_min(1.0))
    image_bev = torch.cat(projected_feats, dim=0)
    valid_count = torch.cat(valid_samples, dim=0)
    return image_bev / valid_count


def benchmark_model_fastpaths(input_file: Path,
                              repo_root: Path,
                              text_hash_dim: int) -> dict:
    text_hash = load_text_hash_module(repo_root)
    records = json.loads(input_file.read_text(encoding='utf-8'))
    texts = [record.get('text', '') for record in records[:128]]
    samples = build_synthetic_samples(64, text_hash_dim)
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    point_batch = [
        torch.randn(1200 + (idx % 4) * 600, 4) for idx in range(8)
    ]

    text_hash_legacy = benchmark_callable(
        lambda: legacy_build_hashed_text_tensor(
            text_hash, texts, text_hash_dim, device, dtype), device)
    text_hash_new = benchmark_callable(
        lambda: text_hash.build_hashed_text_tensor(
            texts, text_hash_dim, device=device, dtype=dtype), device)
    sample_text_legacy = benchmark_callable(
        lambda: legacy_build_sample_text_features(
            text_hash, samples, text_hash_dim, device, dtype), device)
    sample_text_new = benchmark_callable(
        lambda: text_hash.build_sample_text_features(
            samples, text_hash_dim, device=device, dtype=dtype), device)
    point_stats_legacy = benchmark_callable(
        lambda: legacy_point_stats(point_batch, 8, device, dtype,
                                   point_cloud_range), device)
    point_stats_new = benchmark_callable(
        lambda: fast_point_stats(point_batch, 8, device, dtype,
                                 point_cloud_range), device)

    attn = BenchmarkTextToBEV(
        text_dim=text_hash_dim, bev_channels=384, nhead=8, dropout=0.0).to(device)
    bev_query = torch.randn(8, 384, 96, 88, device=device)
    text_token = torch.randn(8, 1, text_hash_dim, device=device)
    attn_legacy = benchmark_callable(
        lambda: legacy_single_token_attention(attn, bev_query, text_token),
        device,
        repeat=25,
        warmup=3)
    attn_new = benchmark_callable(
        lambda: fast_single_token_attention(attn, bev_query, text_token),
        device,
        repeat=25,
        warmup=3)
    attn_diff = float(
        (legacy_single_token_attention(attn, bev_query, text_token) -
         fast_single_token_attention(attn, bev_query, text_token)).abs().max().
        item())

    img_feat = torch.randn(4, 256, 32, 88, device=device)
    target_feat = torch.randn(4, 384, 96, 88, device=device)
    proj_samples = build_synthetic_samples(4, text_hash_dim)
    point_cloud_range_proj = [0, -25.6, -3, 51.2, 25.6, 2]
    height_samples = (-1.0, 0.0, 1.0)
    projection_cache = build_projection_cache(point_cloud_range_proj,
                                              height_samples, 96, 88, device,
                                              dtype)
    proj_legacy = benchmark_callable(
        lambda: legacy_image_projection(img_feat, point_cloud_range_proj,
                                        height_samples, target_feat,
                                        proj_samples),
        device,
        repeat=12,
        warmup=2)
    proj_new = benchmark_callable(
        lambda: fast_image_projection(img_feat, point_cloud_range_proj,
                                      height_samples, target_feat, proj_samples,
                                      projection_cache),
        device,
        repeat=12,
        warmup=2)
    proj_diff = float(
        (legacy_image_projection(img_feat, point_cloud_range_proj, height_samples,
                                 target_feat, proj_samples) -
         fast_image_projection(img_feat, point_cloud_range_proj, height_samples,
                               target_feat, proj_samples,
                               projection_cache)).abs().max().item())

    return {
        'device': device.type,
        'text_hash_tensor': {
            'legacy_ms': text_hash_legacy * 1000.0,
            'new_ms': text_hash_new * 1000.0,
            'speedup': text_hash_legacy / max(text_hash_new, 1e-12),
        },
        'sample_text_features': {
            'legacy_ms': sample_text_legacy * 1000.0,
            'new_ms': sample_text_new * 1000.0,
            'speedup': sample_text_legacy / max(sample_text_new, 1e-12),
        },
        'point_stats': {
            'legacy_ms': point_stats_legacy * 1000.0,
            'new_ms': point_stats_new * 1000.0,
            'speedup': point_stats_legacy / max(point_stats_new, 1e-12),
        },
        'single_token_attention': {
            'legacy_ms': attn_legacy * 1000.0,
            'new_ms': attn_new * 1000.0,
            'speedup': attn_legacy / max(attn_new, 1e-12),
            'max_abs_diff': attn_diff,
        },
        'image_to_bev_projection': {
            'legacy_ms': proj_legacy * 1000.0,
            'new_ms': proj_new * 1000.0,
            'speedup': proj_legacy / max(proj_new, 1e-12),
            'max_abs_diff': proj_diff,
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter_dir = output_dir / 'converter'
    text_dir = output_dir / 'texts'
    prompt_path = output_dir / 'radar_llm_prompts_full.jsonl'

    results = {
        'data_root': str(data_root),
        'workers': args.workers,
        'converter': benchmark_radar_converter(
            data_root, converter_dir, args.workers, not args.skip_lidar_check),
        'text_generation': benchmark_radar_text(data_root, text_dir,
                                                args.workers),
    }
    results['prompt_generation'] = benchmark_llm_prompts(
        text_dir / 'radar_texts_full.json', prompt_path, args.workers)
    results['text_hash_micro_benchmark'] = benchmark_text_hash(
        text_dir / 'radar_texts_full.json', repo_root, args.text_hash_dim,
        args.repeat)
    results['model_fastpaths'] = benchmark_model_fastpaths(
        text_dir / 'radar_texts_full.json', repo_root, args.text_hash_dim)

    result_path = output_dir / 'benchmark_results.json'
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'Saved benchmark results to {result_path}')


if __name__ == '__main__':
    main()
