from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent


def _status(prefix: str, message: str) -> None:
    print(f'[{prefix}] {message}')


def _unwrap_dataset_cfg(dataset_cfg):
    current = dataset_cfg
    while current is not None and hasattr(current, 'get') and current.get(
            'dataset') is not None:
        current = current.get('dataset')
    return current


def _iter_evaluators(evaluator_cfg) -> Iterable:
    if evaluator_cfg is None:
        return []
    if isinstance(evaluator_cfg, (list, tuple)):
        return evaluator_cfg
    return [evaluator_cfg]


def _check_path(path_str: str, label: str, failures: list[str]) -> None:
    path = Path(path_str)
    if path.exists():
        _status('PASS', f'{label}: {path}')
    else:
        failures.append(f'{label} missing: {path}')
        _status('FAIL', f'{label}: {path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Windows-local preflight checks for 3ddetection.')
    parser.add_argument(
        '--config',
        default='configs/pointpillars/pointpillars_radar_vod.py',
        help='Config file to validate.')
    parser.add_argument(
        '--dataset-root',
        default=None,
        help='Optional VOD root override. Sets MMD3D_VOD_ROOT for this run.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional checkpoint path to validate for test mode.')
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        default='train',
        help='Validation mode for messaging only.')
    parser.add_argument(
        '--allow-cpu',
        action='store_true',
        help='Do not fail if torch CUDA is unavailable.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    if args.dataset_root:
        os.environ['MMD3D_VOD_ROOT'] = args.dataset_root

    _status('INFO', f'Python executable: {sys.executable}')
    _status('INFO', f'Python version: {sys.version.split()[0]}')

    try:
        import mmcv
        import mmdet
        import mmdet3d
        import mmengine
        import numba
        import torch
        from mmengine.config import Config

        from local_paths import apply_placeholder_mapping, resolve_vod_base
        from mmdet3d.evaluation.functional.kitti_utils.rotate_iou import \
            probe_rotate_iou_backends
    except Exception as exc:
        _status('FAIL', f'Core imports failed: {exc}')
        return 1

    _status(
        'PASS',
        'OpenMMLab stack: '
        f'mmengine={mmengine.__version__}, mmcv={mmcv.__version__}, '
        f'mmdet={mmdet.__version__}, mmdet3d={mmdet3d.__version__}')
    _status('PASS', f'numba={numba.__version__}')

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        _status(
            'PASS',
            f'torch={torch.__version__}, CUDA={torch.version.cuda}, '
            f'device={device_name}')
    else:
        message = (
            f'torch={torch.__version__}, CUDA build={torch.version.cuda}, '
            'torch.cuda.is_available()=False')
        if args.allow_cpu:
            _status('WARN', message)
        else:
            failures.append(message)
            _status('FAIL', message)

    rotate_backend = probe_rotate_iou_backends()
    if rotate_backend['selected_backend'] in ('mmcv', 'python'):
        _status(
            'PASS',
            'KITTI/VOD rotate IoU backend: '
            f"{rotate_backend['selected_backend']} "
            f"({rotate_backend['reason']})")
    else:
        failures.append('No safe rotate IoU backend was selected.')
        _status('FAIL', f'Rotate IoU backend info: {rotate_backend}')

    if rotate_backend['numba_nvvm_available']:
        _status('PASS', 'numba CUDA NVVM is available.')
    else:
        _status(
            'WARN',
            'numba CUDA NVVM unavailable: '
            f"{rotate_backend['numba_error'] or 'unknown error'}")

    try:
        dataset_root = resolve_vod_base(args.dataset_root)
        _status('PASS', f'Resolved dataset root: {dataset_root}')
    except Exception as exc:
        failures.append(f'Failed to resolve dataset root: {exc}')
        _status('FAIL', f'Resolved dataset root failed: {exc}')
        dataset_root = None

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        _status('FAIL', f'Config missing: {config_path}')
        return 1

    _status('PASS', f'Config file: {config_path}')
    cfg = Config.fromfile(str(config_path))
    apply_placeholder_mapping(cfg, args.dataset_root)

    for loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        loader_cfg = cfg.get(loader_name)
        if loader_cfg is None:
            _status('INFO', f'{loader_name}: not configured')
            continue
        dataset_cfg = _unwrap_dataset_cfg(loader_cfg.get('dataset'))
        if dataset_cfg is None:
            _status('WARN', f'{loader_name}: dataset config missing')
            continue
        ann_file = dataset_cfg.get('ann_file')
        if ann_file:
            _check_path(ann_file, f'{loader_name}.ann_file', failures)

    uses_rotate_iou = False
    for evaluator_name in ('val_evaluator', 'test_evaluator'):
        evaluator_cfg = cfg.get(evaluator_name)
        if evaluator_cfg is None:
            _status('INFO', f'{evaluator_name}: not configured')
            continue
        for idx, evaluator in enumerate(_iter_evaluators(evaluator_cfg)):
            prefix = evaluator_name if idx == 0 else f'{evaluator_name}[{idx}]'
            evaluator_type = evaluator.get('type', '<unknown>')
            _status('PASS', f'{prefix}.type: {evaluator_type}')
            ann_file = evaluator.get('ann_file')
            if ann_file:
                _check_path(ann_file, f'{prefix}.ann_file', failures)
            if evaluator_type == 'KittiMetric' and evaluator.get(
                    'metric', 'bbox') != 'img_bbox':
                uses_rotate_iou = True

    if uses_rotate_iou:
        _status(
            'PASS',
            'Evaluator path will use KITTI/VOD rotated IoU, and the selected '
            f"backend is {rotate_backend['selected_backend']}.")
    else:
        _status('INFO', 'Evaluator path does not require rotated IoU.')

    train_cfg = cfg.get('train_cfg')
    if train_cfg is not None:
        val_interval = train_cfg.get('val_interval', None)
        _status('INFO', f"train_cfg.val_interval: {val_interval}")

    if args.checkpoint:
        _check_path(args.checkpoint, 'checkpoint', failures)

    if failures:
        _status('FAIL', 'Preflight failed:')
        for failure in failures:
            _status('FAIL', f'  - {failure}')
        return 1

    _status('PASS', f'Preflight completed for {args.mode} mode.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
