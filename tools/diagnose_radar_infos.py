from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CLASSES = ('Car', 'Pedestrian', 'Cyclist')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check MMDet3D radar info pkl integrity.')
    parser.add_argument('--data-root', default='dataset/radar')
    parser.add_argument('--train-pkl', default='dataset/radar/radar_infos_train.pkl')
    parser.add_argument('--val-pkl', default='dataset/radar/radar_infos_val.pkl')
    parser.add_argument('--test-pkl', default='')
    parser.add_argument('--classes', nargs='+', default=list(DEFAULT_CLASSES))
    parser.add_argument('--sample-count', type=int, default=5)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def normalize_infos(obj: Any) -> tuple[dict, list[dict]]:
    if isinstance(obj, dict):
        if 'data_list' not in obj:
            raise ValueError('dict pkl is missing data_list')
        return obj.get('metainfo', {}), obj['data_list']
    if isinstance(obj, list):
        return {}, obj
    raise TypeError(f'Unsupported pkl top-level type: {type(obj)}')


def resolve_under(root: Path, subdir: str, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / 'training' / subdir / path


def get_categories(metainfo: dict) -> dict:
    categories = metainfo.get('categories', {})
    return {str(k): int(v) for k, v in categories.items()}


def check_one(path: Path, data_root: Path, classes: set[str],
              sample_count: int) -> dict:
    print(f'\n=== {path} ===')
    if not path.exists():
        print(f'MISSING_PKL {path}')
        return {'path': path, 'exists': False, 'data_list': [], 'metainfo': {}}

    metainfo, data_list = normalize_infos(load_pickle(path))
    categories = get_categories(metainfo)
    valid_category_ids = set(categories.values())

    print('metainfo:', metainfo)
    print('num_samples:', len(data_list))
    if not data_list:
        return {'path': path, 'exists': True, 'data_list': [], 'metainfo': metainfo}

    first_keys = set(data_list[0].keys())
    print('first_sample_keys:', sorted(first_keys))
    rng = random.Random(2026)
    for idx in rng.sample(range(len(data_list)), min(sample_count, len(data_list))):
        info = data_list[idx]
        print('random_sample:', idx, 'sample_idx=', info.get('sample_idx'),
              'token=', info.get('token'), 'num_instances=',
              len(info.get('instances', [])))

    issues = {
        'top_key_mismatch': [],
        'instance_key_mismatch': [],
        'missing_required': [],
        'missing_paths': [],
        'bad_calib': [],
        'bad_boxes': [],
        'bad_labels': [],
        'empty_gt': [],
        'bad_sample_idx': [],
        'duplicate_token': [],
        'class_name_not_in_categories': [],
    }

    for cls in classes:
        if cls not in set(categories):
            issues['class_name_not_in_categories'].append(cls)

    first_inst_keys = None
    seen_tokens = {}
    for idx, info in enumerate(data_list):
        keys = set(info.keys())
        if keys != first_keys:
            issues['top_key_mismatch'].append(
                (idx, sorted(keys - first_keys), sorted(first_keys - keys)))

        sample_idx = info.get('sample_idx')
        if sample_idx != idx:
            issues['bad_sample_idx'].append((idx, sample_idx, info.get('token')))

        token = info.get('token')
        if token is not None:
            token = str(token)
            if token in seen_tokens:
                issues['duplicate_token'].append((idx, seen_tokens[token], token))
            seen_tokens[token] = idx

        for key in ('sample_idx', 'images', 'lidar_points', 'instances'):
            if key not in info or info[key] is None:
                issues['missing_required'].append((idx, key))

        lidar_path = (info.get('lidar_points') or {}).get('lidar_path')
        pts_path = resolve_under(data_root, 'velodyne', lidar_path)
        if pts_path is None or not pts_path.exists():
            issues['missing_paths'].append((idx, 'lidar_path', str(pts_path)))

        cam2 = (info.get('images') or {}).get('CAM2') or {}
        img_path = resolve_under(data_root, 'image_2', cam2.get('img_path'))
        if img_path is None or not img_path.exists():
            issues['missing_paths'].append((idx, 'img_path', str(img_path)))

        for calib_key in ('cam2img', 'lidar2cam', 'lidar2img'):
            mat = np.asarray(cam2.get(calib_key, []), dtype=np.float64)
            if mat.shape != (4, 4) or not np.isfinite(mat).all():
                issues['bad_calib'].append((idx, calib_key, mat.shape))

        instances = info.get('instances') or []
        if not instances:
            issues['empty_gt'].append((idx, sample_idx, info.get('token')))

        for inst_idx, inst in enumerate(instances):
            inst_keys = set(inst.keys())
            if first_inst_keys is None:
                first_inst_keys = inst_keys
            elif inst_keys != first_inst_keys:
                issues['instance_key_mismatch'].append(
                    (idx, inst_idx, sorted(inst_keys - first_inst_keys),
                     sorted(first_inst_keys - inst_keys)))

            label = inst.get('bbox_label_3d')
            if label not in valid_category_ids:
                issues['bad_labels'].append((idx, inst_idx, label))

            box = np.asarray(inst.get('bbox_3d', []), dtype=np.float64)
            if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                issues['bad_boxes'].append((idx, inst_idx, box.tolist()))

            bbox = np.asarray(inst.get('bbox', []), dtype=np.float64)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                issues['bad_boxes'].append((idx, inst_idx, 'bad_2d_bbox',
                                            bbox.tolist()))

    for name, values in issues.items():
        print(f'{name}: {len(values)}')
        if values:
            print('  first10:', values[:10])

    return {
        'path': path,
        'exists': True,
        'metainfo': metainfo,
        'data_list': data_list,
        'first_keys': first_keys,
        'first_inst_keys': first_inst_keys,
    }


def compare_infos(left: dict, right: dict) -> None:
    if not left.get('exists') or not right.get('exists'):
        return
    if not left['data_list'] or not right['data_list']:
        return
    print(f'\n=== compare {left["path"].name} vs {right["path"].name} ===')
    print('metainfo_equal:', left['metainfo'] == right['metainfo'])
    print('top_keys_equal:', left.get('first_keys') == right.get('first_keys'))
    print('instance_keys_equal:',
          left.get('first_inst_keys') == right.get('first_inst_keys'))
    left_tokens = {str(x.get('token')) for x in left['data_list'] if 'token' in x}
    right_tokens = {str(x.get('token')) for x in right['data_list'] if 'token' in x}
    if left_tokens and right_tokens:
        print('token_overlap:', len(left_tokens & right_tokens))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    classes = set(args.classes)
    train = check_one(Path(args.train_pkl), data_root, classes, args.sample_count)
    val = check_one(Path(args.val_pkl), data_root, classes, args.sample_count)
    compare_infos(train, val)
    if args.test_pkl:
        test = check_one(Path(args.test_pkl), data_root, classes, args.sample_count)
        compare_infos(val, test)


if __name__ == '__main__':
    main()
