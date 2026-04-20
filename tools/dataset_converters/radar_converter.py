from __future__ import annotations

import argparse
import os
import pickle
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


DEFAULT_CLASSES = [
    'bicycle', 'bicycle_rack', 'Car', 'Cyclist', 'human_depiction',
    'moped_scooter', 'motor', 'Pedestrian', 'ride_other', 'ride_uncertain',
    'rider', 'truck', 'vehicle_other'
]

DEFAULT_KEEP_CLASSES = ['Pedestrian', 'Cyclist', 'Car']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert the VOD radar dataset to MMDet3D infos.')
    parser.add_argument(
        '--data-root',
        default=os.getenv('MMD3D_VOD_ROOT', 'D:/VOD_ascii'),
        help='Path to the VOD root, view_of_delft_PUBLIC root, or a sensor '
        'modality directory.')
    parser.add_argument(
        '--modality',
        default=os.getenv('MMD3D_VOD_MODALITY', 'radar'),
        help='Sensor modality to convert. Examples: radar, radar_5frames.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory. Defaults to <data-root>.')
    parser.add_argument(
        '--pkl-prefix',
        default='radar',
        help='Prefix of generated info pkl files.')
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=DEFAULT_KEEP_CLASSES,
        help='Classes to keep. Other labels are ignored.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='ImageSets splits to convert.')
    parser.add_argument(
        '--skip-lidar-check',
        action='store_true',
        help='Skip checking that every .bin file is divisible by 7 floats.')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker threads used for per-sample metadata parsing.')
    return parser.parse_args()


def extend_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float32)
    extended = np.eye(4, dtype=np.float32)
    extended[:mat.shape[0], :mat.shape[1]] = mat
    return extended


def read_split_ids(split_file: Path) -> List[str]:
    return [
        line for line in split_file.read_text(encoding='utf-8').splitlines()
        if line
    ]


def read_jpeg_size(image_path: Path) -> tuple[int, int]:
    with image_path.open('rb') as f:
        f.seek(2)
        while True:
            marker = f.read(1)
            while marker == b'\xff':
                marker = f.read(1)
            if marker in [b'\xc0', b'\xc1', b'\xc2', b'\xc3']:
                f.read(3)
                height, width = struct.unpack('>HH', f.read(4))
                return width, height
            block_size = struct.unpack('>H', f.read(2))[0]
            f.seek(block_size - 2, 1)


def read_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    calib = {}
    with calib_path.open('r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            values = np.fromstring(value, sep=' ', dtype=np.float32)
            if values.size == 0:
                continue
            calib[key] = values

    for key in ['P0', 'P1', 'P2', 'P3']:
        calib[key] = extend_matrix(calib[key].reshape(3, 4))
    calib['R0_rect'] = extend_matrix(calib['R0_rect'].reshape(3, 3))
    calib['Tr_velo_to_cam'] = extend_matrix(
        calib['Tr_velo_to_cam'].reshape(3, 4))
    tr_imu_to_velo = calib.get('Tr_imu_to_velo')
    if tr_imu_to_velo is None or tr_imu_to_velo.size != 12:
        tr_imu_to_velo = np.eye(3, 4, dtype=np.float32)
    calib['Tr_imu_to_velo'] = extend_matrix(
        tr_imu_to_velo.reshape(3, 4))
    return calib


def parse_track_id(raw_value: str):
    try:
        value = float(raw_value)
    except ValueError:
        return raw_value
    if value.is_integer():
        return int(value)
    return raw_value


def validate_lidar_7d(lidar_path: Path) -> None:
    file_size = lidar_path.stat().st_size
    float_size = np.dtype(np.float32).itemsize
    num_float32 = file_size // float_size
    if file_size % float_size != 0:
        raise ValueError(f'{lidar_path} size is not aligned to float32.')
    if num_float32 % 7 != 0:
        raise ValueError(
            f'{lidar_path} has {num_float32} float32 values, which cannot be '
            'reshaped to VOD radar points with shape (-1, 7).')


def resolve_dataset_roots(data_root: Path, modality: str) -> tuple[Path, Path]:
    modality_names = ('lidar', 'radar', 'radar_3frames', 'radar_5frames')
    if data_root.name in modality_names:
        base_root = data_root.parent
    elif data_root.name == 'view_of_delft_PUBLIC':
        base_root = data_root
    elif (data_root / 'view_of_delft_PUBLIC').exists():
        base_root = data_root / 'view_of_delft_PUBLIC'
    elif any((data_root / name).exists() for name in modality_names):
        base_root = data_root
    else:
        raise FileNotFoundError(
            f'Unable to resolve a VOD dataset root from {data_root}.')

    modality_root = base_root / modality
    if not modality_root.exists():
        raise FileNotFoundError(
            f'Unable to locate modality "{modality}" under {base_root}.')
    return base_root, modality_root


def resolve_first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        'None of the candidate paths exist:\n' +
        '\n'.join(str(candidate) for candidate in candidates))


def resolve_optional_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_split_file(base_root: Path, modality_root: Path,
                       split: str) -> Path:
    return resolve_first_existing(
        modality_root / 'ImageSets' / f'{split}.txt',
        base_root / 'lidar' / 'ImageSets' / f'{split}.txt',
        base_root / 'ImageSets' / f'{split}.txt')


def resolve_image_path(base_root: Path, sample_folder: str,
                       sample_id: str) -> Path:
    candidates = []
    for folder in (sample_folder, 'training'):
        for suffix in ('.jpg', '.png'):
            candidates.append(base_root / 'lidar' / folder / 'image_2' /
                              f'{sample_id}{suffix}')
    return resolve_first_existing(*candidates)


def build_sample_folder_index(data_root: Path) -> Dict[str, str]:
    folder_index = {}
    for folder in ('training', 'testing'):
        velodyne_dir = data_root / folder / 'velodyne'
        if not velodyne_dir.exists():
            continue
        for lidar_path in velodyne_dir.glob('*.bin'):
            folder_index.setdefault(lidar_path.stem, folder)
    return folder_index


def parse_label_file(label_path: Path, categories: Dict[str, int]) -> List[Dict]:
    """Parse VOD labels without treating column 2 as truncated.

    In newer VOD annotations the second column is a track id. It must not be
    stored as ``truncated`` and must not be used for difficulty.
    """
    instances = []
    if not label_path.exists():
        return instances

    with label_path.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for group_id, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 15:
            continue
        name = fields[0]
        if name not in categories:
            continue

        h, w, length = [float(v) for v in fields[8:11]]
        loc = [float(v) for v in fields[11:14]]
        instance = dict(
            bbox=[float(v) for v in fields[4:8]],
            bbox_label=categories[name],
            bbox_3d=loc + [length, h, w, float(fields[14])],
            bbox_label_3d=categories[name],
            track_id=parse_track_id(fields[1]),
            occluded=int(float(fields[2])),
            alpha=float(fields[3]),
            score=float(fields[15]) if len(fields) > 15 else 0.0,
            index=group_id,
            group_id=group_id,
            difficulty=0,
            num_lidar_pts=-1)
        instances.append(instance)
    return instances


def build_data_info(base_root: Path, modality_root: Path, sample_id: str,
                    data_index: int,
                    categories: Dict[str, int],
                    split: str = 'train',
                    check_lidar: bool = True,
                    sample_folder: Optional[str] = None) -> Dict:
    if sample_folder is None:
        sample_folder = 'testing' if split == 'test' else 'training'

    image_path = resolve_image_path(base_root, sample_folder, sample_id)
    lidar_path = resolve_first_existing(
        modality_root / sample_folder / 'velodyne' / f'{sample_id}.bin',
        modality_root / 'training' / 'velodyne' / f'{sample_id}.bin',
        modality_root / 'testing' / 'velodyne' / f'{sample_id}.bin')
    calib_path = resolve_first_existing(
        modality_root / sample_folder / 'calib' / f'{sample_id}.txt',
        modality_root / 'training' / 'calib' / f'{sample_id}.txt',
        modality_root / 'testing' / 'calib' / f'{sample_id}.txt',
        base_root / 'radar' / sample_folder / 'calib' / f'{sample_id}.txt',
        base_root / 'radar' / 'training' / 'calib' / f'{sample_id}.txt',
        base_root / 'lidar' / sample_folder / 'calib' / f'{sample_id}.txt',
        base_root / 'lidar' / 'training' / 'calib' / f'{sample_id}.txt')
    label_path = resolve_optional_existing(
        base_root / 'lidar' / sample_folder / 'label_2' / f'{sample_id}.txt',
        base_root / 'lidar' / 'training' / 'label_2' / f'{sample_id}.txt')

    image_rel_path = image_path.relative_to(base_root)
    lidar_rel_path = lidar_path.relative_to(base_root)

    if check_lidar:
        validate_lidar_7d(lidar_path)
    width, height = read_jpeg_size(image_path)
    calib = read_calib(calib_path)
    lidar2cam = calib['R0_rect'] @ calib['Tr_velo_to_cam']

    return dict(
        sample_idx=data_index,
        token=sample_id,
        images=dict(
            CAM2=dict(
                img_path=image_rel_path.as_posix(),
                height=height,
                width=width,
                cam2img=calib['P2'].tolist(),
                lidar2cam=lidar2cam.tolist(),
                lidar2img=(calib['P2'] @ lidar2cam).tolist())),
        lidar_points=dict(
            num_pts_feats=7,
            lidar_path=lidar_rel_path.as_posix(),
            Tr_velo_to_cam=calib['Tr_velo_to_cam'].tolist(),
            Tr_imu_to_velo=calib['Tr_imu_to_velo'].tolist()),
        point_cloud=dict(
            num_features=7, velodyne_path=lidar_rel_path.as_posix()),
        instances=parse_label_file(label_path, categories))


def convert_split(base_root: Path, modality_root: Path, split: str,
                  categories: Dict[str, int],
                  check_lidar: bool = True,
                  workers: int = 1,
                  sample_folder_index: Optional[Dict[str, str]] = None
                  ) -> List[Dict]:
    split_file = resolve_split_file(base_root, modality_root, split)
    sample_ids = read_split_ids(split_file)
    data_list = []
    total = len(sample_ids)

    worker_count = max(1, workers)
    default_folder = 'testing' if split == 'test' else 'training'
    folder_index = sample_folder_index or {}
    build_one = partial(
        build_data_info,
        base_root,
        modality_root,
        categories=categories,
        split=split,
        check_lidar=check_lidar)

    indexed_ids = list(enumerate(sample_ids, 1))
    if worker_count == 1:
        for index, sample_id in indexed_ids:
            if index == 1 or index == total or index % 500 == 0:
                print(f'[{split}] converting {index}/{total}')
            data_list.append(
                build_one(
                    sample_id=sample_id,
                    data_index=index - 1,
                    sample_folder=folder_index.get(sample_id, default_folder)))
        return data_list

    def iter_jobs() -> Iterable[Dict]:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            yield from executor.map(
                lambda item: build_one(
                    sample_id=item[1],
                    data_index=item[0] - 1,
                    sample_folder=folder_index.get(item[1], default_folder)),
                indexed_ids)

    for index, data_info in enumerate(iter_jobs(), 1):
        if index == 1 or index == total or index % 500 == 0:
            print(f'[{split}] converting {index}/{total}')
        data_list.append(data_info)
    return data_list


def dump_infos(data_list: List[Dict], out_path: Path,
               categories: Dict[str, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    infos = dict(
        metainfo=dict(
            dataset='radar',
            info_version='1.1',
            categories=categories),
        data_list=data_list)
    with out_path.open('wb') as f:
        pickle.dump(infos, f)
    print(f'Saved {len(data_list)} samples to {out_path}')


def create_vod_infos(data_root: str,
                     out_dir: Optional[str] = None,
                     pkl_prefix: str = 'radar',
                     class_names: Optional[List[str]] = None,
                     splits: Optional[List[str]] = None,
                     check_lidar: bool = True,
                     workers: int = 1,
                     modality: str = 'radar') -> Dict[str, List[Dict]]:
    data_root = Path(data_root)
    base_root, modality_root = resolve_dataset_roots(data_root, modality)
    out_dir = Path(out_dir) if out_dir else base_root / 'infos' / modality
    class_names = class_names if class_names is not None else DEFAULT_KEEP_CLASSES
    splits = splits if splits is not None else ['train', 'val', 'test']
    categories = {name: label for label, name in enumerate(class_names)}
    sample_folder_index = build_sample_folder_index(modality_root)

    split_infos = {}
    for split in splits:
        try:
            resolve_split_file(base_root, modality_root, split)
        except FileNotFoundError:
            print(f'Skip split "{split}" because no split file was found.')
            continue
        data_list = convert_split(
            base_root,
            modality_root,
            split,
            categories,
            check_lidar,
            workers=workers,
            sample_folder_index=sample_folder_index)
        split_infos[split] = data_list
        dump_infos(data_list, out_dir / f'{pkl_prefix}_infos_{split}.pkl',
                   categories)

    if 'train' in split_infos and 'val' in split_infos:
        dump_infos(split_infos['train'] + split_infos['val'],
                   out_dir / f'{pkl_prefix}_infos_trainval.pkl',
                   categories)
    return split_infos


def main() -> None:
    args = parse_args()
    create_vod_infos(
        data_root=args.data_root,
        out_dir=args.out_dir,
        pkl_prefix=args.pkl_prefix,
        class_names=args.class_names,
        splits=args.splits,
        check_lidar=not args.skip_lidar_check,
        workers=args.workers,
        modality=args.modality)


if __name__ == '__main__':
    main()
