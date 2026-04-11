from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_CLASSES = [
    'bicycle', 'bicycle_rack', 'Car', 'Cyclist', 'human_depiction',
    'moped_scooter', 'motor', 'Pedestrian', 'ride_other', 'ride_uncertain',
    'rider', 'truck', 'vehicle_other'
]

DEFAULT_KEEP_CLASSES = [
    'bicycle', 'Car', 'Cyclist', 'motor', 'Pedestrian', 'ride_other', 'truck'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert the KITTI-style radar dataset to MMDet3D infos.')
    parser.add_argument(
        '--data-root',
        default='dataset/radar',
        help='Root path of the radar dataset.')
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
        default=['train', 'val'],
        help='ImageSets splits to convert.')
    return parser.parse_args()


def extend_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float32)
    extended = np.eye(4, dtype=np.float32)
    extended[:mat.shape[0], :mat.shape[1]] = mat
    return extended


def read_split_ids(split_file: Path) -> List[str]:
    with split_file.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


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
            values = value.strip().split()
            if not values:
                continue
            calib[key] = np.array([float(v) for v in values], dtype=np.float32)

    for key in ['P0', 'P1', 'P2', 'P3']:
        calib[key] = extend_matrix(calib[key].reshape(3, 4))
    calib['R0_rect'] = extend_matrix(calib['R0_rect'].reshape(3, 3))
    calib['Tr_velo_to_cam'] = extend_matrix(
        calib['Tr_velo_to_cam'].reshape(3, 4))
    calib['Tr_imu_to_velo'] = extend_matrix(
        calib.get('Tr_imu_to_velo', np.eye(3, 4, dtype=np.float32)).reshape(
            3, 4))
    return calib


def parse_label_file(label_path: Path, categories: Dict[str, int]) -> List[Dict]:
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
            truncated=float(fields[1]),
            occluded=int(float(fields[2])),
            alpha=float(fields[3]),
            score=float(fields[15]) if len(fields) > 15 else 0.0,
            index=group_id,
            group_id=group_id,
            difficulty=int(float(fields[2])),
            num_lidar_pts=-1)
        instances.append(instance)
    return instances


def build_data_info(data_root: Path, sample_id: str, data_index: int,
                    categories: Dict[str, int]) -> Dict:
    image_path = data_root / 'training' / 'image_2' / f'{sample_id}.jpg'
    lidar_path = data_root / 'training' / 'velodyne' / f'{sample_id}.bin'
    calib_path = data_root / 'training' / 'calib' / f'{sample_id}.txt'
    label_path = data_root / 'training' / 'label_2' / f'{sample_id}.txt'

    width, height = read_jpeg_size(image_path)
    calib = read_calib(calib_path)
    lidar2cam = calib['R0_rect'] @ calib['Tr_velo_to_cam']

    return dict(
        sample_idx=data_index,
        token=sample_id,
        images=dict(
            CAM2=dict(
                img_path=image_path.name,
                height=height,
                width=width,
                cam2img=calib['P2'].tolist(),
                lidar2cam=lidar2cam.tolist(),
                lidar2img=(calib['P2'] @ lidar2cam).tolist())),
        lidar_points=dict(
            num_pts_feats=7,
            lidar_path=lidar_path.name,
            Tr_velo_to_cam=calib['Tr_velo_to_cam'].tolist(),
            Tr_imu_to_velo=calib['Tr_imu_to_velo'].tolist()),
        instances=parse_label_file(label_path, categories))


def convert_split(data_root: Path, split: str,
                  categories: Dict[str, int]) -> List[Dict]:
    split_file = data_root / 'ImageSets' / f'{split}.txt'
    sample_ids = read_split_ids(split_file)
    data_list = []
    total = len(sample_ids)
    for index, sample_id in enumerate(sample_ids, 1):
        if index == 1 or index == total or index % 500 == 0:
            print(f'[{split}] converting {index}/{total}')
        data_list.append(
            build_data_info(data_root, sample_id, index - 1, categories))
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


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root
    categories = {name: DEFAULT_CLASSES.index(name) for name in args.class_names}

    split_infos = {}
    for split in args.splits:
        split_file = data_root / 'ImageSets' / f'{split}.txt'
        if not split_file.exists():
            print(f'Skip split "{split}" because {split_file} does not exist.')
            continue
        data_list = convert_split(data_root, split, categories)
        split_infos[split] = data_list
        dump_infos(data_list, out_dir / f'{args.pkl_prefix}_infos_{split}.pkl',
                   categories)

    if 'train' in split_infos and 'val' in split_infos:
        dump_infos(split_infos['train'] + split_infos['val'],
                   out_dir / f'{args.pkl_prefix}_infos_trainval.pkl',
                   categories)


if __name__ == '__main__':
    main()
