# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_wrappers import CBGSDataset
from .det3d_dataset import Det3DDataset
from .kitti_dataset import KittiDataset
from .lyft_dataset import LyftDataset
from .nuscenes_dataset import NuScenesDataset
from .radar_dataset import RadarDataset
from .radar_text_dataset import RadarTextDataset
# 室内分割相关基类在当前瘦身工程中可能被移除，按存在情况导入。
try:
    from .s3dis_dataset import S3DISDataset, S3DISSegDataset
except ModuleNotFoundError:
    S3DISDataset = None
    S3DISSegDataset = None

try:
    from .scannet_dataset import (ScanNetDataset, ScanNetInstanceSegDataset,
                                  ScanNetSegDataset)
except ModuleNotFoundError:
    ScanNetDataset = None
    ScanNetInstanceSegDataset = None
    ScanNetSegDataset = None

# yapf: enable
from .sunrgbd_dataset import SUNRGBDDataset
# yapf: disable
from .transforms import (AffineResize, BackgroundPointsFilter, GlobalAlignment,
                         GlobalRotScaleTrans, IndoorPatchPointSample,
                         IndoorPointSample, LoadAnnotations3D,
                         LoadPointsFromDict, LoadPointsFromFile,
                         LoadPointsFromMultiSweeps, NormalizePointsColor,
                         ObjectNameFilter, ObjectNoise, ObjectRangeFilter,
                         ObjectSample, PointSample, PointShuffle,
                         PointsRangeFilter, RandomDropPointsColor,
                         RandomFlip3D, RandomJitterPoints, RandomResize3D,
                         RandomShiftScale, Resize3D, VoxelBasedPointSampler)
from .utils import get_loading_pipeline
from .waymo_dataset import WaymoDataset

__all__ = [
    'KittiDataset', 'RadarDataset', 'RadarTextDataset', 'CBGSDataset',
    'NuScenesDataset', 'LyftDataset', 'ObjectSample', 'RandomFlip3D',
    'ObjectNoise', 'GlobalRotScaleTrans', 'PointShuffle',
    'ObjectRangeFilter', 'PointsRangeFilter', 'LoadPointsFromFile',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'Det3DDataset', 'LoadPointsFromMultiSweeps', 'WaymoDataset',
    'BackgroundPointsFilter', 'VoxelBasedPointSampler',
    'get_loading_pipeline', 'RandomDropPointsColor', 'RandomJitterPoints',
    'ObjectNameFilter', 'AffineResize', 'RandomShiftScale',
    'LoadPointsFromDict', 'Resize3D', 'RandomResize3D'
]

if S3DISDataset is not None and S3DISSegDataset is not None:
    __all__.extend(['S3DISSegDataset', 'S3DISDataset'])

if (ScanNetDataset is not None and ScanNetSegDataset is not None
        and ScanNetInstanceSegDataset is not None):
    __all__.extend(
        ['ScanNetDataset', 'ScanNetSegDataset', 'ScanNetInstanceSegDataset'])
