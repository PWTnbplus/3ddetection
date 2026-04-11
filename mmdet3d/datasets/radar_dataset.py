from mmdet3d.registry import DATASETS

from .kitti_dataset import KittiDataset


@DATASETS.register_module()
class RadarDataset(KittiDataset):
    """KITTI-style radar dataset with project-specific class names."""

    METAINFO = {
        'classes': ('bicycle', 'bicycle_rack', 'Car', 'Cyclist',
                    'human_depiction', 'moped_scooter', 'motor',
                    'Pedestrian', 'ride_other', 'ride_uncertain', 'rider',
                    'truck', 'vehicle_other'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),
                    (0, 0, 192), (197, 226, 255), (0, 60, 100),
                    (0, 0, 142), (255, 77, 255), (153, 69, 1),
                    (120, 166, 157), (0, 182, 199), (255, 179, 0),
                    (255, 0, 0)]
    }
