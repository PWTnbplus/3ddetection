_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

dataset_type = 'RadarDataset'
data_root = 'dataset/radar/'
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

class_names = [
    'bicycle', 'Car', 'Cyclist', 'motor', 'Pedestrian', 'ride_other', 'truck'
]
metainfo = dict(classes=class_names)

anchor_ranges = [
    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
    [0, -39.68, -1.78, 69.12, 39.68, -1.78],
    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
    [0, -39.68, -1.78, 69.12, 39.68, -1.78],
]

anchor_sizes = [
    [1.843, 0.594, 1.174],
    [4.165, 1.839, 1.573],
    [1.946, 0.777, 1.751],
    [2.250, 1.007, 1.601],
    [0.649, 0.637, 1.650],
    [1.207, 0.736, 1.114],
    [6.814, 2.492, 2.765],
]

model = dict(
    type='ImagePointVoxelNet',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    fusion=dict(
        type='ImageRadarBEVFusion',
        image_channels=256,
        bev_channels=384,
        image_dropout=0.1),
    voxel_encoder=dict(point_cloud_range=point_cloud_range),
    bbox_head=dict(
        num_classes=len(class_names),
        anchor_generator=dict(
            ranges=anchor_ranges,
            sizes=anchor_sizes,
            rotations=[0, 1.57],
            reshape_out=False)),
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1) for _ in class_names
        ]))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='radar_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne', img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        ann_file='radar_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        ann_file='radar_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'radar_infos_val.pkl',
    metric='bbox',
    format_only=True,
    pklfile_prefix='work_dirs/pointpillars_radar/predictions',
    submission_prefix='work_dirs/pointpillars_radar/submission',
    backend_args=backend_args)
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'radar_infos_test.pkl',
    metric='bbox',
    format_only=True,
    pklfile_prefix='work_dirs/pointpillars_radar_test/predictions',
    submission_prefix='work_dirs/pointpillars_radar_test/submission',
    backend_args=backend_args)

lr = 0.001
epoch_num = 80
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
val_cfg = dict()
test_cfg = dict()
