_base_ = './pointpillars_hv_secfpn_8xb6-160e_radar-3d.py'

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

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
        image_dropout=0.1))

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
    dataset=dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne', img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality)))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        pipeline=test_pipeline,
        modality=input_modality))

test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne', img='training/image_2'),
        pipeline=test_pipeline,
        modality=input_modality))
