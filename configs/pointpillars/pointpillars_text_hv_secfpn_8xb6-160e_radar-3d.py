_base_ = './pointpillars_hv_secfpn_8xb6-160e_radar-3d.py'

dataset_type = 'RadarTextDataset'
backend_args = None
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

text_meta_keys = (
    'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
    'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'num_pts_feats', 'pcd_trans', 'sample_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pcd_rotation_angle', 'lidar_path',
    'transformation_3d_flow', 'trans_mat', 'affine_aug', 'sweep_img_metas',
    'ori_cam2img', 'cam2global', 'crop_offset', 'img_crop_offset',
    'resize_img_shape', 'lidar2cam', 'ori_lidar2img', 'num_ref_frames',
    'num_views', 'ego2global', 'axis_align_matrix', 'text')

model = dict(
    type='TextVoxelNet',
    text_hash_dim=512,
    text_channels=384,
    text_dropout=0.1)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=text_meta_keys)
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
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
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=text_meta_keys)
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            type=dataset_type,
            text_ann_file='texts/radar_llm_texts_prediction_train.json',
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        text_ann_file='texts/radar_llm_texts_prediction_val.json',
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        text_ann_file='texts/radar_llm_texts_prediction_test.json',
        pipeline=test_pipeline))
