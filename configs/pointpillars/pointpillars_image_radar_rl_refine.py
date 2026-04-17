_base_ = './pointpillars_image_radar_rl.py'

custom_imports = dict(
    imports=[
        'mmdet3d.models.detectors.image_point_voxelnet_rl',
        'mmdet3d.models.fusion.image_radar_bev_fusion_rl',
        'mmdet3d.models.detectors.image_point_voxelnet_rl_refine',
        'mmdet3d.models.fusion.radar_guided_fusion_refine',
    ],
    allow_failed_imports=False)

work_dir = './work_dirs/pointpillars_image_radar_7d_rl_refine'

default_hooks = dict(
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1, max_keep_ckpts=10))

model = dict(
    type='ImagePointVoxelNetRLRefine',
    refine_block=dict(
        type='RadarGuidedFusionRefineCBAM',
        channels=384,
        reduction=16,
        spatial_kernel_size=7,
        radar_hidden_channels=128,
        refine_scale_init=0.1,
        return_refine_stats=False))
