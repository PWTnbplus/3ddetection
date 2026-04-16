_base_ = './pointpillars_image_radar_rl.py'

custom_imports = dict(
    imports=[
        'mmdet3d.models.detectors.image_point_voxelnet_rl',
        'mmdet3d.models.fusion.image_radar_bev_fusion_rl',
        'mmdet3d.models.detectors.image_point_voxelnet_rl_align',
        'mmdet3d.models.fusion.image_radar_bev_align',
        'mmdet3d.models.losses.foreground_alignment_loss',
    ],
    allow_failed_imports=False)

work_dir = './work_dirs/pointpillars_image_radar_7d_rl_align'

default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', interval=1))

model = dict(
    type='ImagePointVoxelNetRLAlign',
    align_module=dict(
        type='RadarGuidedBEVAlign',
        channels=384,
        hidden_channels=128,
        mode='residual_align',
        detach_radar_guidance=False,
        return_align_stats=False),
    alignment_loss=dict(
        type='ForegroundAlignmentLoss',
        loss_weight=0.05,
        use_cosine=True,
        eps=1e-6))
