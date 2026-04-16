_base_ = './pointpillars_image_radar.py'

custom_imports = dict(
    imports=[
        'mmdet3d.models.detectors.image_point_voxelnet_rl',
        'mmdet3d.models.fusion.image_radar_bev_fusion_rl',
    ],
    allow_failed_imports=False)

work_dir = './work_dirs/pointpillars_image_radar_7d_rl'

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=6))

model = dict(
    type='ImagePointVoxelNetRL',
    fusion=dict(
        type='ImageRadarBEVFusionRL',
        gate_hidden_channels=128,
        gate_temperature=1.0,
        gate_entropy_weight=0.0,
        return_gate_stats=False))

train_cfg = dict(val_interval=1)
