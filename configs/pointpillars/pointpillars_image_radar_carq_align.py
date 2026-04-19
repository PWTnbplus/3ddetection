_base_ = './pointpillars_image_radar.py'

# Local aliases are needed because variables from _base_ are not in this
# config file's Python execution scope before MMEngine merges configs.
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car']


custom_imports = dict(
    imports=[
        'mmdet3d.models.detectors.image_point_voxelnet_carq',
        'mmdet3d.models.fusion.carq_image_radar_fusion',
        'mmdet3d.models.roi_heads.carq_align_refiner',
    ],
    allow_failed_imports=False)

work_dir = './work_dirs/pointpillars_image_radar_7d_carq_align'

default_hooks = dict(
    checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1, max_keep_ckpts=10))

model = dict(
    type='ImagePointVoxelNetCARQ',
    fusion=dict(
        type='CARQImageRadarBEVFusion',
        gate_hidden_channels=128,
        gate_temperature=1.0,
        gate_entropy_weight=0.0,
        return_gate_stats=False),
    proposal_refiner=dict(
        type='RadarCameraProposalRefiner',
        train_jitter_times=1,
        max_train_proposals=96,
        topk_test=(30, 30, 50),
        score_thr=0.05,
        loss_refine_weight=1.0,
        loss_gate_weight=0.2,
        loss_align_weight=0.05,
        loss_stability_weight=0.02,
        align_module=dict(
            type='InstanceCrossModalAlign',
            in_channels=384,
            embed_dims=128,
            point_cloud_range=point_cloud_range,
            grid_size=5,
            # class order follows the config: Pedestrian, Cyclist, Car.
            enlarge_ratio=(2.0, 2.0, 1.5)),
        refine_head=dict(
            type='ClassAwareRefineHead',
            embed_dims=128,
            num_classes=len(class_names),
            class_dim_masks=(
                (1.0, 1.0, 0.5, 0.2, 0.2, 0.5, 0.1),
                (1.0, 1.0, 0.7, 0.5, 0.8, 0.5, 0.8),
                (1.0, 1.0, 1.5, 1.5, 1.5, 1.2, 1.2))),
        quality_gate=dict(
            type='GeometryReliabilityGate',
            embed_dims=128,
            vector_gate=True,
            # Conservative start: early training stays close to coarse boxes.
            init_bias=-2.0)))

train_cfg = dict(val_interval=1)


