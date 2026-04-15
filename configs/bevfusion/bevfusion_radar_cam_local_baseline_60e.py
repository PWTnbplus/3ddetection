_base_ = './bevfusion_radar_cam_local_baseline.py'

work_dir = './work_dirs/bevfusion_radar_cam_local_baseline'

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=60,
        end=60,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=24,
        end=60,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)