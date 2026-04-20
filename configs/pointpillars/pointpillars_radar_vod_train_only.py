_base_ = ['./pointpillars_radar_vod.py']

# Local safety config for long Windows runs. Use the main config for full
# train+val, or this variant to postpone validation until a separate test run.
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
