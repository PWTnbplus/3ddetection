_base_ = './pointpillars_hv_secfpn_8xb6-160e_radar-3d.py'

data_root = '__VOD_BASE__'

test_dataloader = dict(
    dataset=dict(ann_file='__INFO_ROOT_RADAR__/radar_infos_train.pkl'))

test_evaluator = dict(
    ann_file='__INFO_ROOT_RADAR__/radar_infos_train.pkl',
    pklfile_prefix='work_dirs/pointpillars_radar_train/predictions',
    submission_prefix='work_dirs/pointpillars_radar_train/submission')
