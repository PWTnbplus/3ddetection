# 优化日志入口

正式版优化日志位于：

- [performance/optimization_log.md](/D:/文章/try/3ddetection/performance/optimization_log.md)

根目录文件只保留入口说明。
- 2026-04-20 10:42：重构 `mmdet3d/utils/text_hash.py`、`mmdet3d/models/fusion/fast_utils.py`、`mmdet3d/datasets/radar_text_dataset.py`、`mmdet3d/models/detectors/text_voxelnet.py`、`mmdet3d/models/fusion/bev_fusion.py`、`mmdet3d/models/fusion/triple_bev_fusion.py`、`mmdet3d/models/fusion/triple_rl_bev_fusion.py`，原因是旧实现存在重复文本哈希、逐样本小 tensor 构造、BEV 投影重复建网格、逐样本 `grid_sample` 和不必要的点云统计搬运；结果一致性影响：保持同一接口与主逻辑，`TripleModalRLBEVFusion` 的单文本 token 交叉注意力改为等价快速路径；使用方式影响：无。
