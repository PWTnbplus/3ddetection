# 优化日志

## 1. 已修改文件
- [radar_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_converter.py)
- [create_data.py](/D:/文章/try/3ddetection/tools/create_data.py)
- [radar_text_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_text_converter.py)
- [radar_llm_prompt_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_llm_prompt_converter.py)
- [text_hash.py](/D:/文章/try/3ddetection/mmdet3d/utils/text_hash.py)
- [__init__.py](/D:/文章/try/3ddetection/mmdet3d/utils/__init__.py)
- [radar_text_dataset.py](/D:/文章/try/3ddetection/mmdet3d/datasets/radar_text_dataset.py)
- [text_voxelnet.py](/D:/文章/try/3ddetection/mmdet3d/models/detectors/text_voxelnet.py)
- [triple_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_bev_fusion.py)
- [triple_rl_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_rl_bev_fusion.py)
- [pointpillars_hv_secfpn_8xb6-160e_radar-3d.py](/D:/文章/try/3ddetection/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_radar-3d.py)
- [pointpillars_triple_modal_radar.py](/D:/文章/try/3ddetection/configs/pointpillars/pointpillars_triple_modal_radar.py)
- [bevfusion_radar_cam_aligned_baseline_vod_3d.py](/D:/文章/try/3ddetection/configs/bevfusion/bevfusion_radar_cam_aligned_baseline_vod_3d.py)
- [bevfusion_radar_cam_aligned_baseline_vod_3d_12e.py](/D:/文章/try/3ddetection/configs/bevfusion/bevfusion_radar_cam_aligned_baseline_vod_3d_12e.py)
- [benchmark_radar_pipeline.py](/D:/文章/try/3ddetection/tools/perf/benchmark_radar_pipeline.py)

## 2. 逐项变更说明

### 2.1 `radar_converter.py`
- 增加 `--workers` 参数。
- 增加 `build_sample_folder_index`，预扫描 `training/testing/velodyne`，避免每个样本重复探测 folder。
- `read_split_ids` 改成一次性读取。
- `read_calib` 改用 `np.fromstring`，减少 Python 数值拆分开销。
- `validate_lidar_7d` 避免重复 `stat`。
- `convert_split` 支持线程并发并保持样本顺序。

### 2.2 `create_data.py`
- 把 CLI 的 `workers` 传递给 radar 数据准备函数，避免入口层丢失并发能力。

### 2.3 `radar_text_converter.py`
- 增加 `--workers` 和 `--pretty-json`。
- `build_records_for_split` 支持并发构建样本文本。
- `dump_json` 改成紧凑流式写出，默认不再做大规模缩进 JSON。
- `read_split_ids` 改成一次性读取。

### 2.4 `radar_llm_prompt_converter.py`
- 增加 `--workers`，支持 prompt 构建并发。

### 2.5 文本哈希公共能力
- 新增 [text_hash.py](/D:/文章/try/3ddetection/mmdet3d/utils/text_hash.py)。
- 把文本哈希抽成公共工具，避免多个模型重复维护相同逻辑。

### 2.6 `RadarTextDataset`
- 新增 `text_hash_dim` 与 `precompute_text_embedding`。
- 数据集初始化时把 `text` 预计算成 `text_embedding`，样本进入模型时直接复用。

### 2.7 文本融合模型
- `TextVoxelNet`、`TripleModalBEVFusion`、`TripleModalRLBEVFusion` 全部改成优先读取 `text_embedding/text_feat`。
- 只有缺失预计算向量时才回退到在线哈希。

### 2.8 配置优化
- 对 VOD/radar 相关主线配置补齐 `pin_memory=True`。
- 对训练和验证 dataloader 补齐 `prefetch_factor`。
- 对图像分辨率较稳定的 VOD BEVFusion/三模态配置开启 `cudnn_benchmark=True`。

### 2.9 基准脚本
- 新增 [benchmark_radar_pipeline.py](/D:/文章/try/3ddetection/tools/perf/benchmark_radar_pipeline.py)，用于复跑当前优化版的转换、文本生成、prompt 生成和文本哈希基准。

### 2.10 第二轮模型热路径重构
- 新增 [fast_utils.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/fast_utils.py)，把图像统计、点云统计、文本置信度构建抽成公共快路径。
- [text_hash.py](/D:/文章/try/3ddetection/mmdet3d/utils/text_hash.py) 增加整句哈希缓存、批量去重构建、样本级统一文本特征构建函数。
- [radar_text_dataset.py](/D:/文章/try/3ddetection/mmdet3d/datasets/radar_text_dataset.py) 在预计算 `text_embedding` 时加入文本去重缓存，避免重复样本文本反复哈希。
- [text_voxelnet.py](/D:/文章/try/3ddetection/mmdet3d/models/detectors/text_voxelnet.py) 改成直接走统一批量文本快路径，消除 detector 侧重复实现。
- [triple_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_bev_fusion.py) 和 [triple_rl_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_rl_bev_fusion.py) 改成共享文本/图像/点云统计快路径，减少逐样本小 tensor 构造和额外设备搬运。
- [bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/bev_fusion.py) 把图像投影到 BEV 改成缓存网格 + 更快的单样本 `grid_sample` 快路径，并把旋转逆变换从通用矩阵求逆收敛为转置快路径。
- `TripleModalRLBEVFusion` 的单文本 token 交叉注意力改成等价快速路径，避免对单 token 仍执行完整 `MultiheadAttention` 查询展开。
- [benchmark_radar_pipeline.py](/D:/文章/try/3ddetection/tools/perf/benchmark_radar_pipeline.py) 扩展出模型侧微基准，对文本哈希、样本文本构建、点云统计、单 token 注意力、图像投影到 BEV 进行旧路径/新路径对比。

## 3. 增量变更记录

- 2026-04-20 10:42：重构 `mmdet3d/utils/text_hash.py`、`mmdet3d/models/fusion/fast_utils.py`、`mmdet3d/datasets/radar_text_dataset.py`、`mmdet3d/models/detectors/text_voxelnet.py`、`mmdet3d/models/fusion/bev_fusion.py`、`mmdet3d/models/fusion/triple_bev_fusion.py`、`mmdet3d/models/fusion/triple_rl_bev_fusion.py`，原因是旧实现存在重复文本哈希、逐样本小 tensor 构造、BEV 投影重复建网格、逐样本 `grid_sample` 和不必要的点云统计搬运；结果一致性影响：保持同一接口与主逻辑，单 token 注意力采用等价快速路径；使用方式影响：无。
- 2026-04-20 10:58：扩展 `tools/perf/benchmark_radar_pipeline.py` 为“数据链路 + 模型热点微基准”双层 benchmark，并修正 `bev_fusion.py` 中投影缓存变量覆盖细节，原因是需要把第二轮模型热路径优化量化出来并保证快路径稳定；结果一致性影响：无；使用方式影响：benchmark 输出会新增 `model_fastpaths` 指标。
- 2026-04-20 11:06：根据实测 benchmark，将 `mmdet3d/models/fusion/bev_fusion.py` 和 `tools/perf/benchmark_radar_pipeline.py` 的图像投影快路径调整为“缓存网格 + 单样本 `grid_sample`”混合方案，原因是批量 `grid_sample` 在当前 CUDA 形状下反而更慢；结果一致性影响：无，投影数值保持一致；使用方式影响：无。
- 2026-04-20 11:11：修正 `tools/perf/benchmark_radar_pipeline.py` 的投影基准口径，改为复用预热后的投影缓存，原因是实际模块会跨迭代复用缓存，不应把缓存构建时间重复记入每次前向；结果一致性影响：无；使用方式影响：benchmark 中 `image_to_bev_projection` 的结果更贴近真实部署。
- 2026-04-20 11:16：继续优化 `mmdet3d/models/fusion/bev_fusion.py` 和 `tools/perf/benchmark_radar_pipeline.py`，在无 3D 增强流时直接复用缓存的齐次坐标，避免每次前向重复 `torch.cat`；结果一致性影响：无；使用方式影响：无。
- 2026-04-20 11:23：更新 `performance/performance_audit_report.md`、`performance/speed_benchmark_before_after.md`、`performance/benchmark_metrics.json`，写入最终 benchmark 数值和第二轮模型热点结果，原因是需要让最终文档与当前代码和实测结果完全一致；结果一致性影响：无；使用方式影响：文档中的加速结论已切换到最新实测值。

## 3. 未做的高风险改动
- 未把 `_point_stats` 完全改写成全批处理版本；当前只去掉了“逐样本搬到 GPU 再算”的明显慢路径，因为它依赖变长点集和在线增强结果，继续做更激进的批处理仍可能影响训练行为。
- 未改 NMS、voxelization 或 CUDA/算子级实现，因为当前环境无法稳定跑完整训练做正确性回归。
