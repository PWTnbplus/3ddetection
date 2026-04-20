# 性能优化说明

## 1. 本次优化覆盖了什么
- VOD/radar infos 生成加速
- 样本级文本模态生成加速
- LLM prompt 构建加速
- 文本检测/三模态融合模型中的文本哈希重复计算消除
- VOD/radar 主线配置的 dataloader 吞吐优化

## 2. 默认行为变化

### 2.1 `radar_text_converter.py`
- 现在默认输出紧凑 JSON，不再默认带缩进。
- 如果你需要人工查看友好的格式，可以显式加 `--pretty-json`。

### 2.2 `RadarTextDataset`
- 现在会默认预计算 `text_embedding`。
- 模型侧会优先使用 `text_embedding`，只有缺失时才退回在线哈希。

## 3. 如何开启优化版数据准备

### 3.1 生成 VOD/radar infos
```bash
python tools/create_data.py vod \
  --root-path D:/文章/try/VOD/view_of_delft_PUBLIC/radar \
  --out-dir D:/文章/try/VOD/view_of_delft_PUBLIC/radar \
  --extra-tag radar \
  --workers 8
```

### 3.2 生成文本模态
```bash
python tools/dataset_converters/radar_text_converter.py \
  --data-root D:/文章/try/VOD/view_of_delft_PUBLIC/radar \
  --source label \
  --workers 8
```

### 3.3 生成 LLM prompt
```bash
python tools/dataset_converters/radar_llm_prompt_converter.py \
  --input-file D:/文章/try/VOD/view_of_delft_PUBLIC/radar/texts/radar_texts_full.json \
  --output-file D:/文章/try/VOD/view_of_delft_PUBLIC/radar/texts/radar_llm_prompts_full.jsonl \
  --workers 8
```

## 4. 如何跑 benchmark
```bash
python tools/perf/benchmark_radar_pipeline.py \
  --data-root D:/文章/try/VOD/view_of_delft_PUBLIC/lidar \
  --output-dir D:/文章/try/3ddetection/performance_benchmark_run \
  --workers 8 \
  --text-hash-dim 768 \
  --repeat 20 \
  --skip-lidar-check
```

说明：
- `--skip-lidar-check` 只在你用 `lidar` 根目录模拟 benchmark 时使用。
- 如果你的 `radar` 根目录软链接完整，可以直接改成 `radar` 根目录并去掉这个参数。

## 5. 训练侧新增的性能收益点
- `RadarTextDataset` 会把 `text` 变成 `text_embedding`
- `TextVoxelNet` / `TripleModalBEVFusion` / `TripleModalRLBEVFusion` 优先复用 `text_embedding`
- 文本构建路径现在会对重复文本做批量去重，不再为同一句文本重复创建哈希 tensor
- `TripleModalBEVFusion` / `TripleModalRLBEVFusion` 的图像统计、点云统计、文本置信度改为共享快路径
- `ImageRadarBEVFusion` 的图像投影到 BEV 现在使用缓存网格和更快的“单样本 `grid_sample` + 共享投影模板”快路径
- `TripleModalRLBEVFusion` 对单文本 token 的交叉注意力使用等价快速路径，避免无意义的全查询注意力展开
- 相关配置已经补齐：
  - `pin_memory=True`
  - `prefetch_factor`
  - 部分 VOD 图像配置开启 `cudnn_benchmark=True`

## 6. 哪些地方仍然值得继续优化
- `_point_stats` 虽然已经去掉了“逐样本搬到 GPU 再算”的慢路径，但仍然保留了变长点集逐样本聚合；如果后续拿到稳定训练环境，可以继续评估是否值得把统计量前移到数据准备阶段。
- 模型端真正的 voxelization、稀疏卷积、NMS、后处理等 CUDA 热点还需要在可运行的训练/推理环境里做 Nsight 或 torch profiler 级别审计。
- 如果后续数据规模继续扩大，可以考虑把文本生成阶段改成 JSONL 索引和分片输出，进一步降低单文件读写压力。

## 7. 最新变更记录
- 2026-04-20 10:42：完成第二轮模型热路径重构，统一文本特征快路径、加入融合统计快工具、把图像投影到 BEV 改成缓存网格 + 批量 `grid_sample`，并将单文本 token 的交叉注意力改为等价快速路径；结果一致性影响：接口保持不变，数值逻辑保持等价；使用方式影响：无。
- 2026-04-20 10:58：扩展 `tools/perf/benchmark_radar_pipeline.py`，让 benchmark 同时覆盖数据链路和模型热点微基准；结果一致性影响：无；使用方式影响：benchmark 输出字段新增 `model_fastpaths`。
- 2026-04-20 11:06：根据实测 benchmark 回退了“批量 `grid_sample`”方案，改成“缓存网格 + 单样本 `grid_sample`”的更快实现；结果一致性影响：无；使用方式影响：无。
- 2026-04-20 11:11：修正 benchmark 的投影计时口径，改成复用预热后的投影缓存，使结果更接近真实训练/推理中的持续迭代场景；结果一致性影响：无；使用方式影响：benchmark 报告中的投影加速结果更准确。
- 2026-04-20 11:16：进一步对无增强场景启用齐次坐标缓存直通，减少投影快路径里的 `torch.cat` 和中间张量构造；结果一致性影响：无；使用方式影响：无。
- 2026-04-20 11:23：同步刷新最终性能报告和 `benchmark_metrics.json`，写入第二轮模型热点的实际提速数据；结果一致性影响：无；使用方式影响：性能说明中的数值已更新到最终版本。
