# 性能审计报告

## 1. 审计范围
- 审计对象：当前瘦身后的 3D 点云检测工程主线，重点覆盖 VOD/radar 数据准备、文本模态生成、文本融合前向热点、训练配置中的 dataloader 吞吐设置。
- 实际执行的动态 profiling：`tools/dataset_converters/radar_converter.py`、`tools/dataset_converters/radar_text_converter.py`、`tools/dataset_converters/radar_llm_prompt_converter.py`。
- 静态代码审计：`mmdet3d/models/detectors/text_voxelnet.py`、`mmdet3d/models/fusion/triple_bev_fusion.py`、`mmdet3d/models/fusion/triple_rl_bev_fusion.py`、VOD/radar 相关训练配置。

## 2. 审计说明
- 当前环境里的 `mmcv` 安装异常，`import mmdet3d` 会在 `mmdet` 读取 `mmcv.__version__` 时失败，因此无法直接跑完整训练/推理基准。
- 为了保证审计结论仍然可信，本次采用两层方法：
  - 第一层：对真实 VOD 数据准备链路做可执行 profiling，定位数据读取、标注解析、JSON 落盘等真实热点。
  - 第二层：对模型侧自定义文本融合模块做源码级热点审计，并对可独立测量的文本哈希环节做微基准。

## 3. 识别出的主线入口
- 数据入口：[tools/create_data.py](/D:/文章/try/3ddetection/tools/create_data.py)
- VOD/radar infos 构建：[radar_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_converter.py)
- 文本模态生成：[radar_text_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_text_converter.py)
- LLM prompt 构建：[radar_llm_prompt_converter.py](/D:/文章/try/3ddetection/tools/dataset_converters/radar_llm_prompt_converter.py)
- 文本检测模型入口：[text_voxelnet.py](/D:/文章/try/3ddetection/mmdet3d/models/detectors/text_voxelnet.py)
- 三模态融合入口：[triple_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_bev_fusion.py)
- 三模态 RL 融合入口：[triple_rl_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_rl_bev_fusion.py)

## 4. 动态 profiling 热点

### 4.1 `radar_converter.create_vod_infos` 基线热点
- 基线耗时：`25.51s`，样本数 `8682`
- 热点排序：
  - `build_data_info` 累积约 `25.00s`
  - `read_jpeg_size` 累积约 `8.07s`
  - `read_calib` 累积约 `6.01s`
  - `parse_label_file` 累积约 `5.19s`
  - `choose_sample_folder` 累积约 `3.47s`
  - `pathlib.exists/stat/open` 累积约 `13s+`
- 根因：
  - 全流程按样本串行执行，小文件 I/O 无并发。
  - 每个样本都重复做 folder 探测和 `stat`。
  - 图像尺寸、标定、标签解析全部在 Python 主线程串行完成。

### 4.2 `radar_text_converter` 基线热点
- 基线耗时：`72.12s`，处理记录数 `21552`
- 热点排序：
  - `json.dump(..., indent=2)` 及其递归编码约 `49.48s`
  - `parse_label_file` 约 `18.26s`
  - `Path.exists/open/stat` 约 `5s+`
  - `build_record_from_objects` 约 `3.65s`
  - `summarize_objects` 约 `2.54s`
- 根因：
  - 输出用了带缩进的整块 JSON，编码和写盘成本远大于实际文本构建。
  - label 解析仍是串行单线程。
  - 先构造大列表再整体 `dump`，内存和编码都偏重。

### 4.3 `radar_llm_prompt_converter` 基线热点
- 基线耗时：`2.68s`，记录数 `8682`
- 热点排序：
  - `load_json` 约 `1.02s`
  - `build_prompt_record/build_user_prompt` 约 `1.30s`
  - `build_object_lines` 约 `0.82s`
- 结论：
  - 这条链路不是主瓶颈，收益空间明显小于 infos 构建和文本生成。

### 4.4 训练侧静态热点
- [text_voxelnet.py](/D:/文章/try/3ddetection/mmdet3d/models/detectors/text_voxelnet.py) 每个 batch 都重新正则分词并做 `zlib.crc32` 哈希。
- [triple_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_bev_fusion.py) 与 [triple_rl_bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/triple_rl_bev_fusion.py) 重复做同样的文本哈希逻辑。
- 三模态配置里 dataloader 之前没有统一打开 `pin_memory/prefetch_factor`，且 VOD BEVFusion 配置把 `cudnn_benchmark` 关掉。
- [bev_fusion.py](/D:/文章/try/3ddetection/mmdet3d/models/fusion/bev_fusion.py) 旧版本会在每次前向重复构建投影网格和齐次坐标，属于高频重复准备成本。
- `_point_stats` 依然保留了变长点集逐样本聚合，但已经确认“逐样本搬到 GPU 再算”是明确慢点，适合改成 CPU 聚合后只搬运小统计结果。

## 5. 优先级排序
1. 先处理数据准备串行 I/O，因为这是最明确、收益最大的真实瓶颈。
2. 处理文本生成 JSON 落盘和串行 label 解析，因为这条链路直接影响多模态数据构建吞吐。
3. 处理训练前向里的重复文本哈希，因为它每 step 都会发生，属于错误层级的重复计算。
4. 补齐 dataloader 吞吐配置，减少 CPU 到 GPU 的供数空转。
5. 重构 `_point_stats` 与图像投影准备路径，优先消灭无意义的数据搬运和重复网格构造。

## 6. 审计结论
- 当前项目最慢的核心原因不是单个“大算子”，而是多个高频小开销叠加：
  - 小文件串行读取
  - 重复 `stat/open`
  - 逐样本串行文本生成
  - 过重的缩进 JSON 输出
  - 训练时反复对同一文本做哈希
- 因此最有效的优化方向不是调参，而是：
  - 并发化样本处理
  - 消除重复 folder/path 探测
  - 文本嵌入预计算
  - 数据加载参数补齐
  - 融合模块快路径共享
  - 投影模板缓存与统计路径瘦身
