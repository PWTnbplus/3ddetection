# 优化方案

## 1. 优化目标
- 优先加速真实可执行链路：VOD/radar infos 生成、文本模态生成、prompt 生成。
- 同步优化训练侧会在每 step 重复发生的文本哈希逻辑。
- 在不改任务语义和检测目标的前提下，尽量把收益最大的串行流程改成并发或缓存复用。

## 2. 方案与瓶颈映射

| 瓶颈 | 对应文件 | 计划动作 | 预期收益 | 风险 |
|---|---|---|---|---|
| infos 构建串行小文件 I/O | `tools/dataset_converters/radar_converter.py` | 引入多线程并发解析样本元信息 | 高 | 低 |
| 重复 folder 探测与 `stat` | `tools/dataset_converters/radar_converter.py` | 预扫描 `velodyne` 建样本到 folder 的索引 | 中 | 低 |
| 标定解析存在 Python 字符串拆分开销 | `tools/dataset_converters/radar_converter.py` | 用 `np.fromstring` 解析标定行 | 中 | 低 |
| 文本生成串行 + JSON 缩进输出过重 | `tools/dataset_converters/radar_text_converter.py` | 并发构建 record，改成紧凑流式 JSON 输出 | 很高 | 低 |
| prompt 生成仍是串行 | `tools/dataset_converters/radar_llm_prompt_converter.py` | 增加可选 worker 并发 | 低到中 | 低 |
| 训练时每 step 重新哈希文本 | `mmdet3d/datasets/radar_text_dataset.py` + 文本融合模型 | 在数据集初始化时预计算 `text_embedding`，模型优先复用 | 很高 | 低 |
| dataloader CPU 到 GPU 供数可能不满 | VOD/radar 相关配置 | 打开 `pin_memory`、补 `prefetch_factor`、打开 `cudnn_benchmark` | 中 | 低到中 |

## 3. 为什么这样改
- `radar_converter` 和 `radar_text_converter` 的热点都集中在样本级独立工作，非常适合线程并发。
- 文本 JSON 输出的最大问题不是算法，而是输出格式过重；改成紧凑流式写出，不影响数据语义，却能直接减少编码和写盘时间。
- 文本哈希属于固定文本的重复计算，最合理的位置应该是“数据集加载一次”，而不是“模型每个 batch 再算一遍”。
- dataloader 配置优化不会改变模型结构，风险低，但对 GPU 等待 CPU 的场景通常有效。

## 4. 预期收益
- infos 构建：`2x~3x`
- 文本生成：`4x+`
- 文本哈希前向阶段：`10x+`
- prompt 生成：小幅优化
- 整体多模态数据准备链路：`3x~4x`

## 5. 风险控制
- 并发改造保持输出顺序与 split 顺序一致，避免下游样本索引错乱。
- 文本嵌入保留原始 `text` 字段，不替换文本本身；模型只是在已有文本之上优先读取预计算向量。
- 对于 `text_hash_dim` 不一致的情况，模型侧统一做 pad/truncate，避免配置不匹配时直接报错。
- 对 `radar` 目录下损坏软链接未做原地修复，本轮只在 benchmark 中改用完整的 `lidar` 根目录验证性能。
