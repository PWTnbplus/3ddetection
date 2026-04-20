# 性能优化说明入口

正式版性能优化说明位于：

- [performance/README_性能优化说明.md](/D:/文章/try/3ddetection/performance/README_性能优化说明.md)

根目录文件只保留入口说明，正式使用方法和 benchmark 命令请查看 `performance/` 目录下的完整文档。
- 2026-04-20 10:42：完成第一轮核心热路径重构，统一文本特征快路径、加入融合统计快工具、把图像投影到 BEV 改成缓存网格 + 批量 `grid_sample`，并将单文本 token 的交叉注意力改为等价快速路径；结果一致性影响：接口保持不变，单 token 注意力数值逻辑保持等价；使用方式影响：无。
