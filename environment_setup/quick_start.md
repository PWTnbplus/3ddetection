# 快速启动

## 1. 激活环境

```powershell
D:\codex_envs\3ddet_env\Scripts\activate
```

如果你希望避开 Windows 中文路径带来的兼容性问题，建议通过 ASCII 别名路径进入项目：

```powershell
cd D:\3ddetection_ascii
```

如果直接使用原路径，也可以：

```powershell
cd D:\文章\try\3ddetection
```

## 2. 快速检查环境是否正常

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

D:\codex_envs\3ddet_env\Scripts\python.exe -c "import mmengine, mmcv, mmdet, mmdet3d, open3d; print(mmengine.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__, open3d.__version__)"

D:\codex_envs\3ddet_env\Scripts\python.exe tools\train.py -h
D:\codex_envs\3ddet_env\Scripts\python.exe tools\test.py -h
D:\codex_envs\3ddet_env\Scripts\python.exe tools\create_data.py -h
```

## 3. 加载当前主线配置

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe -c "from mmengine.config import Config; cfg = Config.fromfile(r'D:\3ddetection_ascii\configs\pointpillars\pointpillars_image_radar_rl_align.py'); print(cfg.model.type)"
```

## 4. 正式训练示例

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\train.py configs\pointpillars\pointpillars_image_radar_rl_align.py
```

说明：

- 正式训练前请先确保 VOD 或目标数据集路径已经按配置文件要求准备好。
- 若需要多卡训练，可继续使用仓库现有 `dist_train.sh` 或 OpenMMLab 分布式启动方式。

## 5. 测试示例

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\test.py <config_path> <checkpoint_path>
```

## 6. 数据准备示例

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\create_data.py kitti --root-path <dataset_root> --out-dir <output_dir>
```

## 7. 重要产物位置

- 依赖清单：`environment_setup/requirements_resolved.txt`
- 环境描述：`environment_setup/environment_resolved.yml`
- 安装日志：`environment_setup/install_log.txt`
- 环境检查报告：`environment_setup/env_check_report.md`
- 当前文档：`environment_setup/quick_start.md`
