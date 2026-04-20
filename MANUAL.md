# 3DDetection Local Manual

## 1. Project scope

This repository is a Windows-localized MMDetection3D-based project for VOD
radar and radar-camera 3D object detection.

The local deployment target in this repair pass is:

- Repository: `D:\文章\try\3ddetection`
- ASCII repo alias: `D:\3ddetection_ascii`
- Dataset root provided by the user: `D:\文章\try\VOD`
- ASCII dataset alias used by the local tooling: `D:\VOD_ascii`
- Python environment: `D:\codex_envs\3ddet_env`

The main repair target was the PointPillars VOD radar training path. The same
local-path injection was also applied to the related radar and BEVFusion
configs.

## 2. What was fixed

- Removed hard-coded Linux dataset paths such as
  `/root/lanyun-fs/dataset/radar_5frames/...` from the active radar configs
  and scripts.
- Added runtime local-path injection in `tools/train.py` and `tools/test.py`
  so one local dataset root can drive multiple configs.
- Added `local_paths.py` to centralize local dataset, info, text, and work-dir
  resolution.
- Reworked KITTI/VOD rotated IoU evaluation so validation no longer hard
  depends on `numba.cuda` / `nvvm.dll`; the local path now defaults to
  `mmcv.ops.box_iou_rotated` and can fall back to a pure Python backend.
- Added `local_preflight.py` to verify Python, CUDA, dataset paths, info
  files, evaluator configuration, and the selected rotate IoU backend before
  training or testing starts.
- Added Windows-local batch entrypoints:
  `prepare_data_local.bat`, `train_local.bat`, and `test_local.bat`.
- Reworked the VOD radar info converter so it can start from the top-level VOD
  directory and generate valid info files even when Linux symlink layout is not
  directly usable on Windows.
- Hardened calibration parsing for empty `Tr_imu_to_velo` fields.
- Generated local info files for both:
  `view_of_delft_PUBLIC/infos/radar_5frames/`
  and `view_of_delft_PUBLIC/infos/radar/`.

## 3. Environment setup

Use the provided environment directly:

```powershell
D:\codex_envs\3ddet_env\Scripts\activate
cd /d D:\3ddetection_ascii
```

If you prefer not to activate the environment, every command in this manual can
be executed with:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe ...
```

## 4. Dataset layout expected by the project

The local tools resolve the VOD dataset from one of these inputs:

- `D:\VOD_ascii`
- `D:\VOD_ascii\view_of_delft_PUBLIC`
- `D:\VOD_ascii\view_of_delft_PUBLIC\radar`
- `D:\VOD_ascii\view_of_delft_PUBLIC\radar_5frames`

The repository now treats `view_of_delft_PUBLIC` as the shared `data_root`
because the image, label, split, and calibration assets are shared across
modalities.

Important folders:

- `D:\VOD_ascii\view_of_delft_PUBLIC\lidar`
- `D:\VOD_ascii\view_of_delft_PUBLIC\radar`
- `D:\VOD_ascii\view_of_delft_PUBLIC\radar_5frames`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames`

Generated info files:

- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_train.pkl`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_val.pkl`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_test.pkl`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar\radar_infos_train.pkl`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar\radar_infos_val.pkl`
- `D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar\radar_infos_test.pkl`

## 5. Local path configuration

The active local path knobs are:

- `MMD3D_VOD_ROOT`
  Points to the VOD root, the `view_of_delft_PUBLIC` root, or a modality
  directory. Default fallback is `D:\VOD_ascii`.
- `MMD3D_VOD_MODALITY`
  Used by local data-preparation scripts. Example: `radar` or
  `radar_5frames`.
- `MMD3D_WORK_ROOT`
  Optional override for work directory root.
- `MMD3D_ROTATE_IOU_BACKEND`
  Optional override for validation/test rotated IoU backend. Supported values:
  `auto`, `mmcv`, `python`. The local batch scripts default to `mmcv` because
  it avoids the Windows `nvvm.dll` dependency.

Runtime configs now use placeholders such as `__VOD_BASE__` and
`__INFO_ROOT_RADAR_5FRAMES__`. `tools/train.py` and `tools/test.py` replace
them at runtime using `local_paths.py`.

## 6. Data preparation

### One-click local preparation

```powershell
cd /d D:\3ddetection_ascii
prepare_data_local.bat
```

### Direct Python command

For `radar_5frames`:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\create_data.py vod ^
  --root-path D:\VOD_ascii ^
  --out-dir D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames ^
  --extra-tag radar ^
  --modality radar_5frames ^
  --workers 4
```

For `radar`:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\dataset_converters\radar_converter.py ^
  --data-root D:\VOD_ascii ^
  --modality radar ^
  --out-dir D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar ^
  --pkl-prefix radar ^
  --workers 4
```

### Optional integrity check

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\diagnose_radar_infos.py ^
  --data-root D:\VOD_ascii\view_of_delft_PUBLIC ^
  --train-pkl D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_train.pkl ^
  --val-pkl D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_val.pkl ^
  --test-pkl D:\VOD_ascii\view_of_delft_PUBLIC\infos\radar_5frames\radar_infos_test.pkl ^
  --sample-count 3
```

### Preflight before train/test

Run this before starting a long local job:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe local_preflight.py ^
  --config configs\pointpillars\pointpillars_radar_vod.py ^
  --dataset-root D:\文章\try\VOD
```

The preflight report now checks:

- Current Python executable and package stack
- `torch.cuda.is_available()`
- `numba` + NVVM status
- Resolved VOD root
- Required `radar_infos_*.pkl` files
- Whether the configured evaluator will trigger KITTI/VOD rotated IoU
- Which rotated IoU backend will actually be used

## 7. Training

### Main repaired training path

```powershell
cd /d D:\文章\try\3ddetection
train_local.bat configs\pointpillars\pointpillars_radar_vod.py work_dirs\pointpillars_radar_vod_local
```

Equivalent direct command:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe local_preflight.py ^
  --config configs\pointpillars\pointpillars_radar_vod.py ^
  --dataset-root D:\文章\try\VOD

D:\codex_envs\3ddet_env\Scripts\python.exe tools\train.py ^
  configs\pointpillars\pointpillars_radar_vod.py ^
  --work-dir work_dirs\pointpillars_radar_vod_local
```

This path now performs validation every epoch and no longer requires
`nvvm.dll` to finish KITTI/VOD evaluation.

### Optional train-only safety config

If you want a long local training run without mid-training validation, use:

```powershell
train_local.bat configs\pointpillars\pointpillars_radar_vod_train_only.py work_dirs\pointpillars_radar_vod_train_only
```

This is only a transition/safety mode. Use
`configs\pointpillars\pointpillars_radar_vod.py` for the full train+val path,
and run `tools\test.py` or `test_local.bat` afterwards to produce metrics.

### Useful verification run

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe tools\train.py ^
  configs\pointpillars\pointpillars_radar_vod.py ^
  --work-dir work_dirs\verify_pointpillars_radar_vod ^
  --cfg-options train_dataloader.batch_size=1 train_dataloader.num_workers=0 ^
  train_dataloader.persistent_workers=False val_dataloader.num_workers=0 ^
  val_dataloader.persistent_workers=False train_cfg.max_epochs=1 ^
  train_cfg.val_interval=1000
```

This verification path already advanced into real training iterations after the
repair.

## 8. Testing

```powershell
cd /d D:\文章\try\3ddetection
test_local.bat configs\pointpillars\pointpillars_radar_vod.py <checkpoint_path> work_dirs\pointpillars_radar_vod_test
```

Equivalent direct command:

```powershell
D:\codex_envs\3ddet_env\Scripts\python.exe local_preflight.py ^
  --config configs\pointpillars\pointpillars_radar_vod.py ^
  --dataset-root D:\文章\try\VOD ^
  --checkpoint <checkpoint_path> ^
  --mode test

D:\codex_envs\3ddet_env\Scripts\python.exe tools\test.py ^
  configs\pointpillars\pointpillars_radar_vod.py ^
  <checkpoint_path> ^
  --work-dir work_dirs\pointpillars_radar_vod_test
```

For the repaired local PointPillars config, `tools\test.py` now uses the
validation split with `KittiMetric` so the local test command directly outputs
metrics.

## 9. Common errors and fixes

### FileNotFoundError for `radar_infos_train.pkl`

Cause:

- Old configs still pointed to Linux dataset roots such as
  `/root/.../radar_infos_train.pkl`.
- Local VOD info files had not been generated yet.

Fix:

- Generate the local `infos/radar_5frames/*.pkl` files.
- Run training through `tools/train.py`, which now injects local paths.

### Validation crashed on `nvvm.dll` / `libNVVM`

Cause:

- KITTI/VOD validation imports `mmdet3d/evaluation/functional/kitti_utils/eval.py`.
- That path computes BEV and 3D IoU through `rotate_iou.py`.
- The original `rotate_iou.py` hard-wired a `numba.cuda` kernel at import
  time, which forced Windows validation to load `nvvm.dll` even though the
  main training loop itself was already running.

Fix:

- The local evaluation path now defaults to `mmcv.ops.box_iou_rotated`.
- If `mmcv` rotated IoU is unavailable, it can fall back to a pure Python
  implementation instead of crashing on NVVM.
- `local_preflight.py` reports the selected backend before any long job starts.

### Windows cannot follow VOD Linux symlinks

Cause:

- `radar` and `radar_5frames` contain Linux-style symbolic links for shared
  folders like `ImageSets`, `image_2`, and `label_2`.

Fix:

- The local radar converter now reads the real shared assets from
  `view_of_delft_PUBLIC/lidar` and writes accessible relative paths into the
  generated info files.

### Empty `Tr_imu_to_velo` in calibration files

Cause:

- Some VOD calib files leave `Tr_imu_to_velo` empty.

Fix:

- The converter now falls back to an identity-style 3x4 matrix when the field
  is missing or malformed.

### BEVFusion custom extension import error

Status:

- Not required for the repaired PointPillars training path.
- Some BEVFusion configs may still require separately built custom CUDA/C++
  ops before they can run on Windows.

## 10. Directory guide

- `configs/pointpillars/`
  Main radar and radar-camera configs, including
  `pointpillars_radar_vod.py` and `pointpillars_radar_vod_train_only.py`.
- `configs/bevfusion/`
  Radar-camera fusion configs with local placeholder paths.
- `tools/create_data.py`
  Unified dataset preparation entry.
- `tools/dataset_converters/radar_converter.py`
  VOD radar info generator with Windows-aware path handling.
- `tools/train.py`
  Training entry with runtime local placeholder replacement.
- `tools/test.py`
  Testing entry with runtime local placeholder replacement.
- `local_preflight.py`
  Startup checker for local Python/CUDA/dataset/evaluator readiness.
- `local_paths.py`
  Local path resolver used by runtime scripts.
- `MANUAL.md`
  Main local usage manual.
- `docs/archive/`
  Archived historical notes and one-off optimization records.

## 11. Archived documents

The following scattered root-level notes were moved into
`docs/archive/legacy_notes/`:

- `README_性能优化说明.md`
- `optimization_log.md`
- `optimization_plan.md`
- `performance_audit_report.md`
- `speed_benchmark_before_after.md`

The `performance/` and `environment_setup/` folders remain available as
supporting material, but `MANUAL.md` is now the recommended entry point.
