@echo off
setlocal

set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"

for %%I in ("%REPO_DIR%..\VOD") do set "DEFAULT_VOD_ROOT=%%~fI"
if not defined MMD3D_VOD_ROOT set "MMD3D_VOD_ROOT=%DEFAULT_VOD_ROOT%"
if not defined MMD3D_ROTATE_IOU_BACKEND set "MMD3D_ROTATE_IOU_BACKEND=mmcv"
if not defined PYTHONUTF8 set "PYTHONUTF8=1"
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"
set "PYTHON_EXE=D:\codex_envs\3ddet_env\Scripts\python.exe"

set "CONFIG=%~1"
set "CHECKPOINT=%~2"

if "%CONFIG%"=="" (
  echo Usage: test_local.bat ^<config^> ^<checkpoint^> [work_dir]
  exit /b 1
)

if "%CHECKPOINT%"=="" (
  echo Usage: test_local.bat ^<config^> ^<checkpoint^> [work_dir]
  exit /b 1
)

set "WORK_DIR=%~3"
if "%WORK_DIR%"=="" set "WORK_DIR=work_dirs\local_test"

"%PYTHON_EXE%" local_preflight.py --config "%CONFIG%" --dataset-root "%MMD3D_VOD_ROOT%" --checkpoint "%CHECKPOINT%" --mode test
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" tools\test.py "%CONFIG%" "%CHECKPOINT%" --work-dir "%WORK_DIR%"

endlocal
