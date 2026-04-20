@echo off
setlocal

set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"

for %%I in ("%REPO_DIR%..\VOD") do set "DEFAULT_VOD_ROOT=%%~fI"
if not defined MMD3D_VOD_ROOT set "MMD3D_VOD_ROOT=%DEFAULT_VOD_ROOT%"
if not defined MMD3D_VOD_MODALITY set "MMD3D_VOD_MODALITY=radar_5frames"
if not defined PYTHONUTF8 set "PYTHONUTF8=1"
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"

"D:\codex_envs\3ddet_env\Scripts\python.exe" tools\create_data.py vod ^
  --root-path "%MMD3D_VOD_ROOT%" ^
  --out-dir "%MMD3D_VOD_ROOT%\view_of_delft_PUBLIC\infos\%MMD3D_VOD_MODALITY%" ^
  --extra-tag radar ^
  --modality "%MMD3D_VOD_MODALITY%" ^
  --workers 4

endlocal
