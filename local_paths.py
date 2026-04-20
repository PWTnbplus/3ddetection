from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).absolute().parent
DEFAULT_WORK_ROOT = REPO_ROOT / 'work_dirs'
MODALITIES = ('lidar', 'radar', 'radar_3frames', 'radar_5frames')


def _normalize_path(path: Path) -> str:
    return path.absolute().as_posix()


def _coerce_vod_base(path: Path) -> Optional[Path]:
    if not path:
        return None
    if path.name in MODALITIES:
        return path.parent
    if path.name == 'view_of_delft_PUBLIC':
        return path
    if (path / 'view_of_delft_PUBLIC').exists():
        return path / 'view_of_delft_PUBLIC'
    if any((path / modality).exists() for modality in MODALITIES):
        return path
    return None


def resolve_vod_base(root: Optional[str] = None) -> Path:
    candidates = []
    if root:
        candidates.append(Path(root))

    env_root = os.getenv('MMD3D_VOD_ROOT')
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend([
        REPO_ROOT.parent / 'VOD',
        REPO_ROOT.parent / 'VOD' / 'view_of_delft_PUBLIC',
        Path('D:/VOD_ascii'),
    ])

    for candidate in candidates:
        base = _coerce_vod_base(candidate)
        if base is not None and base.exists():
            return base.absolute()

    raise FileNotFoundError(
        'Unable to locate the local VOD dataset. Set MMD3D_VOD_ROOT to the '
        'VOD root, view_of_delft_PUBLIC root, or a specific radar modality '
        'directory.')


def resolve_vod_modality_root(modality: str,
                              root: Optional[str] = None) -> Path:
    base = resolve_vod_base(root)
    modality_root = base / modality
    if not modality_root.exists():
        raise FileNotFoundError(
            f'Unable to locate modality "{modality}" under {base}.')
    return modality_root.absolute()


def resolve_info_root(modality: str, root: Optional[str] = None) -> Path:
    base = resolve_vod_base(root)
    info_root = base / 'infos' / modality
    info_root.mkdir(parents=True, exist_ok=True)
    return info_root.absolute()


def resolve_work_dir(work_name: str) -> Path:
    work_root = Path(os.getenv('MMD3D_WORK_ROOT', str(DEFAULT_WORK_ROOT)))
    return (work_root / work_name).absolute()


def build_runtime_paths(modality: str,
                        work_name: str,
                        root: Optional[str] = None) -> dict:
    base = resolve_vod_base(root)
    modality_root = resolve_vod_modality_root(modality, root)
    info_root = resolve_info_root(modality, root)
    return dict(
        vod_root=_normalize_path(base),
        modality_root=_normalize_path(modality_root),
        info_root=_normalize_path(info_root) + '/',
        work_dir=_normalize_path(resolve_work_dir(work_name)))


def info_file(modality: str,
              split: str,
              prefix: str = 'radar',
              root: Optional[str] = None) -> str:
    return _normalize_path(
        resolve_info_root(modality, root) / f'{prefix}_infos_{split}.pkl')


def text_file(modality: str,
              filename: str,
              root: Optional[str] = None) -> str:
    return _normalize_path(
        resolve_vod_modality_root(modality, root) / 'texts' / filename)


def placeholder_mapping(root: Optional[str] = None) -> dict[str, str]:
    base = resolve_vod_base(root)
    mapping = {
        '__VOD_BASE__': _normalize_path(base),
        '__INFO_ROOT_RADAR__': _normalize_path(resolve_info_root('radar', root)),
        '__INFO_ROOT_RADAR_3FRAMES__': _normalize_path(
            resolve_info_root('radar_3frames', root)),
        '__INFO_ROOT_RADAR_5FRAMES__': _normalize_path(
            resolve_info_root('radar_5frames', root)),
        '__TEXT_ROOT_RADAR__': _normalize_path(
            resolve_vod_modality_root('radar', root) / 'texts'),
        '__TEXT_ROOT_RADAR_3FRAMES__': _normalize_path(
            resolve_vod_modality_root('radar_3frames', root) / 'texts'),
        '__TEXT_ROOT_RADAR_5FRAMES__': _normalize_path(
            resolve_vod_modality_root('radar_5frames', root) / 'texts'),
    }
    return mapping


def apply_placeholder_mapping(obj, root: Optional[str] = None):
    mapping = placeholder_mapping(root)

    def replace(value):
        if isinstance(value, str):
            for token, resolved in mapping.items():
                value = value.replace(token, resolved)
            return value
        if hasattr(value, '_cfg_dict'):
            replace(value._cfg_dict)
            return value
        if isinstance(value, MutableMapping):
            for key in list(value.keys()):
                value[key] = replace(value[key])
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        return value

    return replace(obj)
