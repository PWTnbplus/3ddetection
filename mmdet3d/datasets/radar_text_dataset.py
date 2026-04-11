import json
from pathlib import Path
from typing import Dict, Optional

from mmdet3d.registry import DATASETS

from .radar_dataset import RadarDataset


@DATASETS.register_module()
class RadarTextDataset(RadarDataset):
    """Radar dataset variant that attaches one generated text per frame."""

    def __init__(self,
                 *args,
                 text_ann_file: str = '',
                 missing_text: str = '',
                 **kwargs) -> None:
        self.text_ann_file = text_ann_file
        self.missing_text = missing_text
        data_root = kwargs.get('data_root', None)
        self.text_map = self._load_text_map(text_ann_file, data_root)
        super().__init__(*args, **kwargs)

    def _load_text_map(self, text_ann_file: str,
                       data_root: Optional[str]) -> Dict[str, str]:
        if not text_ann_file:
            return {}

        text_path = Path(text_ann_file)
        if not text_path.is_absolute() and data_root is not None:
            text_path = Path(data_root) / text_path

        with text_path.open('r', encoding='utf-8') as f:
            records = json.load(f)

        return {
            str(record.get('sample_id', '')): record.get('text', '')
            for record in records
        }

    def parse_data_info(self, info: dict) -> dict:
        info = super().parse_data_info(info)
        sample_id = str(info.get('token', info.get('sample_idx', '')))
        info['text'] = self.text_map.get(sample_id, self.missing_text)
        return info
