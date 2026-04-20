import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.utils.text_hash import hash_text_to_numpy

from .radar_dataset import RadarDataset


@DATASETS.register_module()
class RadarTextDataset(RadarDataset):
    """Radar dataset variant that attaches one generated text per frame."""

    def __init__(self,
                 *args,
                 text_ann_file: str = '',
                 missing_text: str = '',
                 text_hash_dim: int = 512,
                 precompute_text_embedding: bool = True,
                 **kwargs) -> None:
        self.text_ann_file = text_ann_file
        self.missing_text = missing_text
        self.text_hash_dim = text_hash_dim
        self.precompute_text_embedding = precompute_text_embedding
        data_root = kwargs.get('data_root', None)
        self.text_map, self.text_embedding_map = self._load_text_map(
            text_ann_file, data_root)
        super().__init__(*args, **kwargs)

    def _load_text_map(self, text_ann_file: str,
                       data_root: Optional[str]) -> Tuple[Dict[str, str],
                                                          Dict[str, np.ndarray]]:
        if not text_ann_file:
            return {}, {}

        text_path = Path(text_ann_file)
        if not text_path.is_absolute() and data_root is not None:
            text_path = Path(data_root) / text_path

        with text_path.open('r', encoding='utf-8') as f:
            records = json.load(f)

        text_map = {}
        text_embedding_map = {}
        embedding_cache = {}
        for record in records:
            sample_id = str(record.get('sample_id', ''))
            text = record.get('text', '')
            text_map[sample_id] = text
            if self.precompute_text_embedding:
                text_embedding = embedding_cache.get(text)
                if text_embedding is None:
                    text_embedding = hash_text_to_numpy(
                        text, self.text_hash_dim, dtype=np.float32)
                    embedding_cache[text] = text_embedding
                text_embedding_map[sample_id] = text_embedding
        return text_map, text_embedding_map

    def parse_data_info(self, info: dict) -> dict:
        info = super().parse_data_info(info)
        sample_id = str(info.get('token', info.get('sample_idx', '')))
        info['text'] = self.text_map.get(sample_id, self.missing_text)
        text_embedding = self.text_embedding_map.get(sample_id, None)
        if text_embedding is not None:
            info['text_embedding'] = text_embedding
        return info
