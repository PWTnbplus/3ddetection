import re
import zlib
from functools import lru_cache
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_TOKEN_PATTERN = re.compile(r'\w+')


@lru_cache(maxsize=65536)
def _hashed_token_values(text: str,
                         hash_dim: int) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    buckets = {}
    for token in _TOKEN_PATTERN.findall(text.lower()):
        token_hash = zlib.crc32(token.encode('utf-8'))
        index = token_hash % hash_dim
        buckets[index] = buckets.get(index, 0.0) + (1.0 if token_hash & 1 else
                                                    -1.0)
    if not buckets:
        return (), ()
    items = sorted(buckets.items())
    indices, values = zip(*items)
    return indices, values


@lru_cache(maxsize=65536)
def _hashed_text_bytes(text: str, hash_dim: int) -> bytes:
    vector = np.zeros(hash_dim, dtype=np.float32)
    indices, values = _hashed_token_values(text, hash_dim)
    if indices:
        vector[np.fromiter(indices, dtype=np.int64)] = np.asarray(
            values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
    return vector.tobytes()


def hash_text_to_numpy(text: str,
                       hash_dim: int,
                       dtype=np.float32) -> np.ndarray:
    vector = np.frombuffer(
        _hashed_text_bytes(text, hash_dim), dtype=np.float32).copy()
    np_dtype = np.dtype(dtype)
    if np_dtype != np.float32:
        vector = vector.astype(np_dtype, copy=False)
    return vector


def build_hashed_text_array(texts: Sequence[str],
                            hash_dim: int,
                            dtype=np.float32) -> np.ndarray:
    if not texts:
        return np.zeros((0, hash_dim), dtype=dtype)
    unique_vectors = {}
    rows = []
    for text in texts:
        vector = unique_vectors.get(text)
        if vector is None:
            vector = hash_text_to_numpy(text, hash_dim, dtype=np.float32)
            unique_vectors[text] = vector
        rows.append(vector)
    return np.asarray(rows, dtype=dtype)


def build_hashed_text_tensor(texts: Iterable[str],
                             hash_dim: int,
                             device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
    texts = list(texts)
    if not texts:
        return torch.zeros((0, hash_dim), device=device, dtype=dtype)
    text_array = build_hashed_text_array(texts, hash_dim, dtype=np.float32)
    return torch.from_numpy(text_array).to(device=device, dtype=dtype)


def _coerce_text_embedding(text_embed,
                           hash_dim: int,
                           dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(text_embed):
        array = text_embed.detach().float().flatten().cpu().numpy()
    else:
        array = np.asarray(text_embed, dtype=np.float32).reshape(-1)
    if array.shape[0] < hash_dim:
        padded = np.zeros(hash_dim, dtype=np.float32)
        padded[:array.shape[0]] = array
        array = padded
    else:
        array = array[:hash_dim]
    if np.dtype(dtype) != np.float32:
        array = array.astype(dtype, copy=False)
    return array


def build_sample_text_features(batch_data_samples,
                               hash_dim: int,
                               device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor:
    batch_size = len(batch_data_samples) if batch_data_samples else 0
    if batch_size == 0:
        return torch.zeros((0, hash_dim), device=device, dtype=dtype)

    text_array = np.zeros((batch_size, hash_dim), dtype=np.float32)
    missing_rows = []
    missing_texts = []

    for row, sample in enumerate(batch_data_samples):
        metainfo = sample.metainfo
        text_embed = metainfo.get('text_embedding',
                                  metainfo.get('text_feat', None))
        if text_embed is None:
            missing_rows.append(row)
            missing_texts.append(str(metainfo.get('text', '')))
            continue
        text_array[row] = _coerce_text_embedding(text_embed, hash_dim)

    if missing_rows:
        text_array[np.asarray(missing_rows, dtype=np.int64)] = (
            build_hashed_text_array(missing_texts, hash_dim, dtype=np.float32))

    text_feats = torch.from_numpy(text_array).to(device=device, dtype=dtype)
    return F.normalize(text_feats, dim=1)
