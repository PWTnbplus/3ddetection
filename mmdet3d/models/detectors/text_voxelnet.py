import re
import zlib
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .voxelnet import VoxelNet


@MODELS.register_module()
class ImagePointVoxelNet(VoxelNet):
    """VoxelNet with image-BEV and point-BEV two-branch fusion.

    The image branch produces an image BEV-like feature map by projecting image
    FPN features to the radar BEV feature resolution, then fuses it with the
    radar BEV feature before the 3D detection head.
    """

    def __init__(self,
                 *args,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 fusion: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_backbone = MODELS.build(img_backbone) if img_backbone else None
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.fusion = MODELS.build(fusion) if fusion else None

    @property
    def with_img_backbone(self) -> bool:
        return self.img_backbone is not None

    @property
    def with_img_neck(self) -> bool:
        return self.img_neck is not None

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        return self.bbox_head.loss(x, batch_data_samples, **kwargs)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[Tensor]]:
        x = self.extract_feat(batch_inputs_dict, data_samples)
        return self.bbox_head.forward(x)

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        radar_bev_feats = super().extract_feat(batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs', None))
        if img_feats is None or self.fusion is None:
            return radar_bev_feats

        return self.fusion(radar_bev_feats, img_feats, batch_data_samples)

    def extract_img_feat(self,
                         imgs: Optional[Tensor]) -> Optional[Sequence[Tensor]]:
        if not self.with_img_backbone or imgs is None:
            return None

        num_views = 1
        if imgs.dim() == 5:
            batch_size, num_views, channels, height, width = imgs.size()
            imgs = imgs.view(batch_size * num_views, channels, height, width)
        else:
            batch_size = imgs.size(0)

        img_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, Tensor):
            img_feats = [img_feats]

        if num_views > 1:
            img_feats = [
                feat.view(batch_size, num_views, feat.size(1), feat.size(2),
                          feat.size(3)).mean(dim=1)
                for feat in img_feats
            ]
        return img_feats



@MODELS.register_module()
class TextVoxelNet(VoxelNet):
    """VoxelNet with a lightweight generated-text fusion branch.

    The text encoder is intentionally dependency-free: it hashes tokens into a
    fixed bag-of-words vector and learns a projection onto the BEV feature map.
    This keeps the first text-modality baseline easy to run on the same server.
    """

    def __init__(self,
                 *args,
                 text_hash_dim: int = 512,
                 text_channels: int = 384,
                 text_dropout: float = 0.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_hash_dim = text_hash_dim
        self.text_channels = text_channels
        self.channel_text_proj = nn.Sequential(
            nn.LayerNorm(text_hash_dim),
            nn.Linear(text_hash_dim, text_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(text_dropout),
            nn.Linear(text_channels, text_channels),
        )
        self.scalar_text_proj = nn.Sequential(
            nn.LayerNorm(text_hash_dim),
            nn.Linear(text_hash_dim, 1),
        )

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat_with_text(batch_inputs_dict, batch_data_samples)
        return self.bbox_head.loss(x, batch_data_samples, **kwargs)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat_with_text(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[Tensor]]:
        x = self.extract_feat_with_text(batch_inputs_dict, data_samples)
        return self.bbox_head.forward(x)

    def extract_feat_with_text(self, batch_inputs_dict: dict,
                               batch_data_samples: OptSampleList):
        feats = super().extract_feat(batch_inputs_dict)
        if not batch_data_samples:
            return feats

        device = feats[0].device
        dtype = feats[0].dtype
        texts = [sample.metainfo.get('text', '') for sample in batch_data_samples]
        text_feats = self._encode_texts(texts, device=device, dtype=dtype)
        channel_bias = self.channel_text_proj(text_feats)
        scalar_bias = self.scalar_text_proj(text_feats)
        return self._fuse_text(feats, channel_bias, scalar_bias)

    def _encode_texts(self, texts: List[str], device: torch.device,
                      dtype: torch.dtype) -> Tensor:
        text_feats = torch.zeros(
            (len(texts), self.text_hash_dim), device=device, dtype=dtype)

        for row, text in enumerate(texts):
            tokens = re.findall(r'[a-z0-9]+', text.lower())
            for token in tokens:
                token_hash = zlib.crc32(token.encode('utf-8'))
                index = token_hash % self.text_hash_dim
                sign = 1.0 if token_hash & 1 else -1.0
                text_feats[row, index] += sign

        norm = text_feats.norm(dim=1, keepdim=True).clamp_min(1.0)
        return text_feats / norm

    def _fuse_text(self, feats, channel_bias: Tensor, scalar_bias: Tensor):
        fused_feats = []
        for feat in feats:
            if feat.shape[1] == self.text_channels:
                bias = channel_bias[:, :, None, None]
            else:
                bias = scalar_bias[:, :, None, None]
            fused_feats.append(feat + bias)
        return fused_feats if isinstance(feats, list) else tuple(fused_feats)
