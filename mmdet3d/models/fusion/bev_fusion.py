from typing import Sequence

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImageRadarBEVFusion(nn.Module):
    """Fuse image BEV-like features with radar BEV features.

    The image branch feature is resized to the radar BEV feature resolution,
    projected to BEV channels, and fused with radar BEV by concat + conv.
    """

    def __init__(self,
                 image_channels: int = 256,
                 bev_channels: int = 384,
                 image_dropout: float = 0.0) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.bev_channels = bev_channels
        self.image_bev_encoder = nn.Sequential(
            nn.Conv2d(image_channels, bev_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(image_dropout),
            nn.Conv2d(
                bev_channels,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(
                bev_channels * 2,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bev_channels,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, radar_bev_feats, img_feats: Sequence[Tensor]):
        fused_feats = []
        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue
            image_bev_feat = self._build_image_bev(img_feats, radar_feat)
            fused_feats.append(
                self.bev_fusion(torch.cat([radar_feat, image_bev_feat], dim=1)))

        if isinstance(radar_bev_feats, tuple):
            return tuple(fused_feats)
        return fused_feats

    def _select_image_feature(self, img_feats: Sequence[Tensor]) -> Tensor:
        for feat in img_feats:
            if feat.shape[1] == self.image_channels:
                return feat
        return img_feats[-1]

    def _build_image_bev(self, img_feats: Sequence[Tensor],
                         target_feat: Tensor) -> Tensor:
        img_feat = self._select_image_feature(img_feats)
        img_feat = torch.nn.functional.interpolate(
            img_feat,
            size=target_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return self.image_bev_encoder(img_feat).to(dtype=target_feat.dtype)
