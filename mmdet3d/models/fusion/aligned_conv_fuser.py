from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CrossModalAlign(nn.Module):
    """Lightweight bidirectional alignment for image and point BEV features."""

    def __init__(self,
                 in_channels: Sequence[int],
                 align_channels: int,
                 gate_reduction: int = 4) -> None:
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError('CrossModalAlign expects image and point inputs.')
        hidden_channels = max(align_channels // gate_reduction, 16)
        self.img_proj = nn.Sequential(
            nn.Conv2d(in_channels[0], align_channels, 1, bias=False),
            nn.BatchNorm2d(align_channels),
            nn.ReLU(True))
        self.pts_proj = nn.Sequential(
            nn.Conv2d(in_channels[1], align_channels, 1, bias=False),
            nn.BatchNorm2d(align_channels),
            nn.ReLU(True))

        self.img_to_pts_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(align_channels, hidden_channels, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, align_channels, 1),
            nn.Sigmoid())
        self.pts_to_img_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(align_channels, hidden_channels, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, align_channels, 1),
            nn.Sigmoid())
        self.img_to_pts_spatial_gate = nn.Sequential(
            nn.Conv2d(align_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid())
        self.pts_to_img_spatial_gate = nn.Sequential(
            nn.Conv2d(align_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid())

    def forward(self, img_bev: torch.Tensor,
                pts_bev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_bev.shape[-2:] != pts_bev.shape[-2:]:
            img_bev = F.interpolate(
                img_bev,
                size=pts_bev.shape[-2:],
                mode='bilinear',
                align_corners=False)

        img_bev = self.img_proj(img_bev)
        pts_bev = self.pts_proj(pts_bev)

        img_to_pts_gate = (
            self.img_to_pts_channel_gate(img_bev) *
            self.img_to_pts_spatial_gate(img_bev))
        pts_to_img_gate = (
            self.pts_to_img_channel_gate(pts_bev) *
            self.pts_to_img_spatial_gate(pts_bev))

        pts_bev = pts_bev * (1.0 + img_to_pts_gate)
        img_bev = img_bev * (1.0 + pts_to_img_gate)
        return img_bev, pts_bev


@MODELS.register_module()
class BasicFusionNeck(nn.Module):
    """Fixed concat-conv fusion baseline before future RL fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, img_bev: torch.Tensor,
                pts_bev: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([img_bev, pts_bev], dim=1))


@MODELS.register_module()
class AlignedConvFuser(nn.Module):
    """Align image and point BEV features before fixed concat-conv fusion."""

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int,
                 align_channels: int = 256,
                 gate_reduction: int = 4) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.align_channels = align_channels
        self.align = CrossModalAlign(
            in_channels=in_channels,
            align_channels=align_channels,
            gate_reduction=gate_reduction)
        self.fusion = BasicFusionNeck(
            in_channels=align_channels, out_channels=out_channels)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError('AlignedConvFuser expects [img_bev, pts_bev].')
        img_bev_before_align = inputs[0]
        pts_bev_before_align = inputs[1]
        # future RL insertion point: img_bev_before_align
        # future RL insertion point: pts_bev_before_align

        img_bev_after_align, pts_bev_after_align = self.align(
            img_bev_before_align, pts_bev_before_align)
        # future RL insertion point: img_bev_after_align
        # future RL insertion point: pts_bev_after_align

        fused_bev = self.fusion(img_bev_after_align, pts_bev_after_align)
        # future RL insertion point: fused_bev
        return fused_bev
