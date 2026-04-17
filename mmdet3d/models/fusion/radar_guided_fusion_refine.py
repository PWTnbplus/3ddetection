from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RadarGuidedFusionRefineCBAM(nn.Module):
    """Lightweight radar-guided refinement after image/radar BEV fusion.

    This block borrows the idea, not the full design, from recent fusion works:
    CVFusion motivates an explicit refine stage after fusion; RICCARDO
    motivates using radar as a spatial prior; RaCFormer motivates emphasizing
    foreground/instance-related high-response regions instead of relying on
    only global average attention. The implementation here stays small and
    suitable for the current one-stage PointPillars image/radar BEV pipeline.
    """

    def __init__(self,
                 channels: int = 384,
                 reduction: int = 16,
                 spatial_kernel_size: int = 7,
                 radar_hidden_channels: int = 128,
                 refine_scale_init: float = 0.1,
                 return_refine_stats: bool = False) -> None:
        super().__init__()
        if spatial_kernel_size % 2 == 0:
            raise ValueError('spatial_kernel_size must be odd.')
        hidden_channels = max(channels // reduction, 1)
        padding = spatial_kernel_size // 2

        self.channels = channels
        self.return_refine_stats = return_refine_stats

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_attn = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel_size,
            padding=padding,
            bias=False)
        self.radar_prior = nn.Sequential(
            nn.Conv2d(
                channels,
                radar_hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(radar_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(radar_hidden_channels, 1, kernel_size=1),
        )

        # A small learnable residual scale keeps early training close to the
        # original RL-fusion detector and lets the model grow refinement only
        # when detection loss supports it.
        self.refine_scale = nn.Parameter(
            torch.tensor(float(refine_scale_init), dtype=torch.float32))
        self.last_refine_stats: Dict[str, Tensor] = {}

    def forward(self,
                fused_feat: Tensor,
                radar_feat: Tensor,
                return_refine_stats: Optional[bool] = None):
        if fused_feat.shape != radar_feat.shape:
            raise ValueError(
                'fused_feat and radar_feat must have the same shape, got '
                f'{tuple(fused_feat.shape)} and {tuple(radar_feat.shape)}.')
        if fused_feat.shape[1] != self.channels:
            raise ValueError(
                f'Expected {self.channels} channels, got {fused_feat.shape[1]}.')

        avg_pool = F.adaptive_avg_pool2d(fused_feat, 1)
        max_pool = F.adaptive_max_pool2d(fused_feat, 1)
        channel_attn = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))

        avg_map = fused_feat.mean(dim=1, keepdim=True)
        max_map = fused_feat.amax(dim=1, keepdim=True)
        cbam_spatial = torch.sigmoid(
            self.spatial_attn(torch.cat([avg_map, max_map], dim=1)))
        radar_prior = torch.sigmoid(self.radar_prior(radar_feat))
        spatial_attn = cbam_spatial * radar_prior

        refine_delta = fused_feat * channel_attn * spatial_attn
        refined_feat = fused_feat + self.refine_scale.to(
            dtype=fused_feat.dtype) * refine_delta

        self.last_refine_stats = self._build_stats(channel_attn, cbam_spatial,
                                                   radar_prior, spatial_attn)
        if return_refine_stats is None:
            return_refine_stats = self.return_refine_stats
        if return_refine_stats:
            return refined_feat, self.last_refine_stats
        return refined_feat

    def get_refine_stats(self) -> Dict[str, Tensor]:
        return self.last_refine_stats

    def _build_stats(self, channel_attn: Tensor, cbam_spatial: Tensor,
                     radar_prior: Tensor,
                     spatial_attn: Tensor) -> Dict[str, Tensor]:
        return dict(
            refine_channel_mean=channel_attn.detach().mean(),
            refine_cbam_spatial_mean=cbam_spatial.detach().mean(),
            refine_radar_prior_mean=radar_prior.detach().mean(),
            refine_spatial_mean=spatial_attn.detach().mean(),
            refine_scale=self.refine_scale.detach())
