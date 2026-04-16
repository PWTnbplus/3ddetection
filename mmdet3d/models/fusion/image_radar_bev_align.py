from typing import Dict, Optional

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RadarGuidedBEVAlign(nn.Module):
    """Lightweight radar-guided pre-fusion image BEV alignment."""

    def __init__(self,
                 channels: int = 384,
                 hidden_channels: int = 128,
                 mode: str = 'residual_align',
                 detach_radar_guidance: bool = False,
                 return_align_stats: bool = False) -> None:
        super().__init__()
        if mode not in ('prior_only', 'residual_align'):
            raise ValueError(f'Unsupported align mode: {mode}')
        self.mode = mode
        self.detach_radar_guidance = detach_radar_guidance
        self.return_align_stats = return_align_stats

        self.radar_prior = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))
        self.delta_conv = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1))

        # Keep the default residual path close to identity at the start.
        nn.init.zeros_(self.delta_conv[-1].weight)
        nn.init.zeros_(self.delta_conv[-1].bias)

    def forward(self,
                img_bev: Tensor,
                radar_bev: Tensor,
                return_align_stats: Optional[bool] = None):
        radar_guidance = radar_bev.detach() if self.detach_radar_guidance else radar_bev
        prior = torch.sigmoid(self.radar_prior(radar_guidance))
        img_guided = img_bev * prior

        if self.mode == 'prior_only':
            aligned = img_guided
            delta = aligned - img_bev
        else:
            delta = self.delta_conv(torch.cat([img_guided, radar_guidance], dim=1))
            aligned = img_bev + delta

        if return_align_stats is None:
            return_align_stats = self.return_align_stats
        if return_align_stats:
            return aligned, self._build_stats(prior, delta)
        return aligned

    def _build_stats(self, prior: Tensor, delta: Tensor) -> Dict[str, Tensor]:
        return dict(
            align_prior_mean=prior.detach().mean(),
            align_delta_abs=delta.detach().abs().mean())
