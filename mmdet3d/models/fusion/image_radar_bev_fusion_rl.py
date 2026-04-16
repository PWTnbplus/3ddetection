from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from .bev_fusion import ImageRadarBEVFusion


@MODELS.register_module()
class ImageRadarBEVFusionRL(ImageRadarBEVFusion):
    """Gate-based image/radar BEV fusion.

    The default forward contract is compatible with ImagePointVoxelNet: it
    returns only fused BEV features. Set return_gate_stats=True explicitly to
    receive ``(fused_feats, gate_stats)`` for debugging.
    """

    def __init__(self,
                 image_channels: int = 256,
                 bev_channels: int = 384,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 height_samples: Sequence[float] = (-1.0, 0.0, 1.0),
                 image_dropout: float = 0.0,
                 gate_hidden_channels: int = 128,
                 gate_temperature: float = 1.0,
                 gate_entropy_weight: float = 0.0,
                 return_gate_stats: bool = False) -> None:
        super().__init__(
            image_channels=image_channels,
            bev_channels=bev_channels,
            point_cloud_range=point_cloud_range,
            height_samples=height_samples,
            image_dropout=image_dropout)
        self.gate_temperature = gate_temperature
        self.gate_entropy_weight = gate_entropy_weight
        self.return_gate_stats = return_gate_stats
        self.gate_net = nn.Sequential(
            nn.Linear(bev_channels * 4, gate_hidden_channels),
            nn.LayerNorm(gate_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_channels, 2),
        )
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(
                bev_channels,
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
        self.last_gate_stats = {}
        self.last_aux_losses = {}

    def forward(self,
                radar_bev_feats,
                img_feats: Sequence[Tensor],
                batch_data_samples=None,
                return_gate_stats: Optional[bool] = None):
        fused_feats = []
        aux_losses = {}
        gate_stats = {}
        self.last_gate_stats = {}
        self.last_aux_losses = {}

        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue

            image_bev_feat = self._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            weights = self._predict_gate_weights(radar_feat, image_bev_feat)
            weighted_feat = (weights[:, 0:1, None, None] * image_bev_feat +
                             weights[:, 1:2, None, None] * radar_feat)
            fused_feats.append(self.bev_fusion(weighted_feat))

            gate_stats = self._build_gate_stats(weights)
            aux_losses = self._build_aux_losses(weights)

        self.last_gate_stats = gate_stats
        self.last_aux_losses = aux_losses
        if isinstance(radar_bev_feats, tuple):
            fused_feats = tuple(fused_feats)

        if return_gate_stats is None:
            return_gate_stats = self.return_gate_stats
        if return_gate_stats:
            return fused_feats, gate_stats
        return fused_feats

    def get_gate_stats(self) -> Dict[str, Tensor]:
        return self.last_gate_stats

    def get_aux_losses(self) -> Dict[str, Tensor]:
        return self.last_aux_losses

    def _predict_gate_weights(self, radar_feat: Tensor,
                              image_bev_feat: Tensor) -> Tensor:
        radar_pool = radar_feat.mean(dim=(2, 3))
        image_pool = image_bev_feat.mean(dim=(2, 3))
        state = torch.cat([
            image_pool,
            radar_pool,
            torch.abs(image_pool - radar_pool),
            image_pool * radar_pool,
        ],
                          dim=1)
        logits = self.gate_net(state.float())
        temperature = max(self.gate_temperature, 1e-6)
        return F.softmax(logits / temperature, dim=1).to(dtype=radar_feat.dtype)

    def _build_gate_stats(self, weights: Tensor) -> Dict[str, Tensor]:
        return dict(
            mean_w_img=weights[:, 0].detach().mean(),
            mean_w_radar=weights[:, 1].detach().mean())

    def _build_aux_losses(self, weights: Tensor) -> Dict[str, Tensor]:
        if (not self.training) or self.gate_entropy_weight <= 0:
            return {}
        entropy = -(weights * (weights + 1e-6).log()).sum(dim=1).mean()
        return dict(loss_gate_entropy=-self.gate_entropy_weight * entropy)
