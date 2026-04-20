from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils.text_hash import build_sample_text_features
from .fast_utils import build_image_stats, build_point_stats, build_text_confidence
from .bev_fusion import ImageRadarBEVFusion


@MODELS.register_module()
class TripleModalBEVFusion(ImageRadarBEVFusion):
    """Weighted image, radar and text BEV fusion."""

    def __init__(self,
                 image_channels: int = 256,
                 bev_channels: int = 384,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 height_samples: Sequence[float] = (-1.0, 0.0, 1.0),
                 image_dropout: float = 0.0,
                 text_hash_dim: int = 512,
                 text_dropout: float = 0.0,
                 policy_hidden_channels: int = 64,
                 policy_temperature: float = 1.0,
                 entropy_loss_weight: float = 0.0,
                 diversity_loss_weight: float = 0.0) -> None:
        super().__init__(
            image_channels=image_channels,
            bev_channels=bev_channels,
            point_cloud_range=point_cloud_range,
            height_samples=height_samples,
            image_dropout=image_dropout)
        self.text_hash_dim = text_hash_dim
        self.policy_temperature = policy_temperature
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.text_bev_encoder = nn.Sequential(
            nn.LayerNorm(text_hash_dim),
            nn.Linear(text_hash_dim, bev_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(text_dropout),
            nn.Linear(bev_channels, bev_channels),
        )
        self.radar_text_proj = nn.Linear(bev_channels, text_hash_dim)
        self.policy_net = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, policy_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(policy_hidden_channels, 3),
        )
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        self.last_weights = None
        self.last_aux_losses = {}

    def forward(self,
                radar_bev_feats,
                img_feats: Sequence[Tensor],
                batch_data_samples=None,
                batch_inputs_dict: Optional[dict] = None):
        fused_feats = []
        aux_losses = {}
        self.last_weights = None
        self.last_aux_losses = {}

        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue

            text_feats = self._build_text_feats(
                batch_data_samples, radar_feat.device, radar_feat.dtype,
                radar_feat.shape[0])
            image_bev_feat = self._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            text_bev_feat = self._build_text_bev(text_feats, radar_feat)
            weights = self._predict_weights(
                radar_feat, text_feats, batch_data_samples, batch_inputs_dict)

            weighted_feat = (weights[:, 0:1, None, None] * radar_feat +
                             weights[:, 1:2, None, None] * image_bev_feat +
                             weights[:, 2:3, None, None] * text_bev_feat)
            fused_feats.append(self.bev_fusion(weighted_feat))
            self.last_weights = weights.detach()
            aux_losses = self._build_aux_losses(weights)

        self.last_aux_losses = aux_losses
        if isinstance(radar_bev_feats, tuple):
            return tuple(fused_feats)
        return fused_feats

    def get_aux_losses(self) -> dict:
        return self.last_aux_losses

    def _build_text_feats(self, batch_data_samples, device: torch.device,
                          dtype: torch.dtype, batch_size: int) -> Tensor:
        if not batch_data_samples:
            return torch.zeros(
                (batch_size, self.text_hash_dim), device=device, dtype=dtype)
        return build_sample_text_features(batch_data_samples,
                                          self.text_hash_dim, device, dtype)

    def _build_text_bev(self, text_feats: Tensor, target_feat: Tensor) -> Tensor:
        text_channels = self.text_bev_encoder(text_feats)
        return text_channels[:, :, None, None].expand_as(target_feat).to(
            dtype=target_feat.dtype)

    def _predict_weights(self, radar_feat: Tensor, text_feats: Tensor,
                         batch_data_samples, batch_inputs_dict) -> Tensor:
        state = self._build_policy_state(
            radar_feat, text_feats, batch_data_samples, batch_inputs_dict)
        logits = self.policy_net(state.detach())
        temperature = max(self.policy_temperature, 1e-6)
        return torch.softmax(logits / temperature, dim=1)

    def _build_policy_state(self, radar_feat: Tensor, text_feats: Tensor,
                            batch_data_samples, batch_inputs_dict) -> Tensor:
        batch_size = radar_feat.shape[0]
        device = radar_feat.device
        dtype = radar_feat.dtype
        image_stats = self._image_stats(batch_inputs_dict, batch_size, device,
                                        dtype)
        point_stats = self._point_stats(batch_inputs_dict, batch_size, device,
                                       dtype)
        text_conf = self._text_confidence(batch_data_samples, batch_size, device,
                                         dtype)
        radar_global = radar_feat.mean(dim=(2, 3))
        radar_text = F.normalize(self.radar_text_proj(radar_global.float()), dim=1)
        text_radar_cos = F.cosine_similarity(
            radar_text, text_feats.float(), dim=1).to(dtype)[:, None]
        prev_loss = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        return torch.cat(
            [image_stats, point_stats, text_conf, text_radar_cos, prev_loss],
            dim=1)

    def _image_stats(self, batch_inputs_dict, batch_size: int,
                     device: torch.device, dtype: torch.dtype) -> Tensor:
        imgs = None if not batch_inputs_dict else batch_inputs_dict.get(
            'imgs', None)
        return build_image_stats(imgs, batch_size, device, dtype)

    def _point_stats(self, batch_inputs_dict, batch_size: int,
                     device: torch.device, dtype: torch.dtype) -> Tensor:
        points = None if not batch_inputs_dict else batch_inputs_dict.get(
            'points', None)
        return build_point_stats(points, batch_size, device, dtype,
                                 self.point_cloud_range)

    def _text_confidence(self, batch_data_samples, batch_size: int,
                         device: torch.device, dtype: torch.dtype) -> Tensor:
        return build_text_confidence(batch_data_samples, batch_size, device,
                                     dtype)

    def _build_aux_losses(self, weights: Tensor) -> dict:
        losses = {}
        if self.entropy_loss_weight > 0:
            entropy = -(weights * (weights + 1e-6).log()).sum(dim=1).mean()
            losses['loss_fusion_entropy'] = -self.entropy_loss_weight * entropy
        if self.diversity_loss_weight > 0:
            target = weights.new_full(weights.shape, 1.0 / weights.shape[1])
            losses['loss_fusion_diversity'] = (
                self.diversity_loss_weight * F.mse_loss(weights, target))
        return losses
