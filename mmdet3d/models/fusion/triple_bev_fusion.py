import math
import re
import zlib
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
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

        text_embeds = []
        texts = []
        for data_sample in batch_data_samples:
            metainfo = data_sample.metainfo
            text_embed = metainfo.get('text_embedding', None)
            if text_embed is None:
                text_embed = metainfo.get('text_feat', None)
            if text_embed is not None:
                text_embed = torch.as_tensor(
                    text_embed, device=device, dtype=dtype).flatten()
                if text_embed.numel() < self.text_hash_dim:
                    text_embed = F.pad(
                        text_embed, (0, self.text_hash_dim - text_embed.numel()))
                text_embeds.append(text_embed[:self.text_hash_dim])
                texts.append(None)
            else:
                text_embeds.append(None)
                texts.append(str(metainfo.get('text', '')))

        hash_feats = self._encode_texts(
            [text or '' for text in texts], device=device, dtype=dtype)
        for row, text_embed in enumerate(text_embeds):
            if text_embed is not None:
                hash_feats[row] = text_embed
        return F.normalize(hash_feats, dim=1)

    def _encode_texts(self, texts: List[str], device: torch.device,
                      dtype: torch.dtype) -> Tensor:
        text_feats = torch.zeros(
            (len(texts), self.text_hash_dim), device=device, dtype=dtype)
        for row, text in enumerate(texts):
            for token in re.findall(r'\w+', text.lower()):
                token_hash = zlib.crc32(token.encode('utf-8'))
                index = token_hash % self.text_hash_dim
                sign = 1.0 if token_hash & 1 else -1.0
                text_feats[row, index] += sign
        return text_feats / text_feats.norm(dim=1, keepdim=True).clamp_min(1.0)

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
        if not batch_inputs_dict or batch_inputs_dict.get('imgs', None) is None:
            return torch.zeros((batch_size, 2), device=device, dtype=dtype)
        imgs = batch_inputs_dict['imgs'].detach().to(device=device, dtype=dtype)
        dims = (1, 2, 3, 4) if imgs.dim() == 5 else (1, 2, 3)
        brightness = imgs.mean(dim=dims).sigmoid()
        variance = imgs.var(dim=dims, unbiased=False).clamp_min(0).sqrt()
        variance = (variance / (variance + 1.0)).clamp(0, 1)
        return torch.stack([brightness, variance], dim=1)

    def _point_stats(self, batch_inputs_dict, batch_size: int,
                     device: torch.device, dtype: torch.dtype) -> Tensor:
        if not batch_inputs_dict or batch_inputs_dict.get('points', None) is None:
            return torch.zeros((batch_size, 2), device=device, dtype=dtype)
        counts = []
        entropies = []
        for points in batch_inputs_dict['points']:
            points = points.detach().to(device=device, dtype=dtype)
            counts.append(torch.as_tensor(
                min(math.log1p(points.shape[0]) / 10.0, 1.0),
                device=device,
                dtype=dtype))
            entropies.append(self._point_xy_entropy(points, device, dtype))
        if len(counts) < batch_size:
            pad = batch_size - len(counts)
            counts.extend([torch.zeros((), device=device, dtype=dtype)] * pad)
            entropies.extend([torch.zeros((), device=device, dtype=dtype)] * pad)
        return torch.stack(
            [torch.stack(counts[:batch_size]),
             torch.stack(entropies[:batch_size])],
            dim=1)

    def _point_xy_entropy(self, points: Tensor, device: torch.device,
                          dtype: torch.dtype) -> Tensor:
        if points.numel() == 0 or points.shape[1] < 2:
            return torch.zeros((), device=device, dtype=dtype)
        if self.point_cloud_range is None:
            xy = points[:, :2]
            xy_min = xy.min(dim=0).values
            xy_max = xy.max(dim=0).values
        else:
            x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
            xy_min = torch.as_tensor([x_min, y_min], device=device, dtype=dtype)
            xy_max = torch.as_tensor([x_max, y_max], device=device, dtype=dtype)
        xy = (points[:, :2] - xy_min) / (xy_max - xy_min).clamp_min(1e-6)
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0) & (xy[:, 1] < 1)
        xy = xy[valid]
        if xy.numel() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        bins = 8
        xy_idx = (xy * bins).long().clamp(0, bins - 1)
        linear_idx = xy_idx[:, 1] * bins + xy_idx[:, 0]
        hist = torch.bincount(linear_idx, minlength=bins * bins).to(dtype)
        probs = hist / hist.sum().clamp_min(1.0)
        entropy = -(probs * (probs + 1e-6).log()).sum()
        return (entropy / math.log(bins * bins)).clamp(0, 1)

    def _text_confidence(self, batch_data_samples, batch_size: int,
                         device: torch.device, dtype: torch.dtype) -> Tensor:
        values = []
        if batch_data_samples:
            for data_sample in batch_data_samples:
                metainfo = data_sample.metainfo
                confidence = metainfo.get(
                    'text_confidence',
                    metainfo.get('text_conf', metainfo.get('llm_confidence', None)))
                if confidence is None:
                    confidence = 1.0 if metainfo.get('text', '') else 0.0
                values.append(float(confidence))
        if len(values) < batch_size:
            values.extend([0.0] * (batch_size - len(values)))
        return torch.as_tensor(
            values[:batch_size], device=device, dtype=dtype).clamp(0, 1)[:, None]

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
