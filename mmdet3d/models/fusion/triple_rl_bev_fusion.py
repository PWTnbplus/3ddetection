from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils.text_hash import build_sample_text_features
from .fast_utils import build_image_stats, build_point_stats, build_text_confidence
from .bev_fusion import ImageRadarBEVFusion


class FeatureResizer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.norm(self.fc(x)))


class TextToBEVCrossAttention(nn.Module):
    def __init__(self, text_dim: int, bev_channels: int, nhead: int,
                 dropout: float) -> None:
        super().__init__()
        self.key_proj = FeatureResizer(text_dim, bev_channels, dropout)
        self.value_proj = FeatureResizer(text_dim, bev_channels, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=bev_channels,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False)

    def forward(self,
                bev_query: Tensor,
                text_feat: Tensor,
                text_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, channels, height, width = bev_query.shape
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if text_feat.shape[1] == 1 and text_mask is None:
            value = self.value_proj(text_feat[:, 0, :])
            embed_dim = self.attn.embed_dim
            v_weight = self.attn.in_proj_weight[2 * embed_dim:, :]
            if self.attn.in_proj_bias is not None:
                v_bias = self.attn.in_proj_bias[2 * embed_dim:]
            else:
                v_bias = None
            value = F.linear(value, v_weight, v_bias)
            value = self.attn.out_proj(value)
            return value[:, :, None, None].expand(batch_size, channels, height,
                                                  width)
        query = bev_query.flatten(2).permute(2, 0, 1)
        key = self.key_proj(text_feat).permute(1, 0, 2)
        value = self.value_proj(text_feat).permute(1, 0, 2)
        aligned = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=text_mask,
            need_weights=False)[0]
        return aligned.permute(1, 2, 0).reshape(
            batch_size, channels, height, width)


class StateActionGate(nn.Module):
    """Lightweight state-action modulation inspired by AEtherFold."""

    def __init__(self, state_dim: int, hidden_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, state: Tensor) -> Tensor:
        return hidden * self.dropout(self.gate(state))


class EnergyStabilityLayer(nn.Module):
    """Lyapunov-style energy damping layer inspired by KHS."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.eps = eps
        self.last_energy_loss = None

    def forward(self, x: Tensor) -> Tensor:
        gamma = torch.sigmoid(self.theta)
        out = gamma * x
        energy_in = x.detach().pow(2).mean()
        energy_out = out.pow(2).mean()
        self.last_energy_loss = F.relu(energy_out - energy_in - self.eps)
        return out


class CBAMRefinement(nn.Module):
    """Channel and spatial refinement for weighted multimodal BEV features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(avg_attn + max_attn)
        spatial_attn = torch.cat(
            [x.mean(dim=1, keepdim=True),
             x.max(dim=1, keepdim=True).values],
            dim=1)
        x = x * torch.sigmoid(self.spatial(spatial_attn))
        return self.refine(x)


class ActorCriticTriplePolicy(nn.Module):
    """Discrete trial controller for radar/image/text fusion weights."""

    def __init__(self,
                 bev_channels: int,
                 extra_state_dim: int = 8,
                 hidden_dim: int = 128,
                 action_space: Sequence[Tuple[float, float, float]] = (
                     (0.80, 0.10, 0.10),
                     (0.60, 0.30, 0.10),
                     (0.60, 0.10, 0.30),
                     (0.34, 0.33, 0.33),
                     (0.20, 0.60, 0.20),
                     (0.20, 0.20, 0.60),
                     (0.10, 0.45, 0.45),
                 ),
                 dropout: float = 0.1) -> None:
        super().__init__()
        actions = torch.tensor(action_space, dtype=torch.float32)
        actions = actions / actions.sum(dim=1, keepdim=True)
        self.register_buffer('actions', actions)
        self.backbone = nn.Sequential(
            nn.Linear(bev_channels * 3 + extra_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        self.state_action_gate = StateActionGate(
            bev_channels * 3 + extra_state_dim, hidden_dim, dropout)
        self.khs = EnergyStabilityLayer()
        self.actor = nn.Linear(hidden_dim, actions.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, radar_feat: Tensor, image_feat: Tensor,
                text_feat: Tensor, extra_state: Tensor,
                pooled_features: Optional[Tuple[Tensor, Tensor,
                                                Tensor]] = None):
        if pooled_features is None:
            radar_pool = radar_feat.mean(dim=(2, 3))
            image_pool = image_feat.mean(dim=(2, 3))
            text_pool = text_feat.mean(dim=(2, 3))
        else:
            radar_pool, image_pool, text_pool = pooled_features
        state = torch.cat([
            radar_pool,
            image_pool,
            text_pool,
            extra_state.detach()
        ],
                          dim=1)
        hidden = self.backbone(state)
        hidden = self.state_action_gate(hidden, state)
        hidden = self.khs(hidden)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        if self.training:
            dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
        else:
            action_idx = logits.argmax(dim=1)
            log_prob = None
            entropy = None

        weights = self.actions[action_idx].to(
            device=radar_feat.device, dtype=radar_feat.dtype).detach()
        return weights, dict(
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            action_idx=action_idx)


@MODELS.register_module()
class TripleModalRLBEVFusion(ImageRadarBEVFusion):
    """Actor-critic trial fusion for radar, image and text BEV features."""

    def __init__(self,
                 image_channels: int = 256,
                 bev_channels: int = 384,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 height_samples: Sequence[float] = (-1.0, 0.0, 1.0),
                 image_dropout: float = 0.0,
                 text_hash_dim: int = 512,
                 text_dropout: float = 0.05,
                 text_mask_prob: float = 0.0,
                 text_noise_std: float = 0.0,
                 nhead: int = 8,
                 controller_hidden_dim: int = 128,
                 action_space: Sequence[Tuple[float, float, float]] = (
                     (0.80, 0.10, 0.10),
                     (0.60, 0.30, 0.10),
                     (0.60, 0.10, 0.30),
                     (0.34, 0.33, 0.33),
                     (0.20, 0.60, 0.20),
                     (0.20, 0.20, 0.60),
                     (0.10, 0.45, 0.45),
                 ),
                 actor_weight: float = 1.0,
                 critic_weight: float = 0.5,
                 entropy_weight: float = 0.02,
                 khs_weight: float = 0.01,
                 consistency_reward_weight: float = 0.1,
                 iou_reward_weight: float = 0.5,
                 weight_jitter_weight: float = 0.05,
                 reward_scale: float = 1.0,
                 reward_clip: float = 2.0,
                 ema_momentum: float = 0.90,
                 warmup_iters: int = 200) -> None:
        super().__init__(
            image_channels=image_channels,
            bev_channels=bev_channels,
            point_cloud_range=point_cloud_range,
            height_samples=height_samples,
            image_dropout=image_dropout)
        self.text_hash_dim = text_hash_dim
        self.text_mask_prob = text_mask_prob
        self.text_noise_std = text_noise_std
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.khs_weight = khs_weight
        self.consistency_reward_weight = consistency_reward_weight
        self.iou_reward_weight = iou_reward_weight
        self.weight_jitter_weight = weight_jitter_weight
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.ema_momentum = ema_momentum
        self.warmup_iters = warmup_iters

        self.text_dropout = nn.Dropout(text_dropout)
        self.text_to_bev = TextToBEVCrossAttention(
            text_dim=text_hash_dim,
            bev_channels=bev_channels,
            nhead=nhead,
            dropout=text_dropout)
        self.rl_agent = ActorCriticTriplePolicy(
            bev_channels=bev_channels,
            hidden_dim=controller_hidden_dim,
            action_space=action_space,
            dropout=text_dropout)
        self.bev_fusion = CBAMRefinement(bev_channels)

        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('best_loss', torch.tensor(float('inf')))
        self.register_buffer('ema_initialized', torch.tensor(False))
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        self.pending_rl_info = None
        self.last_rl_info = None
        self.last_weights = None
        self.prev_weights = None
        self.last_action_idx = None
        self.last_consistency_reward = None

    def forward(self,
                radar_bev_feats,
                img_feats: Sequence[Tensor],
                batch_data_samples=None,
                batch_inputs_dict: Optional[dict] = None):
        fused_feats = []
        self.pending_rl_info = None
        self.last_rl_info = None
        self.last_action_idx = None
        self.last_weights = None
        self.last_consistency_reward = None

        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue

            image_bev_feat = self._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            text_feat = self._build_text_feats(
                batch_data_samples, radar_feat.device, radar_feat.dtype,
                radar_feat.shape[0])
            text_feat = self._regularize_text_feats(text_feat)
            text_tokens = self.text_dropout(text_feat).unsqueeze(1)
            text_bev_feat = self.text_to_bev(radar_feat, text_tokens)
            pooled_features = (
                radar_feat.mean(dim=(2, 3)),
                image_bev_feat.mean(dim=(2, 3)),
                text_bev_feat.mean(dim=(2, 3)),
            )
            extra_state = self._build_extra_state(
                radar_feat, image_bev_feat, text_bev_feat,
                batch_data_samples, batch_inputs_dict, pooled_features)
            self.last_consistency_reward = extra_state[:, 5:7].mean(
                dim=1, keepdim=True).detach().clamp(-1.0, 1.0)
            weights, rl_info = self.rl_agent(
                radar_feat,
                image_bev_feat,
                text_bev_feat,
                extra_state,
                pooled_features=pooled_features)
            weighted_feat = (weights[:, 0:1, None, None] * radar_feat +
                             weights[:, 1:2, None, None] * image_bev_feat +
                             weights[:, 2:3, None, None] * text_bev_feat)
            fused_feats.append(self.bev_fusion(weighted_feat) + radar_feat)

            if self.training:
                self.pending_rl_info = rl_info
                self.last_action_idx = rl_info['action_idx'].detach()
            self.last_weights = weights.detach()

        if isinstance(radar_bev_feats, tuple):
            return tuple(fused_feats)
        return fused_feats

    def _build_text_feats(self, batch_data_samples, device: torch.device,
                          dtype: torch.dtype, batch_size: int) -> Tensor:
        if not batch_data_samples:
            return torch.zeros(
                (batch_size, self.text_hash_dim), device=device, dtype=dtype)
        return build_sample_text_features(batch_data_samples,
                                          self.text_hash_dim, device, dtype)

    def _regularize_text_feats(self, text_feats: Tensor) -> Tensor:
        if not self.training:
            return text_feats

        if self.text_noise_std > 0:
            text_feats = (
                text_feats + torch.randn_like(text_feats) * self.text_noise_std)

        if self.text_mask_prob > 0:
            keep_mask = torch.rand(
                text_feats.shape[0],
                1,
                device=text_feats.device,
                dtype=text_feats.dtype) >= self.text_mask_prob
            text_feats = text_feats * keep_mask

        return F.normalize(text_feats, dim=1)

    def _build_extra_state(self, radar_feat: Tensor, image_feat: Tensor,
                           text_feat: Tensor, batch_data_samples,
                           batch_inputs_dict,
                           pooled_features: Optional[Tuple[Tensor, Tensor,
                                                           Tensor]] = None
                           ) -> Tensor:
        batch_size = radar_feat.shape[0]
        device = radar_feat.device
        dtype = radar_feat.dtype
        image_stats = self._image_stats(batch_inputs_dict, batch_size, device,
                                        dtype)
        point_stats = self._point_stats(batch_inputs_dict, batch_size, device,
                                        dtype)
        text_conf = self._text_confidence(batch_data_samples, batch_size, device,
                                          dtype)
        if pooled_features is None:
            radar_pool = radar_feat.mean(dim=(2, 3))
            image_pool = image_feat.mean(dim=(2, 3))
            text_pool = text_feat.mean(dim=(2, 3))
        else:
            radar_pool, image_pool, text_pool = pooled_features
        radar_image_cos = F.cosine_similarity(
            radar_pool.float(), image_pool.float(), dim=1).to(dtype)[:, None]
        radar_text_cos = F.cosine_similarity(
            radar_pool.float(), text_pool.float(), dim=1).to(dtype)[:, None]
        prev_loss = self.loss_ema.expand(batch_size, 1).to(
            device=device, dtype=dtype)
        return torch.cat([
            image_stats, point_stats, text_conf, radar_image_cos,
            radar_text_cos, prev_loss
        ],
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

    @torch.no_grad()
    def _build_reward(self, task_loss: Tensor,
                      iou_reward: Optional[Tensor] = None) -> Tensor:
        current_loss = task_loss.detach().float()
        if not bool(self.ema_initialized):
            self.loss_ema.copy_(current_loss)
            self.best_loss.copy_(current_loss)
            self.ema_initialized.fill_(True)

        improvement = self.loss_ema - current_loss
        breakthrough = torch.clamp(self.best_loss - current_loss, min=0.0)
        stability = -0.05 * torch.abs(current_loss - self.loss_ema)
        reward = improvement + 0.5 * breakthrough + stability
        if iou_reward is not None:
            reward = reward + self.iou_reward_weight * iou_reward.detach().float().mean()
        if self.last_consistency_reward is not None:
            reward = reward + self.consistency_reward_weight * (
                self.last_consistency_reward.mean())
        if (self.prev_weights is not None and self.last_weights is not None
                and self.prev_weights.shape == self.last_weights.shape):
            jitter = torch.abs(self.last_weights - self.prev_weights).mean()
            reward = reward - self.weight_jitter_weight * jitter
        reward = torch.clamp(
            reward * self.reward_scale,
            min=-self.reward_clip,
            max=self.reward_clip)

        self.loss_ema.mul_(self.ema_momentum).add_(
            current_loss * (1.0 - self.ema_momentum))
        self.best_loss.copy_(torch.minimum(self.best_loss, current_loss))
        self.global_step.add_(1)
        if self.global_step.item() < self.warmup_iters:
            reward = reward * 0.1
        return reward

    def set_rl_reward(self,
                      task_loss: Tensor,
                      batch_size: int,
                      iou_reward: Optional[Tensor] = None) -> None:
        if (not self.training) or self.pending_rl_info is None:
            return

        reward_scalar = self._build_reward(task_loss, iou_reward)
        reward = reward_scalar.view(1, 1).expand(batch_size, 1)
        rl_info = self.pending_rl_info
        advantage = reward - rl_info['value']
        actor_loss = -(
            rl_info['log_prob'].unsqueeze(-1) * advantage.detach()).mean()
        critic_loss = F.mse_loss(rl_info['value'], reward.detach())
        entropy_loss = -rl_info['entropy'].mean()
        self.last_rl_info = dict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            reward=reward.mean().detach())
        self.pending_rl_info = None
        if self.last_weights is not None:
            self.prev_weights = self.last_weights.detach()

    def get_aux_losses(self) -> Dict[str, Tensor]:
        if (not self.training) or self.last_rl_info is None:
            return {}
        khs_loss = self.rl_agent.khs.last_energy_loss
        if khs_loss is None:
            khs_loss = self.loss_ema.new_tensor(0.0)
        return dict(
            loss_rl_actor=self.actor_weight * self.last_rl_info['actor_loss'],
            loss_rl_critic=self.critic_weight * self.last_rl_info['critic_loss'],
            loss_rl_entropy=self.entropy_weight *
            self.last_rl_info['entropy_loss'],
            loss_rl_khs=self.khs_weight * khs_loss)

    def get_monitor_vars(self) -> Dict[str, Tensor]:
        logs = dict(
            rl_loss_ema=self.loss_ema.detach(),
            rl_best_loss=self.best_loss.detach(),
            rl_global_step=self.global_step.detach().float())
        if self.last_rl_info is not None:
            logs['rl_reward'] = self.last_rl_info['reward']
        if self.last_action_idx is not None:
            logs['rl_action_mean'] = self.last_action_idx.float().mean()
        if self.last_weights is not None:
            logs['rl_weight_rad'] = self.last_weights[:, 0].mean()
            logs['rl_weight_img'] = self.last_weights[:, 1].mean()
            logs['rl_weight_text'] = self.last_weights[:, 2].mean()
        if self.last_consistency_reward is not None:
            logs['rl_consistency_reward'] = self.last_consistency_reward.mean()
        return logs
