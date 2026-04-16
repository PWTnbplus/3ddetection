from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ForegroundAlignmentLoss(nn.Module):
    """Foreground-only consistency loss for aligned image/radar BEV features."""

    def __init__(self,
                 loss_weight: float = 0.05,
                 use_cosine: bool = True,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.use_cosine = use_cosine
        self.eps = eps

    def forward(self,
                img_bev_aligned: Tensor,
                radar_bev: Tensor,
                fg_mask: Optional[Tensor] = None) -> Tensor:
        if self.loss_weight <= 0:
            return img_bev_aligned.sum() * 0

        if fg_mask is None:
            fg_mask = img_bev_aligned.new_ones(
                (img_bev_aligned.shape[0], 1, img_bev_aligned.shape[2],
                 img_bev_aligned.shape[3]))
        fg_mask = fg_mask.to(device=img_bev_aligned.device,
                             dtype=img_bev_aligned.dtype)
        if fg_mask.shape[-2:] != img_bev_aligned.shape[-2:]:
            fg_mask = F.interpolate(
                fg_mask, size=img_bev_aligned.shape[-2:], mode='nearest')

        valid = fg_mask > 0.5
        if valid.sum() == 0:
            return img_bev_aligned.sum() * 0

        if self.use_cosine:
            img_norm = F.normalize(img_bev_aligned.float(), dim=1, eps=self.eps)
            radar_norm = F.normalize(radar_bev.float(), dim=1, eps=self.eps)
            cosine = (img_norm * radar_norm).sum(dim=1, keepdim=True)
            loss = (1 - cosine)[valid].mean()
        else:
            loss = F.l1_loss(
                img_bev_aligned.float()[valid.expand_as(img_bev_aligned)],
                radar_bev.float()[valid.expand_as(radar_bev)])
        return loss * self.loss_weight
