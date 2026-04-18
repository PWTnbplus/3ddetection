from typing import Dict, Optional, Sequence

from torch import Tensor

from mmdet3d.registry import MODELS
from .image_radar_bev_fusion_rl import ImageRadarBEVFusionRL


@MODELS.register_module()
class CARQImageRadarBEVFusion(ImageRadarBEVFusionRL):
    """Image-radar BEV fusion that exposes intermediate CARQ features."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_bev_features: Dict[str, Sequence[Tensor]] = {}

    def forward(self,
                radar_bev_feats,
                img_feats: Sequence[Tensor],
                batch_data_samples=None,
                return_gate_stats: Optional[bool] = None):
        fused_feats = []
        image_bev_feats = []
        radar_used_feats = []
        aux_losses = {}
        gate_stats = {}
        self.last_gate_stats = {}
        self.last_aux_losses = {}
        self.last_bev_features = {}

        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue

            image_bev_feat = self._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            weights = self._predict_gate_weights(radar_feat, image_bev_feat)
            weighted_feat = (weights[:, 0:1, None, None] * image_bev_feat +
                             weights[:, 1:2, None, None] * radar_feat)
            fused_feat = self.bev_fusion(weighted_feat)

            fused_feats.append(fused_feat)
            image_bev_feats.append(image_bev_feat)
            radar_used_feats.append(radar_feat)
            gate_stats = self._build_gate_stats(weights)
            aux_losses = self._build_aux_losses(weights)

        self.last_gate_stats = gate_stats
        self.last_aux_losses = aux_losses
        self.last_bev_features = dict(
            radar_bev=tuple(radar_used_feats),
            image_bev=tuple(image_bev_feats),
            fused_bev=tuple(fused_feats))

        if isinstance(radar_bev_feats, tuple):
            fused_feats = tuple(fused_feats)

        if return_gate_stats is None:
            return_gate_stats = self.return_gate_stats
        if return_gate_stats:
            return fused_feats, gate_stats
        return fused_feats

    def get_bev_features(self) -> Dict[str, Sequence[Tensor]]:
        return self.last_bev_features
