from typing import Dict, Optional, Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .image_point_voxelnet_rl import ImagePointVoxelNetRL
from .voxelnet import VoxelNet


@MODELS.register_module()
class ImagePointVoxelNetCARQ(ImagePointVoxelNetRL):
    """PointPillars image-radar detector with CARQ-Align proposal refinement."""

    def __init__(self, *args, proposal_refiner: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proposal_refiner = MODELS.build(proposal_refiner) if proposal_refiner else None
        self.last_carq_feats: Dict[str, Tensor] = {}

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'get_aux_losses'):
            losses.update(self.fusion.get_aux_losses())
        if self.proposal_refiner is not None:
            losses.update(self.proposal_refiner.loss(self.last_carq_feats, batch_data_samples))
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        if self.proposal_refiner is not None:
            results_list = self.proposal_refiner.predict(self.last_carq_feats, results_list)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        radar_bev_feats = VoxelNet.extract_feat(self, batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs', None))
        if img_feats is None or self.fusion is None:
            self.last_carq_feats = self._pack_single_feat(radar_bev_feats)
            return radar_bev_feats
        fused_feats = self.fusion(radar_bev_feats, img_feats, batch_data_samples)
        self.last_carq_feats = self._pack_carq_feats(radar_bev_feats, fused_feats)
        return fused_feats

    def _pack_carq_feats(self, radar_bev_feats, fused_feats) -> Dict[str, Tensor]:
        bev_cache = self.fusion.get_bev_features() if hasattr(self.fusion, 'get_bev_features') else {}
        radar_bev = self._first_4d(bev_cache.get('radar_bev', radar_bev_feats))
        image_bev = self._first_4d(bev_cache.get('image_bev', None))
        fused_bev = self._first_4d(bev_cache.get('fused_bev', fused_feats))
        if image_bev is None:
            image_bev = fused_bev
        return dict(radar_bev=radar_bev, image_bev=image_bev, fused_bev=fused_bev)

    def _pack_single_feat(self, feats) -> Dict[str, Tensor]:
        feat = self._first_4d(feats)
        return dict(radar_bev=feat, image_bev=feat, fused_bev=feat)

    def _first_4d(self, feats) -> Optional[Tensor]:
        if feats is None:
            return None
        if isinstance(feats, Tensor):
            return feats
        for feat in feats:
            if isinstance(feat, Tensor) and feat.dim() == 4:
                return feat
        return None
