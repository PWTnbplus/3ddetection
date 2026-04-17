from typing import Dict, Optional, Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .image_point_voxelnet_rl import ImagePointVoxelNetRL
from .voxelnet import VoxelNet


@MODELS.register_module()
class ImagePointVoxelNetRLRefine(ImagePointVoxelNetRL):
    """RL image/radar detector with post-fusion radar-guided refinement.

    The detector keeps the original RL gate state unchanged: gate weights are
    still predicted from raw image/radar BEV features inside the fusion module.
    The new refine block is inserted only after RL fusion and before bbox_head.
    """

    def __init__(self,
                 *args,
                 refine_block: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refine_block = MODELS.build(refine_block) if refine_block else None
        self.last_refine_stats: Dict[str, Tensor] = {}

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'get_aux_losses'):
            losses.update(self.fusion.get_aux_losses())
        return losses

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        radar_bev_feats = VoxelNet.extract_feat(self, batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs', None))
        if img_feats is None or self.fusion is None:
            return radar_bev_feats

        fused_feats = self.fusion(radar_bev_feats, img_feats,
                                  batch_data_samples)
        return self._refine_fused_feats(fused_feats, radar_bev_feats)

    def get_refine_stats(self) -> Dict[str, Tensor]:
        return self.last_refine_stats

    def _refine_fused_feats(self, fused_feats, radar_bev_feats):
        self.last_refine_stats = {}
        if self.refine_block is None:
            return fused_feats

        refined_feats = []
        for fused_feat, radar_feat in zip(fused_feats, radar_bev_feats):
            if fused_feat.shape != radar_feat.shape:
                refined_feats.append(fused_feat)
                continue
            if fused_feat.shape[1] != self.refine_block.channels:
                refined_feats.append(fused_feat)
                continue
            refined_feats.append(self.refine_block(fused_feat, radar_feat))

        if hasattr(self.refine_block, 'get_refine_stats'):
            self.last_refine_stats = self.refine_block.get_refine_stats()

        if isinstance(fused_feats, tuple):
            return tuple(refined_feats)
        return refined_feats
