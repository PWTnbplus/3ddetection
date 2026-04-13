from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .text_voxelnet import ImagePointVoxelNet


@MODELS.register_module()
class TripleModalVoxelNet(ImagePointVoxelNet):
    """Image, radar and text VoxelNet with controller-weighted BEV fusion."""

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'get_aux_losses'):
            losses.update(self.fusion.get_aux_losses())
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[Tensor]:
        x = self.extract_feat(batch_inputs_dict, data_samples)
        return self.bbox_head.forward(x)

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        radar_bev_feats = super(ImagePointVoxelNet,
                                self).extract_feat(batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs', None))
        if img_feats is None or self.fusion is None:
            return radar_bev_feats
        return self.fusion(
            radar_bev_feats,
            img_feats,
            batch_data_samples,
            batch_inputs_dict=batch_inputs_dict)
