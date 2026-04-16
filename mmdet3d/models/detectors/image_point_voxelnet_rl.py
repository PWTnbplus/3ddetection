from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import SampleList
from .text_voxelnet import ImagePointVoxelNet


@MODELS.register_module()
class ImagePointVoxelNetRL(ImagePointVoxelNet):
    """ImagePointVoxelNet variant that collects fusion auxiliary losses."""

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'get_aux_losses'):
            losses.update(self.fusion.get_aux_losses())
        return losses
