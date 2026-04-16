from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .image_point_voxelnet_rl import ImagePointVoxelNetRL
from .voxelnet import VoxelNet


@MODELS.register_module()
class ImagePointVoxelNetRLAlign(ImagePointVoxelNetRL):
    """RL image/radar detector with pre-fusion image BEV alignment.

    The RL gate state is computed from raw image/radar BEV features. The
    alignment module only changes the image BEV used by the final fusion.
    """

    def __init__(self,
                 *args,
                 align_module: Optional[dict] = None,
                 alignment_loss: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.align_module = MODELS.build(align_module) if align_module else None
        self.alignment_loss = (
            MODELS.build(alignment_loss) if alignment_loss else None)
        self.last_alignment_aux_losses = {}
        self.last_align_stats = {}

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'get_aux_losses'):
            losses.update(self.fusion.get_aux_losses())
        losses.update(self.last_alignment_aux_losses)
        return losses

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        self.last_alignment_aux_losses = {}
        self.last_align_stats = {}
        radar_bev_feats = VoxelNet.extract_feat(self, batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs', None))
        if img_feats is None or self.fusion is None or self.align_module is None:
            return radar_bev_feats
        return self._align_and_fuse(radar_bev_feats, img_feats,
                                    batch_data_samples)

    def _align_and_fuse(self, radar_bev_feats, img_feats,
                        batch_data_samples):
        fused_feats = []
        gate_stats = {}
        gate_aux_losses = {}
        align_aux_losses = {}
        align_stats = {}
        self.last_alignment_aux_losses = {}
        self.last_align_stats = {}
        self.fusion.last_gate_stats = {}
        self.fusion.last_aux_losses = {}

        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.fusion.bev_channels:
                fused_feats.append(radar_feat)
                continue

            raw_img_bev = self.fusion._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            aligned_result = self.align_module(
                raw_img_bev, radar_feat, return_align_stats=True)
            if isinstance(aligned_result, tuple):
                aligned_img_bev, align_stats = aligned_result
            else:
                aligned_img_bev = aligned_result

            # Gate uses raw features by design; fusion uses aligned image BEV.
            weights = self.fusion._predict_gate_weights(radar_feat, raw_img_bev)
            weighted_feat = (
                weights[:, 0:1, None, None] * aligned_img_bev +
                weights[:, 1:2, None, None] * radar_feat)
            fused_feats.append(self.fusion.bev_fusion(weighted_feat))

            gate_stats = self.fusion._build_gate_stats(weights)
            gate_aux_losses = self.fusion._build_aux_losses(weights)
            align_aux_losses = self._build_alignment_losses(
                aligned_img_bev, radar_feat, batch_data_samples)

        self.fusion.last_gate_stats = gate_stats
        self.fusion.last_aux_losses = gate_aux_losses
        self.last_alignment_aux_losses = align_aux_losses
        self.last_align_stats = align_stats
        if isinstance(radar_bev_feats, tuple):
            return tuple(fused_feats)
        return fused_feats

    def _build_alignment_losses(self, img_bev_aligned: Tensor,
                                radar_bev: Tensor,
                                batch_data_samples) -> Dict[str, Tensor]:
        if (not self.training) or self.alignment_loss is None:
            return {}
        fg_mask = self._build_gt_bev_mask(batch_data_samples, radar_bev)
        return dict(
            loss_bev_align=self.alignment_loss(
                img_bev_aligned, radar_bev, fg_mask))

    def _build_gt_bev_mask(self, batch_data_samples,
                           target_feat: Tensor) -> Optional[Tensor]:
        point_cloud_range = getattr(self.fusion, 'point_cloud_range', None)
        if not batch_data_samples or point_cloud_range is None:
            return None

        batch_size, _, height, width = target_feat.shape
        device = target_feat.device
        mask = target_feat.new_zeros((batch_size, 1, height, width))
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

        for batch_idx, data_sample in enumerate(batch_data_samples):
            gt_instances = getattr(data_sample, 'gt_instances_3d', None)
            if gt_instances is None or not hasattr(gt_instances, 'bboxes_3d'):
                continue
            boxes = gt_instances.bboxes_3d.tensor.to(device=device)
            if boxes.numel() == 0:
                continue
            for box in boxes:
                x, y = box[0], box[1]
                dx = box[3].abs().clamp_min(1e-3)
                dy = box[4].abs().clamp_min(1e-3)
                x0 = ((x - dx * 0.5 - x_min) / (x_max - x_min) * width).floor()
                x1 = ((x + dx * 0.5 - x_min) / (x_max - x_min) * width).ceil()
                y0 = ((y - dy * 0.5 - y_min) / (y_max - y_min) * height).floor()
                y1 = ((y + dy * 0.5 - y_min) / (y_max - y_min) * height).ceil()
                x0 = int(torch.clamp(x0, 0, width - 1).item())
                x1 = int(torch.clamp(x1, 1, width).item())
                y0 = int(torch.clamp(y0, 0, height - 1).item())
                y1 = int(torch.clamp(y1, 1, height).item())
                if x1 > x0 and y1 > y0:
                    mask[batch_idx, :, y0:y1, x0:x1] = 1
        return mask
