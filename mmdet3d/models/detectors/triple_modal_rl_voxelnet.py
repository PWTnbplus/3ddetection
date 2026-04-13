from typing import Tuple

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox_overlaps_nearest_3d
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .text_voxelnet import ImagePointVoxelNet


@MODELS.register_module()
class TripleModalRLVoxelNet(ImagePointVoxelNet):
    """Image, radar and text VoxelNet with delayed-reward RL fusion."""

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses, predictions = self.bbox_head.loss_and_predict(
            x, batch_data_samples, **kwargs)
        if self.fusion is not None and hasattr(self.fusion, 'set_rl_reward'):
            task_loss = self._sum_task_losses(losses)
            iou_reward = self._build_iou_reward(predictions, batch_data_samples)
            self.fusion.set_rl_reward(
                task_loss, len(batch_data_samples), iou_reward=iou_reward)
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

    def _sum_task_losses(self, losses: dict) -> Tensor:
        task_loss = None
        for key, value in losses.items():
            if not key.startswith('loss'):
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if not torch.is_tensor(item):
                    continue
                item = item.mean()
                task_loss = item if task_loss is None else task_loss + item
        if task_loss is None:
            device = next(self.parameters()).device
            task_loss = torch.zeros((), device=device)
        return task_loss

    def _build_iou_reward(self, predictions, batch_data_samples) -> Tensor:
        rewards = []
        device = next(self.parameters()).device
        for pred_instances, data_sample in zip(predictions, batch_data_samples):
            pred_boxes = pred_instances.bboxes_3d.tensor
            gt_boxes = data_sample.gt_instances_3d.bboxes_3d.tensor
            if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                rewards.append(torch.zeros((), device=device))
                continue

            box_dim = min(pred_boxes.shape[-1], gt_boxes.shape[-1])
            pred_boxes = pred_boxes[:, :box_dim].detach()
            gt_boxes = gt_boxes[:, :box_dim].detach().to(pred_boxes.device)
            overlaps = bbox_overlaps_nearest_3d(
                pred_boxes, gt_boxes, coordinate='lidar')
            max_iou = overlaps.max(dim=1).values
            best_iou = max_iou.max()

            tier_reward = torch.where(
                best_iou >= 0.7,
                best_iou.new_tensor(3.0),
                torch.where(best_iou >= 0.5, best_iou.new_tensor(1.0),
                            best_iou.new_tensor(-1.0)))
            dense_reward = max_iou.mean() * 2.0 - 0.5
            rewards.append(tier_reward + dense_reward)

        if not rewards:
            return torch.zeros((), device=device)
        return torch.stack(rewards).mean()
