from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@dataclass
class CARQProposals:
    boxes: Tensor
    labels: Tensor
    scores: Tensor
    batch_inds: Tensor
    target_boxes: Optional[Tensor] = None


def _mlp(in_channels: int, hidden_channels: int, out_channels: int,
         num_layers: int = 2) -> nn.Sequential:
    layers = []
    last = in_channels
    for _ in range(max(num_layers - 1, 0)):
        layers += [nn.Linear(last, hidden_channels), nn.LayerNorm(hidden_channels), nn.ReLU(inplace=True)]
        last = hidden_channels
    layers.append(nn.Linear(last, out_channels))
    return nn.Sequential(*layers)


@MODELS.register_module()
class InstanceCrossModalAlign(BaseModule):
    """BEV-only instance-level cross-modal alignment for CARQ-Align."""

    def __init__(self,
                 in_channels: int = 384,
                 embed_dims: int = 256,
                 point_cloud_range: Sequence[float] = (0, -25.6, -3, 51.2, 25.6, 2),
                 grid_size: int = 7,
                 enlarge_ratio: Sequence[float] = (2.0, 2.0, 1.5),
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size
        self.register_buffer('enlarge_ratio', torch.tensor(enlarge_ratio, dtype=torch.float32), persistent=False)
        self.fuse_proj = nn.Linear(in_channels, embed_dims)
        self.radar_proj = nn.Linear(in_channels, embed_dims)
        self.image_proj = nn.Linear(in_channels, embed_dims)
        self.query_mlp = _mlp(embed_dims + 10, embed_dims, embed_dims)
        self.offset_head = _mlp(embed_dims * 4, embed_dims, 3)
        self.token_fuse = _mlp(embed_dims * 4 + 3, embed_dims, embed_dims)
        self.consistency_head = _mlp(embed_dims * 3, embed_dims, 1)

    def forward(self, proposals: CARQProposals, fused_bev: Tensor,
                radar_bev: Tensor, image_bev: Tensor) -> Tuple[Tensor, Dict]:
        if proposals.boxes.numel() == 0:
            empty = fused_bev.new_zeros((0, self.embed_dims))
            return empty, dict(offset=fused_bev.new_zeros((0, 3)), consistency=fused_bev.new_zeros((0, 1)), quality_cue=fused_bev.new_zeros((0, 1)))

        grid = self._build_rotated_grid(proposals.boxes, proposals.labels, fused_bev.shape[-2:], fused_bev.device)
        fuse_roi = self._grid_sample(fused_bev, grid, proposals.batch_inds)
        radar_roi = self._grid_sample(radar_bev, grid, proposals.batch_inds)
        image_roi = self._grid_sample(image_bev, grid, proposals.batch_inds)

        fuse_tok = self.fuse_proj(fuse_roi.transpose(1, 2))
        radar_tok = self.radar_proj(radar_roi.transpose(1, 2))
        image_tok = self.image_proj(image_roi.transpose(1, 2))
        fuse_pool = fuse_tok.mean(dim=1)
        radar_pool = radar_tok.mean(dim=1)
        image_pool = image_tok.mean(dim=1)
        query = self.query_mlp(torch.cat([fuse_pool, self._encode_geometry(proposals)], dim=-1))

        radar_attn = self._attend(query, radar_tok)
        image_attn = self._attend(query, image_tok)
        offset = self.offset_head(torch.cat([query, fuse_pool, radar_attn, image_attn], dim=-1))
        aligned_grid = self._apply_local_offset(grid, offset)
        image_aligned = self._grid_sample(image_bev, aligned_grid, proposals.batch_inds)
        image_aligned_tok = self.image_proj(image_aligned.transpose(1, 2))
        image_aligned_attn = self._attend(query, image_aligned_tok)

        consistency = torch.cosine_similarity(radar_attn.float(), image_aligned_attn.float(), dim=-1).unsqueeze(-1).to(dtype=fused_bev.dtype)
        inst_token = self.token_fuse(torch.cat([query, fuse_pool, radar_attn, image_aligned_attn, offset], dim=-1))
        quality_cue = torch.sigmoid(self.consistency_head(torch.cat([query, radar_attn, image_aligned_attn], dim=-1)))
        return inst_token, dict(offset=offset, consistency=consistency, quality_cue=quality_cue)

    def _attend(self, query: Tensor, tokens: Tensor) -> Tensor:
        attn = torch.einsum('nd,ngd->ng', query.float(), tokens.float()) / (query.shape[-1] ** 0.5)
        attn = attn.softmax(dim=-1).to(dtype=tokens.dtype)
        return torch.einsum('ng,ngd->nd', attn, tokens)

    def _grid_sample(self, feat: Tensor, grid: Tensor, batch_inds: Tensor) -> Tensor:
        # Memory-light bilinear sampler for proposal ROIs. It gathers only the
        # N * G * G queried BEV locations instead of invoking grid_sample on a
        # full BEV map per proposal.
        num_props, grid_h, grid_w, _ = grid.shape
        batch_size, channels, height, width = feat.shape
        num_points = grid_h * grid_w
        if num_props == 0:
            return feat.new_zeros((0, channels, num_points))

        x = (grid[..., 0].clamp(-1, 1) + 1) * 0.5 * (width - 1)
        y = (grid[..., 1].clamp(-1, 1) + 1) * 0.5 * (height - 1)
        x0 = x.floor().long().clamp(0, width - 1)
        y0 = y.floor().long().clamp(0, height - 1)
        x1 = (x0 + 1).clamp(0, width - 1)
        y1 = (y0 + 1).clamp(0, height - 1)

        x0f = x0.to(dtype=feat.dtype)
        y0f = y0.to(dtype=feat.dtype)
        wx = (x.to(dtype=feat.dtype) - x0f).clamp(0, 1)
        wy = (y.to(dtype=feat.dtype) - y0f).clamp(0, 1)
        wa = ((1 - wx) * (1 - wy)).view(num_props, num_points)
        wb = ((1 - wx) * wy).view(num_props, num_points)
        wc = (wx * (1 - wy)).view(num_props, num_points)
        wd = (wx * wy).view(num_props, num_points)

        idx_a = (y0 * width + x0).view(num_props, num_points)
        idx_b = (y1 * width + x0).view(num_props, num_points)
        idx_c = (y0 * width + x1).view(num_props, num_points)
        idx_d = (y1 * width + x1).view(num_props, num_points)

        flat_feat = feat.flatten(2)
        outputs = feat.new_zeros((num_props, channels, num_points))
        for batch_idx in batch_inds.unique(sorted=True):
            mask = batch_inds == batch_idx
            src = flat_feat[int(batch_idx.item())]
            for out_idx, weight in ((idx_a, wa), (idx_b, wb), (idx_c, wc), (idx_d, wd)):
                gathered = src[:, out_idx[mask].reshape(-1)]
                gathered = gathered.view(channels, int(mask.sum().item()), num_points).permute(1, 0, 2)
                outputs[mask] += gathered * weight[mask].unsqueeze(1)
        return outputs
    def _build_rotated_grid(self, boxes: Tensor, labels: Tensor, feat_hw: Tuple[int, int], device: torch.device) -> Tensor:
        g = self.grid_size
        base_y, base_x = torch.meshgrid(torch.linspace(-0.5, 0.5, g, device=device), torch.linspace(-0.5, 0.5, g, device=device), indexing='ij')
        base = torch.stack([base_x.reshape(-1), base_y.reshape(-1)], dim=-1)
        ratios = self.enlarge_ratio.to(device=device)[labels.clamp(min=0, max=self.enlarge_ratio.numel() - 1)]
        dims = boxes[:, 3:5].abs().clamp_min(1e-2) * ratios[:, None]
        local = base[None] * dims[:, None]
        yaw = boxes[:, 6]
        cos_yaw = yaw.cos()
        sin_yaw = yaw.sin()
        rot_x = local[..., 0] * cos_yaw[:, None] - local[..., 1] * sin_yaw[:, None]
        rot_y = local[..., 0] * sin_yaw[:, None] + local[..., 1] * cos_yaw[:, None]
        xy = torch.stack([rot_x, rot_y], dim=-1) + boxes[:, None, :2]
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        norm_x = (xy[..., 0] - x_min) / max(x_max - x_min, 1e-6) * 2 - 1
        norm_y = (xy[..., 1] - y_min) / max(y_max - y_min, 1e-6) * 2 - 1
        return torch.stack([norm_x, norm_y], dim=-1).view(boxes.shape[0], g, g, 2)

    def _apply_local_offset(self, grid: Tensor, offset: Tensor) -> Tensor:
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        dx = offset[:, 0] / max(x_max - x_min, 1e-6) * 2
        dy = offset[:, 1] / max(y_max - y_min, 1e-6) * 2
        return grid + torch.stack([dx, dy], dim=-1)[:, None, None, :].to(dtype=grid.dtype)

    def _encode_geometry(self, proposals: CARQProposals) -> Tensor:
        boxes = proposals.boxes
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        center = boxes[:, :3].clone()
        center[:, 0] = (center[:, 0] - x_min) / max(x_max - x_min, 1e-6)
        center[:, 1] = (center[:, 1] - y_min) / max(y_max - y_min, 1e-6)
        center[:, 2] = (center[:, 2] - z_min) / max(z_max - z_min, 1e-6)
        dims = boxes[:, 3:6].abs().clamp_min(1e-3).log()
        yaw = torch.stack([boxes[:, 6].sin(), boxes[:, 6].cos()], dim=-1)
        label_norm = proposals.labels.to(dtype=boxes.dtype).unsqueeze(-1) / 2.0
        return torch.cat([center, dims, yaw, proposals.scores[:, None], label_norm], dim=-1)


@MODELS.register_module()
class ClassAwareRefineHead(BaseModule):
    """Shared trunk plus class-specific geometry experts."""

    def __init__(self, embed_dims: int = 256, num_classes: int = 3,
                 class_dim_masks: Optional[Sequence[Sequence[float]]] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.shared = _mlp(embed_dims + num_classes, embed_dims, embed_dims)
        self.experts = nn.ModuleList([_mlp(embed_dims, embed_dims, 7) for _ in range(num_classes)])
        if class_dim_masks is None:
            class_dim_masks = [[1.0, 1.0, 0.5, 0.2, 0.2, 0.5, 0.1], [1.0, 1.0, 0.7, 0.5, 0.8, 0.5, 0.8], [1.0, 1.0, 1.5, 1.5, 1.5, 1.2, 1.2]]
        self.register_buffer('class_dim_masks', torch.tensor(class_dim_masks, dtype=torch.float32), persistent=False)

    def forward(self, inst_token: Tensor, labels: Tensor) -> Tensor:
        if inst_token.numel() == 0:
            return inst_token.new_zeros((0, 7))
        safe_labels = labels.clamp(min=0, max=self.num_classes - 1)
        one_hot = F.one_hot(safe_labels, num_classes=self.num_classes).to(dtype=inst_token.dtype)
        shared = self.shared(torch.cat([inst_token, one_hot], dim=-1))
        expert_outs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        delta = expert_outs[torch.arange(labels.numel(), device=labels.device), safe_labels]
        return delta * self.class_dim_masks.to(delta.device)[safe_labels].to(delta.dtype)


@MODELS.register_module()
class GeometryReliabilityGate(BaseModule):
    """Proposal quality gate that controls box refinement strength."""

    def __init__(self, embed_dims: int = 256, vector_gate: bool = True,
                 init_bias: float = -2.0, init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.vector_gate = vector_gate
        out_dims = 7 if vector_gate else 1
        self.gate_mlp = _mlp(embed_dims + 11, embed_dims, out_dims)
        nn.init.constant_(self.gate_mlp[-1].bias, init_bias)

    def forward(self, inst_token: Tensor, proposals: CARQProposals,
                align_info: Dict, delta: Tensor) -> Tensor:
        if inst_token.numel() == 0:
            out_dims = 7 if self.vector_gate else 1
            return inst_token.new_zeros((0, out_dims))
        geom = torch.cat([proposals.boxes[:, :7], proposals.scores[:, None], align_info['consistency'], align_info['quality_cue'], delta.detach().abs().mean(dim=-1, keepdim=True)], dim=-1)
        return torch.sigmoid(self.gate_mlp(torch.cat([inst_token, geom], dim=-1)))


@MODELS.register_module()
class RadarCameraProposalRefiner(BaseModule):
    """CARQ-Align proposal refiner with GT-jitter training and top-k inference."""

    def __init__(self, align_module: dict, refine_head: dict, quality_gate: dict,
                 train_jitter_std: Sequence[float] = (0.25, 0.25, 0.12, 0.08, 0.08, 0.05, 0.12),
                 train_jitter_times: int = 2, max_train_proposals: int = 192,
                 topk_test: Sequence[int] = (50, 50, 80), score_thr: float = 0.05,
                 loss_refine_weight: float = 1.0, loss_gate_weight: float = 0.2,
                 loss_align_weight: float = 0.05, loss_stability_weight: float = 0.02,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.align_module = MODELS.build(align_module)
        self.refine_head = MODELS.build(refine_head)
        self.quality_gate = MODELS.build(quality_gate)
        self.train_jitter_std = train_jitter_std
        self.train_jitter_times = train_jitter_times
        self.max_train_proposals = max_train_proposals
        self.topk_test = topk_test
        self.score_thr = score_thr
        self.loss_refine_weight = loss_refine_weight
        self.loss_gate_weight = loss_gate_weight
        self.loss_align_weight = loss_align_weight
        self.loss_stability_weight = loss_stability_weight

    def loss(self, feats: Dict[str, Tensor], batch_data_samples) -> Dict:
        proposals = self._sample_gt_jitter_proposals(batch_data_samples, feats['fused_bev'])
        if proposals.boxes.numel() == 0:
            return {}
        inst_token, align_info = self.align_module(proposals, feats['fused_bev'], feats['radar_bev'], feats['image_bev'])
        delta = self.refine_head(inst_token, proposals.labels)
        alpha = self.quality_gate(inst_token, proposals, align_info, delta)
        refined = self.apply_delta(proposals.boxes, delta, alpha)
        target_delta = self.encode_delta(proposals.boxes, proposals.target_boxes)
        loss_refine = F.smooth_l1_loss(delta, target_delta, reduction='none').mean(dim=-1)
        loss_refine = self._class_balanced_mean(loss_refine, proposals.labels)
        refined_error = self.encode_delta(refined, proposals.target_boxes).abs().mean(dim=-1)
        coarse_error = target_delta.abs().mean(dim=-1)
        gate_target = (refined_error.detach() < coarse_error.detach()).to(dtype=alpha.dtype)
        gate_pred = alpha.mean(dim=-1) if alpha.shape[-1] > 1 else alpha[:, 0]
        loss_gate = F.binary_cross_entropy(gate_pred, gate_target)
        loss_align = (1.0 - align_info['consistency']).clamp_min(0).mean()
        pedcyc = proposals.labels != 2
        loss_stability = (alpha[pedcyc] * delta[pedcyc]).abs().mean() if pedcyc.any() else delta.sum() * 0
        return dict(loss_carq_refine=self.loss_refine_weight * loss_refine,
                    loss_carq_gate=self.loss_gate_weight * loss_gate,
                    loss_carq_align=self.loss_align_weight * loss_align,
                    loss_carq_pedcyc_stability=self.loss_stability_weight * loss_stability)

    def predict(self, feats: Dict[str, Tensor], results_list: List) -> List:
        proposals, select_inds = self._select_test_proposals(results_list)
        if proposals.boxes.numel() == 0:
            return results_list
        inst_token, align_info = self.align_module(proposals, feats['fused_bev'], feats['radar_bev'], feats['image_bev'])
        delta = self.refine_head(inst_token, proposals.labels)
        alpha = self.quality_gate(inst_token, proposals, align_info, delta)
        refined = self.apply_delta(proposals.boxes, delta, alpha)
        offset = 0
        for batch_idx, result in enumerate(results_list):
            inds = select_inds[batch_idx]
            num = inds.numel()
            if num == 0:
                continue
            boxes = result.bboxes_3d.tensor.clone()
            boxes[inds, :7] = refined[offset:offset + num]
            result.bboxes_3d = result.bboxes_3d.new_box(boxes)
            offset += num
        return results_list

    def apply_delta(self, boxes: Tensor, delta: Tensor, alpha: Tensor) -> Tensor:
        if alpha.shape[-1] == 1:
            alpha = alpha.expand_as(delta)
        update = alpha * delta
        refined = boxes.clone()
        diag = torch.linalg.norm(boxes[:, 3:5].abs().clamp_min(1e-3), dim=-1)
        refined[:, 0] = boxes[:, 0] + update[:, 0] * diag
        refined[:, 1] = boxes[:, 1] + update[:, 1] * diag
        refined[:, 2] = boxes[:, 2] + update[:, 2] * boxes[:, 5].abs().clamp_min(1e-3)
        refined[:, 3:6] = boxes[:, 3:6].abs().clamp_min(1e-3) * torch.exp(update[:, 3:6].clamp(min=-1.0, max=1.0))
        refined[:, 6] = boxes[:, 6] + update[:, 6]
        return refined

    def encode_delta(self, boxes: Tensor, target: Tensor) -> Tensor:
        diag = torch.linalg.norm(boxes[:, 3:5].abs().clamp_min(1e-3), dim=-1)
        delta = boxes.new_zeros((boxes.shape[0], 7))
        delta[:, 0] = (target[:, 0] - boxes[:, 0]) / diag.clamp_min(1e-3)
        delta[:, 1] = (target[:, 1] - boxes[:, 1]) / diag.clamp_min(1e-3)
        delta[:, 2] = (target[:, 2] - boxes[:, 2]) / boxes[:, 5].abs().clamp_min(1e-3)
        delta[:, 3:6] = (target[:, 3:6].abs().clamp_min(1e-3) / boxes[:, 3:6].abs().clamp_min(1e-3)).log()
        delta[:, 6] = torch.atan2(torch.sin(target[:, 6] - boxes[:, 6]), torch.cos(target[:, 6] - boxes[:, 6]))
        return delta

    def _sample_gt_jitter_proposals(self, batch_data_samples, ref_feat: Tensor) -> CARQProposals:
        boxes_list, labels_list, batch_inds_list, target_list = [], [], [], []
        device = ref_feat.device
        std = ref_feat.new_tensor(self.train_jitter_std)
        for batch_idx, data_sample in enumerate(batch_data_samples):
            gt_instances = getattr(data_sample, 'gt_instances_3d', None)
            if gt_instances is None or not hasattr(gt_instances, 'bboxes_3d'):
                continue
            gt_boxes = gt_instances.bboxes_3d.tensor.to(device=device)
            gt_labels = gt_instances.labels_3d.to(device=device).long()
            if gt_boxes.numel() == 0:
                continue
            repeat_boxes = gt_boxes[:, :7].repeat_interleave(self.train_jitter_times, dim=0)
            repeat_labels = gt_labels.repeat_interleave(self.train_jitter_times, dim=0)
            noise = torch.randn_like(repeat_boxes) * std
            jitter = repeat_boxes.clone()
            diag = torch.linalg.norm(jitter[:, 3:5].abs().clamp_min(1e-3), dim=-1)
            jitter[:, 0] += noise[:, 0] * diag
            jitter[:, 1] += noise[:, 1] * diag
            jitter[:, 2] += noise[:, 2] * jitter[:, 5].abs().clamp_min(1e-3)
            jitter[:, 3:6] = jitter[:, 3:6].abs().clamp_min(1e-3) * torch.exp(noise[:, 3:6])
            jitter[:, 6] += noise[:, 6]
            boxes_list.append(jitter)
            labels_list.append(repeat_labels)
            batch_inds_list.append(torch.full((jitter.shape[0],), batch_idx, device=device, dtype=torch.long))
            target_list.append(repeat_boxes)
        if not boxes_list:
            empty = ref_feat.new_zeros((0, 7))
            empty_long = torch.zeros((0,), device=device, dtype=torch.long)
            return CARQProposals(empty, empty_long, empty.new_zeros((0,)), empty_long, empty)
        boxes = torch.cat(boxes_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        batch_inds = torch.cat(batch_inds_list, dim=0)
        target = torch.cat(target_list, dim=0)
        if boxes.shape[0] > self.max_train_proposals:
            perm = torch.randperm(boxes.shape[0], device=device)[:self.max_train_proposals]
            boxes, labels, batch_inds, target = boxes[perm], labels[perm], batch_inds[perm], target[perm]
        return CARQProposals(boxes, labels, boxes.new_ones(boxes.shape[0]), batch_inds, target)

    def _select_test_proposals(self, results_list: List) -> Tuple[CARQProposals, List[Tensor]]:
        boxes_list, labels_list, scores_list, batch_inds_list, select_inds = [], [], [], [], []
        device = None
        for batch_idx, result in enumerate(results_list):
            boxes = result.bboxes_3d.tensor
            scores = result.scores_3d
            labels = result.labels_3d.long()
            device = boxes.device
            keep_all = []
            for cls_idx, topk in enumerate(self.topk_test):
                cls_keep = torch.where((labels == cls_idx) & (scores >= self.score_thr))[0]
                if cls_keep.numel() > topk:
                    cls_keep = cls_keep[scores[cls_keep].topk(topk).indices]
                keep_all.append(cls_keep)
            keep = torch.cat(keep_all) if keep_all else labels.new_zeros((0,))
            select_inds.append(keep)
            if keep.numel() == 0:
                continue
            boxes_list.append(boxes[keep, :7])
            labels_list.append(labels[keep])
            scores_list.append(scores[keep])
            batch_inds_list.append(torch.full((keep.numel(),), batch_idx, device=boxes.device, dtype=torch.long))
        if not boxes_list:
            device = device or torch.device('cpu')
            empty = torch.zeros((0, 7), device=device)
            empty_long = torch.zeros((0,), device=device, dtype=torch.long)
            return CARQProposals(empty, empty_long, empty.new_zeros((0,)), empty_long), select_inds
        return CARQProposals(torch.cat(boxes_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(batch_inds_list, dim=0)), select_inds

    def _class_balanced_mean(self, values: Tensor, labels: Tensor) -> Tensor:
        parts = [values[labels == cls].mean() for cls in labels.unique() if (labels == cls).any()]
        return torch.stack(parts).mean() if parts else values.mean()



