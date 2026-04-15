from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from .ops import Voxelization


@MODELS.register_module()
class PointPillarsRadarBranch(nn.Module):
    """PointPillars-style BEV encoder for 7-D radar points."""

    def __init__(self,
                 voxelize_cfg: dict,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: dict) -> None:
        super().__init__()
        voxelize_cfg = voxelize_cfg.copy()
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce', False)
        self.voxel_layer = Voxelization(**voxelize_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

    @torch.no_grad()
    def voxelize(self, points: List[Tensor]):
        feats, coords, sizes = [], [], []
        for batch_idx, point in enumerate(points):
            ret = self.voxel_layer(point)
            if len(ret) == 3:
                feat, coord, num_points = ret
            else:
                feat, coord = ret
                num_points = None
            feats.append(feat)
            coords.append(F.pad(coord, (1, 0), mode='constant',
                                value=batch_idx))
            if num_points is not None:
                sizes.append(num_points)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if sizes:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        return feats, coords, sizes

    def forward(self, batch_inputs_dict: Dict[str, List[Tensor]]) -> Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast(device_type='cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = len(points)

        if feats.dim() == 3:
            # BEVFusion voxelization returns [batch, x, y, z], while
            # PointPillars encoders expect [batch, z, y, x].
            coords = coords[:, [0, 3, 2, 1]].contiguous()
            feats = self.voxel_encoder(feats, sizes, coords)

        x = self.middle_encoder(feats, coords, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x
