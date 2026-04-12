from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImageRadarBEVFusion(nn.Module):
    """Fuse image BEV-like features with radar BEV features.

    The image branch feature is resized to the radar BEV feature resolution,
    projected to BEV channels, and fused with radar BEV by concat + conv.
    """

    def __init__(self,
                 image_channels: int = 256,
                 bev_channels: int = 384,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 height_samples: Sequence[float] = (-1.0, 0.0, 1.0),
                 image_dropout: float = 0.0) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.bev_channels = bev_channels
        self.point_cloud_range = point_cloud_range
        self.height_samples = height_samples
        self.image_bev_encoder = nn.Sequential(
            nn.Conv2d(image_channels, bev_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(image_dropout),
            nn.Conv2d(
                bev_channels,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(
                bev_channels * 2,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bev_channels,
                bev_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                radar_bev_feats,
                img_feats: Sequence[Tensor],
                batch_data_samples=None):
        fused_feats = []
        for radar_feat in radar_bev_feats:
            if radar_feat.shape[1] != self.bev_channels:
                fused_feats.append(radar_feat)
                continue
            image_bev_feat = self._build_image_bev(
                img_feats, radar_feat, batch_data_samples)
            fused_feats.append(
                self.bev_fusion(torch.cat([radar_feat, image_bev_feat], dim=1)))

        if isinstance(radar_bev_feats, tuple):
            return tuple(fused_feats)
        return fused_feats

    def _select_image_feature(self, img_feats: Sequence[Tensor]) -> Tensor:
        for feat in img_feats:
            if feat.shape[1] == self.image_channels:
                return feat
        return img_feats[-1]

    def _build_image_bev(self,
                         img_feats: Sequence[Tensor],
                         target_feat: Tensor,
                         batch_data_samples=None) -> Tensor:
        img_feat = self._select_image_feature(img_feats)
        if self.point_cloud_range is not None and batch_data_samples:
            projected_feat = self._project_image_to_bev(
                img_feat, target_feat, batch_data_samples)
            if projected_feat is not None:
                return self.image_bev_encoder(projected_feat).to(
                    dtype=target_feat.dtype)

        img_feat = torch.nn.functional.interpolate(
            img_feat,
            size=target_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return self.image_bev_encoder(img_feat).to(dtype=target_feat.dtype)

    def _project_image_to_bev(self, img_feat: Tensor, target_feat: Tensor,
                              batch_data_samples) -> Optional[Tensor]:
        if len(batch_data_samples) != target_feat.shape[0]:
            return None

        device = target_feat.device
        dtype = torch.float32
        sample_img_feat = img_feat.float()
        _, _, bev_h, bev_w = target_feat.shape
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        xs = torch.linspace(
            x_min, x_max, bev_w + 1, device=device, dtype=dtype)[:-1]
        ys = torch.linspace(
            y_min, y_max, bev_h + 1, device=device, dtype=dtype)[:-1]
        xs = xs + (x_max - x_min) / bev_w * 0.5
        ys = ys + (y_max - y_min) / bev_h * 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        zs = torch.as_tensor(self.height_samples, device=device, dtype=dtype)
        zs = zs[(zs >= z_min) & (zs <= z_max)]
        if zs.numel() == 0:
            zs = torch.as_tensor([(z_min + z_max) * 0.5],
                                 device=device,
                                 dtype=dtype)

        ones = torch.ones_like(grid_x)
        bev_samples = []
        valid_samples = []
        for z in zs:
            grid_z = torch.full_like(grid_x, z)
            bev_samples.append(
                torch.stack([grid_x, grid_y, grid_z, ones], dim=-1))

        points = torch.stack(bev_samples, dim=0).reshape(-1, 4)
        projected_feats = []
        for batch_idx, data_sample in enumerate(batch_data_samples):
            lidar2img = data_sample.metainfo.get('lidar2img', None)
            if lidar2img is None:
                return None
            if isinstance(lidar2img, (list, tuple)):
                lidar2img = lidar2img[0]

            lidar2img = torch.as_tensor(lidar2img, device=device, dtype=dtype)
            lidar2img = lidar2img[:3, :4]
            points_xyz = self._undo_3d_augmentation(
                points[:, :3], data_sample.metainfo)
            points_hom = torch.cat([points_xyz, points[:, 3:4]], dim=1)
            projected = points_hom @ lidar2img.t()
            depth = projected[:, 2].clamp(min=1e-5)
            pixel_x = projected[:, 0] / depth
            pixel_y = projected[:, 1] / depth

            input_shape = data_sample.metainfo.get(
                'batch_input_shape',
                data_sample.metainfo.get(
                    'pad_shape', data_sample.metainfo.get('img_shape', None)))
            if input_shape is None:
                return None
            img_shape_h, img_shape_w = input_shape[:2]
            norm_x = pixel_x / img_shape_w * 2 - 1
            norm_y = pixel_y / img_shape_h * 2 - 1
            sample_grid = torch.stack([norm_x, norm_y], dim=-1)
            sample_grid = sample_grid.view(1, zs.numel(), bev_h * bev_w, 2)
            sampled = torch.nn.functional.grid_sample(
                sample_img_feat[batch_idx:batch_idx + 1],
                sample_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampled = sampled.view(1, sample_img_feat.shape[1], zs.numel(),
                                   bev_h, bev_w)

            valid = ((pixel_x >= 0) & (pixel_x < img_shape_w) &
                     (pixel_y >= 0) & (pixel_y < img_shape_h) &
                     (projected[:, 2] > 1e-5))
            valid = valid.view(1, 1, zs.numel(), bev_h, bev_w).to(
                dtype=sampled.dtype)
            projected_feats.append((sampled * valid).sum(dim=2))
            valid_samples.append(valid.sum(dim=2).clamp_min(1.0))

        image_bev = torch.cat(projected_feats, dim=0)
        valid_count = torch.cat(valid_samples, dim=0)
        return image_bev / valid_count

    def _undo_3d_augmentation(self, points: Tensor, metainfo: dict) -> Tensor:
        flow = metainfo.get('transformation_3d_flow', [])
        if not flow:
            return points

        points = points.clone()
        for op in reversed(flow):
            if op == 'T' and 'pcd_trans' in metainfo:
                trans = torch.as_tensor(
                    metainfo['pcd_trans'],
                    device=points.device,
                    dtype=points.dtype)
                points = points - trans
            elif op == 'S' and 'pcd_scale_factor' in metainfo:
                points = points / float(metainfo['pcd_scale_factor'])
            elif op == 'R' and 'pcd_rotation' in metainfo:
                rotation = torch.as_tensor(
                    metainfo['pcd_rotation'],
                    device=points.device,
                    dtype=points.dtype)
                points = points @ torch.inverse(rotation)
            elif op == 'HF':
                points[:, 1] = -points[:, 1]
            elif op == 'VF':
                points[:, 0] = -points[:, 0]
        return points
