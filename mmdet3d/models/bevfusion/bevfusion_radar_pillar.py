from copy import deepcopy
from typing import Optional

import numpy as np
from torch import Tensor

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType, OptMultiConfig
from .bevfusion import BEVFusion


@MODELS.register_module()
class BEVFusionRadarPillar(BEVFusion):
    """BEVFusion variant with an isolated PointPillars radar branch."""

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        radar_branch: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        decoder_backbone: Optional[dict] = None,
        decoder_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ) -> None:
        Base3DDetector.__init__(
            self, data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.radar_branch = MODELS.build(radar_branch)
        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None
        self.decoder_backbone = MODELS.build(decoder_backbone)
        self.decoder_neck = MODELS.build(decoder_neck)
        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

    def extract_pts_feat(self, batch_inputs_dict) -> Tensor:
        return self.radar_branch(batch_inputs_dict)

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(
                imgs, deepcopy(points), lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix,
                batch_input_metas)
            features.append(img_feature)

        features.append(self.extract_pts_feat(batch_inputs_dict))

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.decoder_backbone(x)
        x = self.decoder_neck(x)
        return x
