# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS).cuda()

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        if cfg.ALIGN.ALIGN:
            self.align = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    # template patch
    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        if cfg.ALIGN.ALIGN:
            zf = [self.align(zf[i]) for i in range(len(zf))]
        self.zf = zf

    # search patch
    # return the 'cls', 'loc', 'cen'
    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            xf = [self.align(xf[i]) for i in range(len(xf))]
        features = self.xcorr_depthwise(xf[0],self.zf[0])
        
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        gn_kn_z = zf
        gn_kn_x = xf
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            zf = [self.align(_zf) for _zf in zf]
            xf = [self.align(_xf) for _xf in xf]

        print('zf:size()', zf.size())
        print('xf:size', xf.size())

        features = self.xcorr_depthwise(xf[0],zf[0])
        
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        
        print("feature_before", features.size())
        features = self.down(features)

        # print("features：", features.size())
        
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        
        # print('locations, cls, loc', locations.size(), cls.size(), loc.size())
        
        # print('cen, label_cls, label_loc', cen.size(), label_cls.size(), label_loc.size())
        
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss

        # get feature
        KD_feature['general_z'] = gn_kn_z
        KD_feature['general_x'] = gn_kn_x
        KD_feature['xcorr'] = features

        return outputs, zf, xf, KD_feature
