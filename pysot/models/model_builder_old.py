# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from syslog import LOG_SYSLOG

import torch
import torch.nn as nn
import torch.nn.functional as F
#import os
from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
            self.neck_2 = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        if cfg.ALIGN.ALIGN:
            self.align = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)
            self.align_2 = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)
        self.car_head_2 = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)
        
        # 逆卷积运算
        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)


    # template patch
    def template(self, z, mark = 1):
        zf = self.backbone(z)
        if mark == 1:
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
            if cfg.ALIGN.ALIGN:
                zf = [self.align(zf[i]) for i in range(len(zf))]
        if mark == 2:
            if cfg.ADJUST.ADJUST:
                zf = self.neck_2(zf)
            if cfg.ALIGN.ALIGN:
                zf = [self.align_2(zf[i]) for i in range(len(zf))]
        self.zf = zf

    # search patch
    # return the 'cls', 'loc', 'cen'
    def track(self, x, mark = 1):
        xf = self.backbone(x)
        if mark == 1:
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            if cfg.ALIGN.ALIGN:
                xf = [self.align(xf[i]) for i in range(len(xf))]
        if mark == 2:
            if cfg.ADJUST.ADJUST:
                    xf = self.neck_2(xf)
            if cfg.ALIGN.ALIGN:
                xf = [self.align_2(xf[i]) for i in range(len(xf))]
        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)
        
        if mark == 1:
            cls, loc, cen = self.car_head(features)
        if mark == 2:
            cls, loc, cen = self.car_head_2(features)
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
    
        
        return cls_loss, reg_loss, cen_loss
        
    def forward(self, data, mark = 1, target = 0):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        
        # 这两个输出是什么样的?
        # 源域和目标域都有label_cls和bbox吗? 区别在哪里?
        # 源域GT; 目标域是salient detection?
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        # print('label_cls: ', label_cls)
        # print('label_loc: ', label_loc)
        

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        
        if mark == 1:
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)
            if cfg.ALIGN.ALIGN:
                zf = [self.align(_zf) for _zf in zf]
                xf = [self.align(_xf) for _xf in xf]
        if mark == 2:
            if cfg.ADJUST.ADJUST:
                zf = self.neck_2(zf)
                xf = self.neck_2(xf)
            if cfg.ALIGN.ALIGN:
                zf = [self.align_2(_zf) for _zf in zf]
                xf = [self.align_2(_xf) for _xf in xf]
                
        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)
    
        # 输出的数据格式是什么样的？与GT有区别吗？
        if mark == 1:
            cls, loc, cen = self.car_head(features)
        if mark == 2:
            cls, loc, cen = self.car_head_2(features)
            
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
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
        if target == 0:
            return outputs, zf, xf
        if target == 1:
            return cls, loc, cen
