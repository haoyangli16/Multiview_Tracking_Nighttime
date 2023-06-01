import torch
from torch import nn
import torch.nn.functional as F
from pysot.models.head.car_head import CARHead
from pysot.core.config import cfg

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1, 2)
    # print("pred: ", pred)
    # print("label: ", label)
    pos = label.data.eq(1).nonzero(as_tuple =False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple =False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def loc_IOU_loss(pred, target, weight=None):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                    (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                    torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                    torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum() / weight.sum()
    else:
        assert losses.numel() != 0
        return losses.mean()
    
def calculate_the_loss_con(cls1, loc1, cen1, cls2, loc2, cen2):
    
    cls_loss = select_cross_entropy_loss(cls1, cls2)
    
    loc1_flatten = (loc1.permute(0, 2, 3, 1).contiguous().view(-1, 4))
    loc2_flatten = (loc2.permute(0, 2, 3, 1).contiguous().view(-1, 4))
    reg_loss = loc_IOU_loss(loc1_flatten, loc2_flatten)

    centerness1_flatten = (cen1.view(-1))
    centerness2_flatten = (cen2.view(-1))
    cen_loss = nn.BCEWithLogitsLoss(centerness1_flatten,centerness2_flatten)

    # get loss
    outputs = {}
    outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        cfg.TRAIN.LOC_WEIGHT * reg_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
    outputs['cls_loss'] = cls_loss
    outputs['loc_loss'] = reg_loss
    outputs['cen_loss'] = cen_loss
    
    return outputs