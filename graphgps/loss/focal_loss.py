import torch
from libauc.losses import FocalLoss
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

@register_loss('focal_loss')
def binary_focal_loss(pred, true):
    if cfg.model.loss_fun == 'focal_loss':
        loss = FocalLoss()(pred, true.float())
        return loss, pred