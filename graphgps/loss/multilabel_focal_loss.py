import torch
from libauc.losses import FocalLoss
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

@register_loss('multilabel_focal_loss')
def multilabel_focal_loss(pred, true):
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'multilabel_focal_loss' and cfg.model.loss_fun != 'multilabel_cross_entropy' and \
            cfg.model.loss_fun != 'weighted_multilabel_cross_entropy':
            raise ValueError("Only '(weighted_)multilabel_cross_entropy' and 'multilabel_focal_loss' loss_funs supported with "
                             "'classification_multilabel' task_type.")
        loss = FocalLoss()
        is_labeled = true == true  # Filter our nans.
        return loss(pred[is_labeled], true[is_labeled].float()), pred