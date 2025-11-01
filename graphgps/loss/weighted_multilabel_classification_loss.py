import torch.nn as nn
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('weighted_multilabel_cross_entropy')
def weighted_multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'multilabel_focal_loss' and cfg.model.loss_fun != 'multilabel_cross_entropy' and \
            cfg.model.loss_fun != 'weighted_multilabel_cross_entropy':
            raise ValueError("Only '(weighted_)multilabel_cross_entropy' and 'multilabel_focal_loss' loss_funs supported with "
                             "'classification_multilabel' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        weight = []
        for i in range(true.shape[1]):
            class_weight = torch.bincount(true[:, i][true[:, i] == true[:, i]])
            _weight = torch.zeros_like(true[:, i]).float().to(true.device)
            _weight[true[:, i] == 0] = class_weight[0]
            _weight[true[:, i] == 1] = class_weight[1]
            weight.append(_weight)
        weight = torch.stack(weight, dim=0)
        is_labeled = true == true  # Filter our nans.
        return weight[is_labeled] * bce_loss(pred[is_labeled], true[is_labeled].float()), pred
