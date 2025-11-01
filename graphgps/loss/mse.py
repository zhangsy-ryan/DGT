import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('mse_losses')
def mse_losses(pred, true):
    if cfg.model.loss_fun == 'mse':
        mse_loss = nn.MSELoss()
        loss = mse_loss(pred, true)
        return loss, pred
