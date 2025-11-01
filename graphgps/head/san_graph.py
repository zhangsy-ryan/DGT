import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from graphgps.encoder.relative_pe_encoder import get_dense_indices_from_sparse


@register_head('san_graph')
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        # graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = batch.x
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = self.pooling_fun(graph_emb, batch.batch)
        # batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


import torch
@register_head('line_graph')
class LineGraphHead(nn.Module):
    """
    DGT prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        list_edge_FC_layers = [nn.Linear(dim_in * 2, dim_in)] + \
        [nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True) for l in range(L)]
        self.edge_FC_layers = nn.ModuleList(list_edge_FC_layers)
        self.out_layer = nn.Linear(2 * dim_in // 2 ** L, dim_out, bias=True)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = batch.x
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_feature = self.pooling_fun(graph_emb, batch.batch)
        
        graph_edge_emb = torch.cat([
            batch.e,
            batch.x[batch.edge_index[0, batch.edge_index[0] < batch.edge_index[1]]] + batch.x[batch.edge_index[1, batch.edge_index[0] < batch.edge_index[1]]]
        ], dim=1)
        for l in range(self.L + 1):
            graph_edge_emb = self.edge_FC_layers[l](graph_edge_emb)
            graph_edge_emb = self.activation(graph_edge_emb)
        graph_edge_feature = self.pooling_fun(graph_edge_emb, batch.e_batch)
        graph_edge_feature = torch.cat([
            graph_edge_feature, 
            graph_edge_feature.new_zeros(
                (graph_feature.size(0) - graph_edge_feature.size(0), 
                 graph_edge_feature.size(1))
            )
        ], dim=0)

        graph_feature = torch.cat([graph_feature, graph_edge_feature], dim=1)
        batch.graph_feature = self.out_layer(graph_feature)
        pred, label = self._apply_index(batch)
        return pred, label
