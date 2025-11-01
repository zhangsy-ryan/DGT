import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_mean, scatter_add
from graphgps.encoder.relative_pe_encoder import get_dense_indices_from_sparse


def pad_batch_size(x, n_batch):
    if x.size(0) < n_batch:
        return torch.cat([x, x.new_zeros(n_batch - x.size(0), *x.size()[1:])], dim=0)
    else:
        return x



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False, attn_dropout=0.0):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        
    def forward(self, h, e_att, e_val, attn_mask):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        n_batch, num_nodes, _ = V_h.shape

        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.view(n_batch, num_nodes, self.num_heads, self.out_dim)

        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        # Compute node-node attention scores.
        # Leave trailing dimension to add the edge features
        scores = torch.einsum('bihk,bjhk->bijh', Q_h, K_h).unsqueeze(-1)

        # Make sure attention scores for padding are 0.
        # Keep -1e24 instead of -infinity to avoid NaN errors in softmax
        attn_mask = attn_mask.view(-1, num_nodes, num_nodes, 1, 1)
        scores = scores - 1e24 * (~attn_mask)
        # scores = scores + torch.log(attn_mask)
        
        # Compute edge features if necessary
        E_att = e_att
        E_val = e_val
        # Match the shape of `scores`
        E_att = E_att.view(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

        # Bias the attention matrix with the edge filters
        scores = scores + E_att
        # The head dimension is not needed anymore, merge it with the feature dimension
        scores = scores.reshape(-1, num_nodes, num_nodes, self.num_heads * self.out_dim)

        # L1-normalize across emitting nodes
        scores = nn.functional.softmax(scores, dim=2)

        # Dropout connections
        attn_mask = self.attn_dropout(attn_mask.float()).squeeze(-1)
        scores = scores * attn_mask

        # Modulate and sum the node messages hi <- Sum_j a(i,j) * Vj
        h = torch.einsum('bijk,bjk->bik', scores, V_h)

        # Add the edge messages
        h += (scores * E_val).sum(2)

        # Make sure return type is contiguous
        return h.contiguous()


class DGTLayer(nn.Module):
    def __init__(self, dim_h, num_heads, act='relu', 
                 dropout=0.0, attn_dropout=0.0, layer_norm=False, batch_norm=True):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = register.act_dict[act]

        self.self_attn_n = MultiHeadAttentionLayer(
            dim_h, dim_h//num_heads, num_heads, attn_dropout=self.attn_dropout)
        self.linear_n = nn.Linear(dim_h, dim_h)
        
        self.self_attn_e = MultiHeadAttentionLayer(
            dim_h, dim_h//num_heads, num_heads, attn_dropout=self.attn_dropout)
        self.linear_e = nn.Linear(dim_h, dim_h)

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_n = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_n = nn.BatchNorm1d(dim_h)
        self.dropout_n = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1_n = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2_n = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            self.norm2_n = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm2_n = nn.BatchNorm1d(dim_h)
        self.ff_dropout1_n = nn.Dropout(dropout)
        self.ff_dropout2_n = nn.Dropout(dropout)

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_e = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_e = nn.BatchNorm1d(dim_h)
        self.dropout_e = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1_e = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2_e = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            self.norm2_e = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm2_e = nn.BatchNorm1d(dim_h)
        self.ff_dropout1_e = nn.Dropout(dropout)
        self.ff_dropout2_e = nn.Dropout(dropout)

    def forward(self, batch):
        # Residual connection
        h_n = batch.x
        h_n_in = h_n
        h_e = batch.e
        h_e_in = h_e

        # Multi-head attention.
        h_n_dense, _ = to_dense_batch(h_n, batch.batch)
        h_n = self.self_attn_n(
            h_n_dense, 
            e_att=batch.edge_attention, 
            e_val=batch.edge_values, 
            attn_mask=batch.attn_mask
        )[batch.mask]
        h_n = self.linear_n(h_n)
        h_n = self.dropout_n(h_n)
        h_n = h_n + h_n_in
        if self.layer_norm:
            h_n = self.norm1_n(h_n, batch.batch)
        if self.batch_norm:
            h_n = self.norm1_n(h_n)
        
        # Feed forward block.
        h_n = h_n + self._n_ff_block(h_n)
        if self.layer_norm:
            h_n = self.norm2_n(h_n, batch.batch)
        if self.batch_norm:
            h_n = self.norm2_n(h_n)


        # Multi-head attention.
        h_e_dense, _ = to_dense_batch(h_e, batch.e_batch, batch_size=batch.mask.size(0))
        h_e = self.self_attn_e(
            h_e_dense, 
            e_att=batch.e2e_edge_attention, 
            e_val=batch.e2e_edge_values, 
            attn_mask=batch.e_attn_mask
        )[batch.e_mask]
        h_e = self.linear_e(h_e)
        h_e = self.dropout_e(h_e)
        h_e = h_e + h_e_in
        if self.layer_norm:
            h_e = self.norm1_e(h_e, batch.e_batch)
        if self.batch_norm:
            h_e = self.norm1_e(h_e)
        
        # Feed forward block.
        h_e = h_e + self._e_ff_block(h_e)
        if self.layer_norm:
            h_e = self.norm2_e(h_e, batch.e_batch)
        if self.batch_norm:
            h_e = self.norm2_e(h_e)

        batch.x = h_n
        batch.e = h_e
        return batch

    def _n_ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1_n(self.activation(self.ff_linear1_n(x)))
        return self.ff_dropout2_n(self.ff_linear2_n(x))

    def _e_ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1_e(self.activation(self.ff_linear1_e(x)))
        return self.ff_dropout2_e(self.ff_linear2_e(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'heads={self.num_heads}'
        return s