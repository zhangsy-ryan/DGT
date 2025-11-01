import torch

from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops
from torch_scatter import scatter
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

# -----------------
#      Utils
# -----------------

def get_rw_landing_probs(ksteps, dense_adj,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Had to recode this as the original function in compute_pe_stats does not work for batches

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    num_nodes = dense_adj.size(1)
    deg = dense_adj.sum(2, keepdim=True) # Out degrees. (batch_size) x (Num nodes) x (1)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    # P = D^-1 * A
    P = deg_inv * dense_adj  # (Batch_size) x (Num nodes) x (Num nodes)

    rws = []
    rws_all = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            rws_all.append(Pk)
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.stack(rws, dim=2)  # (Batch_size) x (Num nodes) x (K steps)
    rw_landing_all = torch.stack(rws_all, dim=3)  # (Batch_size) x (Num nodes) x (Num nodes) x (K steps)

    return rw_landing, rw_landing_all


# -------------------------
# Copied from latest version of torch geometric
# -------------------------
from typing import List
from torch import Tensor

from torch_geometric.utils import degree


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.
    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
    :rtype: :class:`List[Tensor]`
    Example:
        >>> src = torch.arange(14)
        >>> batch = torch.tensor([0, 0, 0, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]), tensor([9]), tensor([10, 11, 12, 13]))
    """
    sizes = (degree(batch, dtype=torch.long)**2).tolist()
    return src.split(sizes, dim)


def reshape_flattened_adj(edge_features, batch):
    '''
    Transforms a PyG batch of edge features of shape (N_1+...+N_B, edge_dim) into a 
    padded tensor of shape (B, N_max, edge_dim),
    where N_i = (number of nodes of graph i)^2, and N_max = max(N_i)
    '''
    edge_features_list = unbatch(edge_features, batch)
    num_nodes = [int(e.size(0)**0.5) for e in edge_features_list]
    max_nodes = max(num_nodes)
    n_batch = len(edge_features_list)
    padded_edge_features = edge_features_list[0].new_zeros((n_batch, max_nodes, max_nodes, edge_features_list[0].size(-1)))
    for i, e in enumerate(edge_features_list):
        if num_nodes[i] == 0: continue
        padded_edge_features[i, :num_nodes[i], :num_nodes[i]] = e.view(num_nodes[i], num_nodes[i], -1)

    return padded_edge_features

def get_dense_indices_from_sparse(edge_index, batch):
    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    return idx0, idx1, idx2


# -----------------
#     Encoders
# -----------------

@register_edge_encoder('RWSEonthefly')
class RWSEcomputer(torch.nn.Module):
    '''
    Compute `batch.pestat_RWSE` and `batch.edge_RWSE`
    In the case of 
    '''
    def __init__(self):
        super().__init__()

        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        self.ksteps = kernel_param.times

    def forward(self, batch):
        dense_adj = to_dense_adj(batch.edge_index, batch=batch.batch)
        # This next line is just to get the node mask (perhaps overkill)
        _, mask = to_dense_batch(batch.edge_index.new_zeros(batch.num_nodes), batch=batch.batch)
        rw_landing, rw_landing_all = get_rw_landing_probs(ksteps=self.ksteps,
                                        dense_adj=dense_adj.double())
        batch.pestat_RWSE = rw_landing[mask].float()
        if cfg.posenc_RWSE.enable_edges:
            batch.edge_RWSE = rw_landing_all.float()

        return batch

@register_edge_encoder('SPDEEdge')
class SPDEEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()

        self.add_dense_edge_features = dense

        # Path lengths go from 1 to spd_max_length.
        # Add two types for self-connections and non-connections
        num_types = cfg.dataset.spd_max_length + 2
        if num_types < 1:
            raise ValueError(f"Invalid 'spd_max_length': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        self.e2e_encoder = torch.nn.Embedding(num_embeddings=num_types,
                                              embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Shifting lengths by 1 and adding 0s on the diagonal to distinguish
        # non-connected nodes from self-connections
        batch.spd_index, batch.spd_lengths = add_self_loops(
            batch.spd_index, batch.spd_lengths + 1, fill_value=0)
        # Doing things in this order (first embedding, then transforming to dense,
        # ensures that padding remains 0)
        spd_embedding = self.encoder(batch.spd_lengths)
        spd_dense = to_dense_adj(batch.spd_index, batch=batch.batch, edge_attr=spd_embedding)

        batch_idx, row, col = get_dense_indices_from_sparse(batch.edge_index, batch.batch)
        batch.edge_attr = spd_dense[batch_idx, row, col]

        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.edge_dense = spd_dense
        
        batch.e2e_spd_index, batch.e2e_spd_lengths = add_self_loops(
            batch.e2e_spd_index, batch.e2e_spd_lengths + 1, fill_value=0)
        e2e_spd_embedding = self.e2e_encoder(batch.e2e_spd_lengths)
        e2e_spd_dense = to_dense_adj(batch.e2e_spd_index, batch=batch.e_batch, edge_attr=e2e_spd_embedding)
        batch_idx, row, col = get_dense_indices_from_sparse(batch.e2e_edge_index, batch.e_batch)
        batch.e2e_edge_attr = e2e_spd_dense[batch_idx, row, col]
        if self.add_dense_edge_features:
            batch.e2e_edge_dense = e2e_spd_dense

        return batch


@register_edge_encoder('RWSEEdge')
class RWSEEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()

        edge_pe_in_dim = len(cfg.posenc_RWSE.kernel.times) # Size of the kernel-based PE embedding
        self.reshape = (cfg.posenc_RWSE.precompute == True)
        self.add_dense_edge_features = dense

        self.encoder = torch.nn.Linear(edge_pe_in_dim, emb_dim) # Watch out for padding here? Use bias=False?
        self.e2e_encoder = torch.nn.Linear(edge_pe_in_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        Ideally this step should be done in the data loading, but the torch geometric DataLoader
        forces us to flatten the dense edge feature matrices
        '''
        # First reshape the edge features into (n_batch, max_nodes, max_nodes, edge_dim)
        if self.reshape:
            batched_edge_features = reshape_flattened_adj(batch.edge_RWSE, batch.batch)
        else:
            batched_edge_features = batch.edge_RWSE
        del batch.edge_RWSE # This is the largest tensor in the batch, deleting it to save space?
        batched_edge_features = self.encoder(batched_edge_features)
        # For the sparse edge_attr we keep the original edges and do not add new ones
        batch_idx, row, col = get_dense_indices_from_sparse(batch.edge_index, batch.batch)
        batch.edge_attr = (batched_edge_features[batch_idx, row, col] + batched_edge_features[batch_idx, col, row]) / 2
        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.edge_dense = batched_edge_features
        
        if self.reshape:
            batched_e2e_edge_features = reshape_flattened_adj(batch.e2e_edge_RWSE, batch.e_batch)
        else:
            batched_e2e_edge_features = batch.e2e_edge_RWSE
        del batch.e2e_edge_RWSE # This is the largest tensor in the batch, deleting it to save space?
        batched_e2e_edge_features = self.e2e_encoder(batched_e2e_edge_features)
        # For the sparse edge_attr we keep the original edges and do not add new ones
        batch_idx, row, col = get_dense_indices_from_sparse(batch.e2e_edge_index, batch.e_batch)
        batch.e2e_edge_attr = batched_e2e_edge_features[batch_idx, row, col]
        if self.add_dense_edge_features:
            # Maybe directly concatenate this instead, as for NodePE?
            batch.e2e_edge_dense = batched_e2e_edge_features

        return batch


@register_edge_encoder('RWSE-SPDEEdge')
class RWSESPDEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, dense=False):
        super().__init__()
        self.add_dense_edge_features = dense
        self.rwse_encoder = RWSEEdgeEncoder(emb_dim, dense)
        self.spde_encoder = SPDEEdgeEncoder(emb_dim, dense)

    def forward(self, batch):
        batch = self.rwse_encoder(batch)
        edge_rwse_sparse = batch.edge_attr
        edge_rwse_dense = batch.edge_dense
        e2e_edge_rwse_sparse = batch.e2e_edge_attr
        e2e_edge_rwse_dense = batch.e2e_edge_dense
        batch = self.spde_encoder(batch)
        edge_spde_sparse = batch.edge_attr
        edge_spde_dense = batch.edge_dense
        e2e_edge_spde_sparse = batch.e2e_edge_attr
        e2e_edge_spde_dense = batch.e2e_edge_dense
        batch.edge_attr = edge_rwse_sparse
        batch.e2e_edge_attr = e2e_edge_rwse_sparse
        if self.add_dense_edge_features:
            batch.edge_dense = edge_rwse_dense
            batch.e2e_edge_dense = e2e_edge_rwse_dense

        return batch
