import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch, to_dense_adj
from graphgps.layer.dgt_layer import DGTLayer
from graphgps.encoder.relative_pe_encoder import get_dense_indices_from_sparse
from torch_scatter import scatter_sum


from math import pi as PI
class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1. / x + a * x_pow_p0 + b * x_pow_p1 + 
                c * x_pow_p2) * (x < 1.0).float()


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (bessel_basis,
                                                             real_sph_harm)

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        # self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)

    def forward(self, angle):
        cbf = torch.cat([f(angle) for f in self.sph_funcs], dim=-1)
        return cbf


def pad_batch_size(x, n_batch):
    if x.size(0) < n_batch:
        return torch.cat([x, x.new_zeros(n_batch - x.size(0), *x.size()[1:])], dim=0)
    else:
        return x


class PositionEncoder(torch.nn.Module):
    def __init__(self):
        super(PositionEncoder, self).__init__()
        self.rbf = BesselBasisLayer(3, 5.0, 5)
        self.sbf = SphericalBasisLayer(3, 3, 5.0, 5)
        self.dis_att_encoder = torch.nn.Linear(3, cfg.gnn.dim_inner)
        self.dis_val_encoder = torch.nn.Linear(3, cfg.gnn.dim_inner)
        self.cos_att_encoder = torch.nn.Linear(3, cfg.gnn.dim_inner)
        self.cos_val_encoder = torch.nn.Linear(3, cfg.gnn.dim_inner)
        self.bdl_encoder = torch.nn.Linear(3, cfg.gnn.dim_inner)
    
    def forward(self, batch):
        pos, pos_mask = to_dense_batch(batch.pos, batch.batch)
        dis = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1, p=2, keepdim=True)
        dis = self.rbf(dis + 1e-6)
        dis_mask = pos_mask.unsqueeze(2) * pos_mask.unsqueeze(1)
        
        vec_src = batch.pos[batch.edge_index[0, batch.edge_index[0] < batch.edge_index[1]]]
        vec_dst = batch.pos[batch.edge_index[1, batch.edge_index[0] < batch.edge_index[1]]]
        vec, vec_mask = to_dense_batch(vec_dst - vec_src, batch.e_batch)
        bdl = (vec_dst - vec_src).norm(dim=-1, p=2, keepdim=True)
        bdl = self.rbf(bdl)
        cos = (vec.unsqueeze(2) * vec.unsqueeze(1)).sum(dim=-1, keepdim=True) / (
            vec.unsqueeze(1).norm(dim=-1, p=2, keepdim=True) + 1e-6) / (
            vec.unsqueeze(2).norm(dim=-1, p=2, keepdim=True) + 1e-6)
        vec_edge_index = to_dense_batch(
            batch.edge_index[:, batch.edge_index[0] < batch.edge_index[1]].t(), 
            batch.e_batch, -1)[0]
        cos_mask = torch.zeros(cos.shape[:3], dtype=torch.long).to(cos.device)
        cos_mask[(vec_edge_index[:, :, None, 0] == vec_edge_index[:, None, :, 0]) & (vec_edge_index[:, :, None, 1] != vec_edge_index[:, None, :, 1]) & (vec_edge_index[:, :, None, 0] >= 0)] = 1
        cos_mask[(vec_edge_index[:, :, None, 1] == vec_edge_index[:, None, :, 1]) & (vec_edge_index[:, :, None, 0] != vec_edge_index[:, None, :, 0]) & (vec_edge_index[:, :, None, 1] >= 0)] = 1
        cos_mask[(vec_edge_index[:, :, None, 0] == vec_edge_index[:, None, :, 1]) & (vec_edge_index[:, :, None, 0] >= 0)] = -1
        cos_mask[(vec_edge_index[:, :, None, 1] == vec_edge_index[:, None, :, 0]) & (vec_edge_index[:, :, None, 1] >= 0)] = -1
        cos = torch.arccos((cos * cos_mask.float().unsqueeze(-1)).clamp(-1.0, 1.0))
        cos = self.sbf(cos)

        dis_att = self.dis_att_encoder(dis) * dis_mask.float().unsqueeze(-1)
        dis_val = self.dis_val_encoder(dis) * dis_mask.float().unsqueeze(-1)
        cos_att = self.cos_att_encoder(cos) * cos_mask.float().unsqueeze(-1).abs()
        cos_val = self.cos_val_encoder(cos) * cos_mask.float().unsqueeze(-1).abs()
        bdl_ebd = self.bdl_encoder(bdl)
        batch.edge_attention += dis_att
        batch.edge_values += dis_val
        batch.e2e_edge_attention += cos_att
        batch.e2e_edge_values += cos_val
        batch.bdl_ebd = bdl_ebd

        return batch


class NodeEdgeEncoder(torch.nn.Module):
    def __init__(self):
        super(NodeEdgeEncoder, self).__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
            torch.nn.Mish(),
            torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
            torch.nn.Mish(),
            torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
        )
    
    def forward(self, batch):
        batch.e = batch.edge_attr[batch.edge_index[0] < batch.edge_index[1]]
        batch.e = self.edge_mlp(
            torch.concat([
                batch.e,
                batch.x[batch.edge_index[0, batch.edge_index[0] < batch.edge_index[1]]] + \
                batch.x[batch.edge_index[1, batch.edge_index[0] < batch.edge_index[1]]],
            ], dim=-1)
        )
        batch.x = self.node_mlp(
            torch.concat([
                batch.x,
                pad_batch_size(scatter_sum(batch.edge_attr, batch.edge_index[1], dim=0), batch.x.size(0)),
            ], dim=-1)
        )
        return batch


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.posenc_RWSE.enable and not cfg.posenc_RWSE.precompute:
            self.rwse_compute = register.edge_encoder_dict['RWSEonthefly']()
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
        self.node_edge_encoder = NodeEdgeEncoder()
        if cfg.model.type == 'DGTModel3D':
            self.position_encoder = PositionEncoder()

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        
        _, mask = to_dense_batch(batch.x, batch.batch)
        batch.mask = mask
        batch.attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        if cfg.model.type == 'DGTModel3D':
            batch.e += batch.bdl_ebd

        _, e_mask = to_dense_batch(batch.e, batch.e_batch, batch_size=batch.mask.size(0))
        batch.e_mask = e_mask
        batch.e_attn_mask = e_mask.unsqueeze(1) * e_mask.unsqueeze(2)

        batch.e2e_edge_dense = pad_batch_size(batch.e2e_edge_dense, batch.mask.size(0))
        batch.e2e_edge_attention = pad_batch_size(batch.e2e_edge_attention, batch.mask.size(0))
        batch.e2e_edge_values = pad_batch_size(batch.e2e_edge_values, batch.mask.size(0))

        return batch


@register_network('DGTModel')
class DGTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(DGTLayer(
                dim_h=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('DGTModel3D')
class DGTModel3D(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(DGTLayer(
                dim_h=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch