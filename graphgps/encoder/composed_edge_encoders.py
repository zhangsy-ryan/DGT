import torch
from torch_geometric.graphgym.config import cfg

from torch_geometric.graphgym.models.encoder import BondEncoder
from torch_geometric.graphgym.register import register_edge_encoder

from graphgps.encoder.type_dict_encoder import TypeDictEdgeEncoder
from graphgps.encoder.relative_pe_encoder import RWSEEdgeEncoder, SPDEEdgeEncoder, RWSESPDEdgeEncoder
from graphgps.encoder.dense_edge_encoder import DenseEdgeEncoder
from graphgps.encoder.topology_edge_encoder import RingEdgeEncoder
from graphgps.encoder.linear_edge_encoder import LinearEdgeEncoder
from graphgps.encoder.dummy_edge_encoder import DummyEdgeEncoder
from graphgps.encoder.ogb_encoder import Bond1Encoder, BondEncoderAvg
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_batch, to_dense_adj

def pad_batch_size(x, n_batch):
    if x.size(0) < n_batch:
        return torch.cat([x, x.new_zeros(n_batch - x.size(0), *x.size()[1:])], dim=0)
    else:
        return x

def concat_edge_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the other an RPE Encoder.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during runtime, once the config is loaded.

    Returns:
        new edge encoder class
    """

    class Concat2EdgeEncoder(torch.nn.Module):
        """Encoder that concatenates two edge encoders.
        """
        enc_type_cls = None
        enc_pe_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()
            
            # Only CSA layer uses dense edge features.
            transformer_layer = cfg.gt.layer_type.split('+')[1]
            self.add_dense_edge_features = (transformer_layer in ['CSA', 'DGT'])
            self.share_edge_features = cfg.dataset.edge_encoder_shared
            # Whether to add special features for bonds that are part of a ring.
            # if cfg.dataset.rings_coalesce_edges == True, 
            # these features are already taken into account in the `edge_attr`,
            # and do not need to be adressed here
            add_rings = (cfg.dataset.rings == True) and (cfg.dataset.rings_coalesce_edges == False)
        
            type_dim = dim_emb
            self.type_encoder = [self.enc_type_cls(type_dim)]
            if self.add_dense_edge_features:
                self.type_encoder.append(DenseEdgeEncoder(type_dim, ignore_rings=~add_rings))
            self.type_encoder = torch.nn.Sequential(*self.type_encoder)

            pe_dim = dim_emb
            if self.enc_pe_cls is not None:
                self.encoder2 = self.enc_pe_cls(pe_dim, dense=self.add_dense_edge_features)
                if add_rings:
                    self.encoder2 = torch.nn.Sequential(self.encoder2, RingEdgeEncoder(pe_dim))

            if self.share_edge_features:
                self.shared_edge_encoder = SharedEdgeEncoder(dim_emb)

        def forward(self, batch):
            batch = self.type_encoder(batch)
            
            if self.enc_pe_cls is not None:
                x = batch.x
                edge_attr = batch.edge_attr
                edge_dense = batch.edge_dense
                e2e_edge_dense = batch.e2e_edge_dense

                batch = self.encoder2(batch)
                batch.x = x
                batch.edge_attr = edge_attr
                batch.edge_dense += pad_batch_size(edge_dense, int(batch.batch.max()) + 1)
                batch.e2e_edge_dense += e2e_edge_dense

            if self.share_edge_features == True:
                batch = self.shared_edge_encoder(batch)
        
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2EdgeEncoder.enc_type_cls = encoder_classes[0]
        Concat2EdgeEncoder.enc_pe_cls = encoder_classes[1]
        Concat2EdgeEncoder.enc2_name = pe_enc_names[0]
        return Concat2EdgeEncoder
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")

# Dataset-specific edge encoders.
edge_ds_encs = {'Bond': BondEncoder,
                'Bond1': Bond1Encoder,
                'BondCustom': BondEncoderAvg,
                'TypeDictEdge': TypeDictEdgeEncoder,
                'LinearEdge': LinearEdgeEncoder,
                'Dummy': DummyEdgeEncoder,
                'Dense': DenseEdgeEncoder,
                'None': None}

# Positional Encoding node encoders.
edge_pe_encs = {'RWSE': RWSEEdgeEncoder,
                'SPDE': SPDEEdgeEncoder,
                'RWSE-SPDE': RWSESPDEdgeEncoder,
                'None': None,
               }

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in edge_ds_encs.items():
    for pe_enc_name, pe_enc_cls in edge_pe_encs.items():
        register_edge_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_edge_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )


@register_edge_encoder('SharedEdge')
class SharedEdgeEncoder(torch.nn.Module):
    '''
    Transforms dense edge features into `edge_att` and `edge_value`.
    To be shared among CSA layers.

    Supposes `batch.edge_dense` has been set.
    (`batch.edge_dense` set via the DenseEdgeEncoder or the RPE modules)
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.attention_encoder = torch.nn.Linear(emb_dim, emb_dim)
        self.value_encoder = torch.nn.Linear(emb_dim, emb_dim)
        self.e2e_attention_encoder = torch.nn.Linear(emb_dim, emb_dim)
        self.e2e_value_encoder = torch.nn.Linear(emb_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        '''
        '''
        batch.edge_attention = self.attention_encoder(batch.edge_dense)
        batch.edge_values = self.value_encoder(batch.edge_dense)
        batch.e2e_edge_attention = self.e2e_attention_encoder(batch.e2e_edge_dense)
        batch.e2e_edge_values = self.e2e_value_encoder(batch.e2e_edge_dense)
        # batch.edge_dense = None # Save space by erasing `edge_dense`

        return batch