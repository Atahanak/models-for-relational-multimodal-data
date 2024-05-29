from __future__ import annotations

from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import (
    LayerNorm, 
    Linear,
    Module, 
    ReLU, 
    LeakyReLU,
    Dropout,
    Sequential, 
    ModuleList,
    Parameter,
    TransformerEncoderLayer,
)

from src.datasets.util.mask import PretrainType
import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
    TimestampEncoder
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from torch_geometric.nn import PNAConv
from torch_geometric.nn import BatchNorm

from ..decoder import SelfSupervisedLPHead
from ..decoder import SupervisedHead
from ..decoder import SelfSupervised_MCM_MV_LP_Head

from icecream import ic
import sys

class FTTransformerPNAFused(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Number of layers.  (default: :obj:`3`)
        col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call
            :class:`torch_frame.nn.encoder.EmbeddingEncoder()` for categorical
            feature and :class:`torch_frame.nn.encoder.LinearEncoder()`
            for numerical feature)
        pretrain (bool): If :obj:`True`, the model will be pre-trained, otherwise it will be trained end-to-end. (default: :obj:`False`)
    """
    def __init__(
        self,

        # general parameters
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        
        # training parameters
        pretrain: set[PretrainType] = False,

        
        # PNA parameters
        deg=None,
        node_dim: int = 1,
        nhidden: int = 128,
        edge_dim: int = None,
        final_dropout: float = 0.5,

        # fttransformer parameters
        feedforward_channels: Optional[int] = None,
        nhead: int = 8,
        dropout: float = 0.2,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        
        self.pretrain = pretrain
        self.channels = channels
        self.nhidden = nhidden
        self.edge_dim = edge_dim
        
        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
                stype.timestamp: TimestampEncoder()
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # fttransformer
        self.cls_embedding = Parameter(torch.empty(channels))
        
        # PNA
        self.node_emb = Linear(node_dim, nhidden)
        self.edge_emb = Linear(edge_dim, nhidden)

        #backbone
        self.backbone = ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.backbone.append(
                    FTTransformerPNAParallelLayer(
                        channels, 
                        nhead, 
                        feedforward_channels, 
                        dropout, 
                        activation, 
                        nhidden,
                        final_dropout,
                        deg
                    )
                )
            else:
                self.backbone.append(
                    FTTransformerPNAFusedLayer(
                        channels, 
                        nhead, 
                        feedforward_channels, 
                        dropout, 
                        activation, 
                        nhidden,
                        final_dropout,
                        deg
                    )
                )

        if pretrain:
            # Only currently accomdates combinations of {MCM + LP} and {MCM + LP + MV}
            num_numerical = len(col_names_dict[stype.numerical]) if stype.numerical in col_names_dict else 0 
            num_categorical = [len(col_stats[col][StatType.COUNT][0]) for col in col_names_dict[stype.categorical]] if stype.categorical in col_names_dict else 0
            if pretrain == {PretrainType.MASK, PretrainType.MASK_VECTOR, PretrainType.LINK_PRED}:
                self.decoder = SelfSupervised_MCM_MV_LP_Head(
                    channels=channels, 
                    num_numerical=num_numerical, 
                    num_categorical=num_categorical, 
                    nhidden=nhidden, 
                    dropout=final_dropout
                )
            else:
                self.decoder = SelfSupervisedLPHead(
                    channels=channels, 
                    num_numerical=num_numerical, 
                    num_categorical=num_categorical, 
                    nhidden=nhidden, 
                    dropout=final_dropout
            )
        else:
            self.decoder = SupervisedHead(channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.encoder.reset_parameters()
        for layer in self.backbone:
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def get_shared_params(self):
        param_groups = [
            self.encoder.parameters(),
            [self.cls_embedding],  # Wrap single parameters in a list
            self.node_emb.parameters(),
            self.edge_emb.parameters(),
            self.backbone[:-1].parameters(),
        ]
        
        # Flatten the param groups into a single list
        flat_params = [param for group in param_groups for param in group]
        
        return flat_params

    def zero_grad_shared_params(self):
        for param in self.get_shared_params():
            if param.grad is not None:
                param.grad.data.zero_()



    def forward(self, x, edge_index, edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        B = len(pos_edge_attr)
        pos_edge_attr, _ = self.encoder(pos_edge_attr)
        
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        x_tab = torch.cat([x_cls, pos_edge_attr], dim=1)

        x_gnn = self.node_emb(x)
        edge_attr, _ = self.encoder(edge_attr)
        edge_attr = edge_attr.view(-1, self.edge_dim)
        edge_attr = self.edge_emb(edge_attr)

        for fused_layer in self.backbone:
            x_tab, x_gnn, edge_attr = fused_layer(x_tab, x_gnn, edge_index, edge_attr)
            # x_t, x_g, e_attr = fused_layer(x_tab, x_gnn, edge_index, edge_attr)
            # x_tab = x_tab + x_t
            # x_gnn = x_gnn + x_g
            # edge_attr = edge_attr + e_attr

        pos_edge_attr = pos_edge_attr.view(-1, self.edge_dim)
        pos_edge_attr = self.edge_emb(pos_edge_attr)
        neg_edge_attr, _ = self.encoder(neg_edge_attr)
        neg_edge_attr = neg_edge_attr.view(-1, self.edge_dim)
        neg_edge_attr = self.edge_emb(neg_edge_attr)
        if self.pretrain:
            out = self.decoder(x_tab[:, 0, :], x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        else:
            out = self.decoder(x)
        return out

class FTTransformerPNAFusedLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128, final_dropout: float = 0.5, deg=None):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden
        # fused_dim = channels + nhidden
        fused_dim = channels + 2*nhidden

        # fttransformer
        self.tab_conv = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=feedforward_channels or channels,
            dropout=dropout,
            activation=activation,
            # Input and output tensors are provided as
            # [batch_size, seq_len, channels]
            batch_first=True,
        )
        self.tab_norm = LayerNorm(channels)

        # PNA
        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.gnn_conv = PNAConv(in_channels=nhidden, 
                                out_channels=nhidden, 
                                aggregators=aggregators, 
                                scalers=scalers, 
                                edge_dim=nhidden, 
                                deg=deg)

        self.gnn_norm = BatchNorm(nhidden)
        # self.edge = Sequential(
        #     Linear(3 * nhidden, nhidden),
        #     ReLU(),
        #     Linear(nhidden, nhidden),
        # )

        # fuse
        self.fuse = Sequential(
            LayerNorm(fused_dim),
            Linear(fused_dim, fused_dim), 
            LeakyReLU(), Dropout(final_dropout), 
            Linear(fused_dim, fused_dim), LeakyReLU(), 
            Dropout(final_dropout),
            Linear(fused_dim, fused_dim)
        )
        self.fuse_norm = LayerNorm(fused_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()
        self.gnn_conv.reset_parameters()
        # for p in self.edge.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        self.gnn_norm.reset_parameters()
        for p in self.fuse.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    # def forward(self, x_tab, x_gnn, edge_index, edge_attr):
    #     x_tab = self.tab_norm(self.tab_conv(x_tab))
    #     x_tab_cls, x_tab = x_tab[:, 0, :], x_tab[:, 1:, :]

    #     x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
    #     edge_attr = edge_attr + self.edge(torch.cat([x_gnn[edge_index[0]], x_gnn[edge_index[1]], edge_attr], dim=-1)) / 2

    #     # fuse node interaction with pooled row embeddings
    #     # x_tab_cls_m = torch.mean(x_tab[:, 0, :], dim=0).unsqueeze(0).flatten()
    #     # x_gnn_int, x_gnn = x_gnn[0, :], x_gnn[1:, :]
    #     # x = torch.cat([x_tab_cls_m, x_gnn_int], dim=-1)
    #     # x = (x + self.fuse(x)) / 2
    #     # x_tab = torch.cat([(x[:self.channels].unsqueeze(0) + x_tab_cls / 2).unsqueeze(1), x_tab], dim=1) 
    #     # x_gnn = torch.cat([(x_gnn_int + x[self.channels:] / 2).unsqueeze(0), x_gnn])

    #     #int_attr, seed_attr, sampled_attr = edge_attr[0,:],  edge_attr[1:1+x_tab_cls.shape[0],:], edge_attr[1+x_tab_cls.shape[0]:,:]
    #     seed_attr, sampled_attr = edge_attr[:x_tab_cls.shape[0],:], edge_attr[x_tab_cls.shape[0]:,:]
    #     x = torch.cat([x_tab_cls, seed_attr], dim=-1)
    #     x = (x + self.fuse(x)) / 2
    #     x_tab = torch.cat([x[:,:self.channels].unsqueeze(1), x_tab], dim=1)
    #     #edge_attr = torch.cat([int_attr.unsqueeze(0), x[:,self.channels:], sampled_attr], dim=0)
    #     edge_attr = torch.cat([x[:,self.channels:], sampled_attr], dim=0)

    #     return x_tab, x_gnn, edge_attr

    def forward(self, x_tab, x_gnn, edge_index, edge_attr):
        x_tab = self.tab_norm(self.tab_conv(x_tab))
        x_tab_cls, x_tab = x_tab[:, 0, :], x_tab[:, 1:, :]

        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
        x_src_gnn = x_gnn[edge_index[0][0:x_tab_cls.shape[0]]]
        x_dst_gnn = x_gnn[edge_index[1][0:x_tab_cls.shape[0]]]

        x = torch.cat([x_tab_cls, x_src_gnn, x_dst_gnn], dim=-1)
        x = (x + self.fuse_norm(self.fuse(x))) / 2

        x_tab = torch.cat([x[:,:self.channels].unsqueeze(1), x_tab], dim=1)
        x_src_gnn = x[:, self.channels:self.channels+self.nhidden]
        x_dst_gnn = x[:, self.channels+self.nhidden:]
        x_gnn[edge_index[0][0:x_tab_cls.shape[0]]] = x_src_gnn
        x_gnn[edge_index[1][0:x_tab_cls.shape[0]]] = x_dst_gnn

        return x_tab, x_gnn, edge_attr
    
class FTTransformerPNAParallelLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128, final_dropout: float = 0.5, deg=None):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden

        # fttransformer
        self.tab_conv = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=feedforward_channels or channels,
            dropout=dropout,
            activation=activation,
            # Input and output tensors are provided as
            # [batch_size, seq_len, channels]
            batch_first=True,
        )
        self.tab_norm = LayerNorm(channels)

        # PNA
        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.gnn_conv = PNAConv(in_channels=nhidden, 
                                out_channels=nhidden, 
                                aggregators=aggregators, 
                                scalers=scalers, 
                                edge_dim=nhidden, 
                                deg=deg)

        self.gnn_norm = BatchNorm(nhidden)
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()

    def forward(self, x_tab, x_gnn, edge_index, edge_attr):
        x_tab = self.tab_norm(self.tab_conv(x_tab))
        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
        return x_tab, x_gnn, edge_attr
