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

from torch_geometric.nn import GINEConv
from torch_geometric.nn import BatchNorm

from ..decoder import SelfSupervisedLPHead
from ..decoder import SupervisedHead

from icecream import ic
import sys

class FTTransformerGINeFused(Module):
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
        pretrain: bool = False,
        
        # GINe parameters
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
        
        if pretrain:
            num_numerical = len(col_names_dict[stype.numerical])
            num_categorical = [len(col_stats[col][StatType.COUNT][0]) for col in col_names_dict[stype.categorical]]

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
        
        # GINe
        self.node_emb = Linear(node_dim, nhidden)
        self.edge_emb = Linear(edge_dim, nhidden)

        #backbone
        self.backbone = ModuleList()
        for _ in range(num_layers):
            self.backbone.append(
                FTTransformerGINeFusedLayer(
                    channels, 
                    nhead, 
                    feedforward_channels, 
                    dropout, 
                    activation, 
                    nhidden,
                    final_dropout
                )
            )

        if pretrain:
            num_numerical = len(col_names_dict[stype.numerical])
            num_categorical = [len(col_stats[col][StatType.COUNT][0]) for col in col_names_dict[stype.categorical]]
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

    def forward(self, tf: TensorFrame, x, edge_index, edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        B = tf.num_rows
        x_gnn = self.node_emb(x)
        #ic(x_gnn.shape)
        #edge_attr = self.edge_emb(self.encoder(edge_attr))
        edge_attr = self.edge_emb(edge_attr)
        #ic(edge_attr.shape)

        x_tab, _ = self.encoder(tf)
        #ic(x_tab.shape)
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        # ic(x_cls[0])
        #ic(x_cls.shape)
        x_tab = torch.cat([x_cls, x_tab], dim=1)
        #ic(x_tab.shape)

        for fused_layer in self.backbone:
            # ic(x_tab[0, 0, :], x_gnn[0, :])
            # ic(x_tab[1, 0, :])
            # ic(x_tab[0, 1, :])
            # ic(x_tab[0, 2, :])
            x_tab, x_gnn = fused_layer(x_tab, x_gnn, edge_index, edge_attr)
            #ic(x_tab.shape, x_gnn.shape)

        # ic(x_tab.shape, x_gnn.shape)
        # ic(x_tab[0, 0, :], x_gnn[0, :])
        # sys.exit()
        #pos_edge_attr = self.edge_emb(self.encoder(pos_edge_attr))
        pos_edge_attr = self.edge_emb(pos_edge_attr)
        #ic(pos_edge_attr.shape)
        #neg_edge_attr = self.edge_emb(self.encoder(neg_edge_attr))
        neg_edge_attr = self.edge_emb(neg_edge_attr)
        #ic(neg_edge_attr.shape)
        if self.pretrain:
            out = self.decoder(x_tab[:, 0, :], x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        else:
            out = self.decoder(x)
        return out

class FTTransformerGINeFusedLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128, final_dropout: float = 0.5):
        super().__init__()
        self.channels = channels
        fused_dim = channels + nhidden

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

        # GINe
        self.gnn_conv= GINEConv(
            Sequential(
            Linear(nhidden, nhidden), 
            ReLU(), 
            Linear(nhidden, nhidden)
        ), edge_dim=nhidden)
        self.gnn_norm = BatchNorm(nhidden)

        # fuse
        self.fuse = Sequential(
            LayerNorm(fused_dim),
            Linear(fused_dim, fused_dim), 
            LeakyReLU(), Dropout(final_dropout), 
            Linear(fused_dim, fused_dim), LeakyReLU(), 
            Dropout(final_dropout),
            Linear(fused_dim, fused_dim)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()
        # for p in self.gnn_conv.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()
        for p in self.fuse.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x_tab, x_gnn, edge_index, edge_attr):
        #ic(x_tab.shape, x_gnn.shape, edge_index.shape, edge_attr.shape)
        x_tab = self.tab_norm(self.tab_conv(x_tab))
        #ic(x_tab.shape)

        x_tab_cls, x_tab = x_tab[:, 0, :], x_tab[:, 1:, :]
        x_tab_cls_m = torch.mean(x_tab[:, 0, :], dim=0).unsqueeze(0).flatten()
        #ic(x_tab_cls.shape)

        # check if x_gnn, edge_index and edge_attr has any NaN values
        # ic(torch.isnan(x_gnn).any() or torch.isnan(edge_index).any() or torch.isnan(edge_attr).any())
        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
        # ic(torch.isnan(x_gnn).any())
        #ic(x_gnn.shape)
        
        #x_gnn_seed, x_gnn_neigh = x_gnn[:x_tab.shape[0], :], x_gnn[x_tab.shape[0]:, :]
        #ic(x_gnn_seed.shape, x_gnn_neigh.shape)
        x_gnn_int, x_gnn = x_gnn[0, :], x_gnn[1:, :] 
        #ic(x_gnn_int.shape)

        x = torch.cat([x_tab_cls_m, x_gnn_int], dim=-1)
        #ic(x.shape)

        x = (x + self.fuse(x)) / 2
        #ic(x.shape)
        
        # ic(x[:self.channels].shape, x_tab_cls.shape, x_tab.shape, x_gnn_int.shape, x[self.channels:].shape, x_gnn.shape)
        x_tab = torch.cat([(x[:self.channels].unsqueeze(0) + x_tab_cls / 2).unsqueeze(1), x_tab], dim=1) 
        #ic(x_tab.shape)
        x_gnn = torch.cat([(x_gnn_int + x[self.channels:] / 2).unsqueeze(0), x_gnn])
        #ic(x_gnn.shape)
        return x_tab, x_gnn