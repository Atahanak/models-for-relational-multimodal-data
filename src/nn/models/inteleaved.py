from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import (
    LayerNorm, 
    Linear,
    Module, 
    ReLU, 
    Sequential, 
    ModuleList,
    Parameter,
    TransformerEncoderLayer,
)

from ..gnn.pna import PNAConvHetero
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from torch_geometric.nn import PNAConv
from torch_geometric.nn import BatchNorm

import logging
logger = logging.getLogger(__name__)

class TABGNNInterleaved(Module):
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
    """
    def __init__(
        self,
        # general parameters
        channels: int,
        num_layers: int,
        encoder: StypeWiseFeatureEncoder = None,
        # PNA parameters
        deg=None,
        node_dim: int = 1,
        nhidden: int = 128,
        edge_dim: int = None,
        reverse_mp: bool = False,
        # fttransformer parameters
        feedforward_channels: Optional[int] = None,
        nhead: int = 8,
        dropout: float = 0.5,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        
        self.channels = channels
        self.nhidden = nhidden
        self.node_dim = node_dim
        self.edge_dim = edge_dim + channels
        self.encoder = encoder
        self.reverse_mp = reverse_mp

        # fttransformer
        self.cls_embedding = Parameter(torch.empty(channels))
        
        # PNA
        self.node_emb = Linear(node_dim, nhidden)
        self.edge_emb = Linear(self.edge_dim, nhidden)
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

        self.backbone = ModuleList()
        for i in range(num_layers):
            self.backbone.append(
                FTTransformerPNAInterleavedLayer(
                    channels, 
                    nhead, 
                    feedforward_channels, 
                    dropout, 
                    activation, 
                    nhidden,
                    deg,
                    self.reverse_mp
                )
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.tab_norm.reset_parameters()
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        # self.encoder.reset_parameters()
        for layer in self.backbone:
            layer.reset_parameters()

    def get_shared_params(self):
        param_groups = [
            self.encoder.parameters(),
            self.tab_conv.parameters(),
            self.tab_norm.parameters(),
            [self.cls_embedding],  # Wrap single parameters in a list
            self.node_emb.parameters(),
            self.edge_emb.parameters(),
            self.backbone.parameters(),
        ]
        
        # Flatten the param groups into a single list
        flat_params = [param for group in param_groups for param in group]
        
        return flat_params

    def zero_grad_shared_params(self):
        for param in self.get_shared_params():
            if param.grad is not None:
                param.grad.data.zero_()

    def forward(self, x, edge_index, edge_attr) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x_gnn = self.node_emb(x.view(-1, self.node_dim))

        x_edge = self.cls_embedding.repeat(edge_index.shape[1], 1, 1)
        edge_attr = torch.cat([x_edge, edge_attr], dim=1)
        edge_attr = (edge_attr + self.tab_norm(self.tab_conv(edge_attr))) / 2

        e_attr = edge_attr
        for layer in self.backbone:
            x_gnn, e_attr = layer(x_gnn, edge_index, e_attr)
        
        edge_attr = (e_attr + edge_attr) / 2
        x_edge, _ = edge_attr[:, 0, :], edge_attr[:, 1:, :]
        return x_gnn, x_edge

class FTTransformerPNAInterleavedLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.5, activation: str = 'relu', nhidden: int = 128, deg=None, reverse_mp: bool = False) -> None:
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

        if not reverse_mp:
            self.gnn_conv = PNAConv(in_channels=nhidden, out_channels=nhidden,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=nhidden, towers=1, pre_layers=1, post_layers=1,
                        divide_input=False)
        else:
            self.gnn_conv = PNAConvHetero(n_hidden=nhidden, in_channels=nhidden, out_channels=nhidden,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=nhidden, towers=1, pre_layers=1, post_layers=1,
                        divide_input=False)

        self.gnn_norm = BatchNorm(nhidden)
        self.gnn_edge_update = Sequential(
                Linear(3 * self.nhidden, self.nhidden),
                ReLU(),
                Linear(self.nhidden, self.nhidden),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()
        for p in self.gnn_edge_update.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x_gnn, edge_index, edge_attr):
        edge_attr = (edge_attr + self.tab_norm(self.tab_conv(edge_attr)) / 2)
        edge_attr_cls, edge_attr_feat = edge_attr[:, 0, :], edge_attr[:, 1:, :]

        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr_cls)))) / 2
        
        src, dst = edge_index
        edge_attr_cls = (edge_attr_cls + self.gnn_edge_update(torch.cat([x_gnn[src], x_gnn[dst], edge_attr_cls], dim=-1))) / 2
        edge_attr = torch.cat([edge_attr_cls.unsqueeze(1), edge_attr_feat], dim=1)

        return x_gnn, edge_attr