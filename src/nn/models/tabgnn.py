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

from torch_geometric.nn import PNAConv
from torch_geometric.nn import BatchNorm

import logging
logger = logging.getLogger(__name__)

class TABGNN(Module):
    r"""
    """
    def __init__(
        self,
        # general parameters
        channels: int,
        num_layers: int,
        encoder: Any,
        # PNA parameters
        deg=None,
        node_dim: int = 1,
        nhidden: int = 128,
        edge_dim: int = None,
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
        self.encoder = encoder
        self.channels = channels
        self.nhidden = nhidden
        self.edge_dim = edge_dim + channels

        self.cls_embedding = Parameter(torch.empty(channels))
        
        self.node_emb = Linear(node_dim, nhidden)
        self.edge_emb = Linear(self.edge_dim, nhidden)

        self.tabular_backbone = ModuleList()
        self.gnn_backbone = ModuleList()
        for i in range(num_layers):
            self.tabular_backbone.append(FTTransformerLayer(channels, nhead, feedforward_channels, 0.2, activation, nhidden))
            self.tabular_backbone.append(FTTransformerLayer(channels, nhead, feedforward_channels, dropout, activation, nhidden))
            self.gnn_backbone.append(PNALayer(channels, nhidden, deg))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        for layer in self.tabular_backbone:
            layer.reset_parameters()
        for layer in self.gnn_backbone:
            layer.reset_parameters()

    def get_shared_params(self):
        param_groups = [
            self.encoder.parameters(),
            [self.cls_embedding],  # Wrap single parameters in a list
            self.node_emb.parameters(),
            self.edge_emb.parameters(),
            self.tabular_backbone.parameters(),
            self.gnn_backbone.parameters(),
        ]
        
        # Flatten the param groups into a single list
        flat_params = [param for group in param_groups for param in group]
        
        return flat_params

    def zero_grad_shared_params(self):
        for param in self.get_shared_params():
            if param.grad is not None:
                param.grad.data.zero_()

    def forward(self, x, edge_index, edge_attr, target_edge_attr) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        B = len(target_edge_attr)
        N = len(edge_attr)

        x = self.node_emb(x)
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        target_edge_attr = torch.cat([x_cls, target_edge_attr], dim=1)
        x_cls = self.cls_embedding.repeat(N, 1, 1)
        edge_attr = torch.cat([x_cls, edge_attr], dim=1)

        for layer in self.tabular_backbone:
            t_edge_attr = layer(edge_attr)
            t_target_edge_attr = layer(target_edge_attr)
        
        edge_attr = (edge_attr + t_edge_attr) / 2
        target_edge_attr = (target_edge_attr + t_target_edge_attr) / 2
        
        target_edge_attr = target_edge_attr.view(-1, self.edge_dim)
        target_edge_attr = self.edge_emb(target_edge_attr)
        edge_attr = edge_attr.view(-1, self.edge_dim)
        edge_attr = self.edge_emb(edge_attr)

        for layer in self.gnn_backbone:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr, target_edge_attr

class PNALayer(Module):
    def __init__(self, channels: int, nhidden: int = 128, deg=None):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden

        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.gnn_conv = PNAConv(in_channels=nhidden, 
                                out_channels=nhidden, 
                                aggregators=aggregators, 
                                scalers=scalers, 
                                edge_dim=nhidden, 
                                deg=deg,
                                towers=1
        )
        self.gnn_norm = BatchNorm(nhidden)
        self.gnn_edge_update = Sequential(
                Linear(3 * self.nhidden, self.nhidden),
                ReLU(),
                Linear(self.nhidden, self.nhidden),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()

    def forward(self, x_gnn, edge_index, edge_attr):
        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
        src, dst = edge_index
        edge_attr = edge_attr + self.gnn_edge_update(torch.cat([x_gnn[src], x_gnn[dst], edge_attr], dim=-1)) / 2
        return x_gnn, edge_attr
    
class FTTransformerLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden

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
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()

    def forward(self, x_tab):
       return (x_tab + self.tab_norm(self.tab_conv(x_tab))) / 2
