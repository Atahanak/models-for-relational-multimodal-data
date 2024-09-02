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

from ..gnn.model import PNAConvHetero

import logging
logger = logging.getLogger(__name__)

class TABGNN2(Module):
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
        reverse_mp: bool = False,
        # fttransformer parameters
        feedforward_channels: Optional[int] = None,
        nhead: int = 8,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        self.encoder = encoder
        self.channels = channels
        self.nhidden = self.channels
        #self.nhidden = nhidden
        self.edge_dim = edge_dim
        self.ff_dim = 2048 #feedforward_channels if feedforward_channels is not None else 4*channels
        
        self.node_emb = Linear(node_dim, self.nhidden)
        self.edge_emb = Linear(self.edge_dim, self.nhidden)

        self.tabular_backbone = ModuleList()
        self.gnn_backbone = ModuleList()
        for i in range(num_layers):
            #self.tabular_backbone.append(FTTransformerLayer(channels, nhead, feedforward_channels, dropout, activation, nhidden))
            self.tabular_backbone.append(RCBAttentionLayer(channels, nhead, self.ff_dim, dropout, activation))
            self.gnn_backbone.append(PNALayer(channels, self.nhidden, deg, reverse_mp))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #torch.nn.init.normal_(self.cls_embedding, std=0.01)
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        for layer in self.tabular_backbone:
            layer.reset_parameters()
        for layer in self.gnn_backbone:
            layer.reset_parameters()

    def get_shared_params(self):
        param_groups = [
            self.encoder.parameters(),
            #[self.cls_embedding],  # Wrap single parameters in a list
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

    def forward(self, x, edge_index, edge_attr, mask) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x = self.node_emb(x)
        t_edge_attr = edge_attr
        for layer in self.tabular_backbone:
            t_edge_attr = layer(t_edge_attr, mask)
        
        edge_attr = (edge_attr + t_edge_attr) / 2
        edge_attr = edge_attr[mask]
        edge_attr = edge_attr.view(-1, self.edge_dim)
        edge_attr = self.edge_emb(edge_attr)

        for layer in self.gnn_backbone:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr

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
        self.encoder = encoder
        self.channels = channels
        # self.nhidden = nhidden
        self.nhidden = channels
        self.edge_dim = edge_dim + channels
        self.ff_dim = 2048 #feedforward_channels if feedforward_channels is not None else 4*channels

        self.cls_embedding = Parameter(torch.empty(channels))
        
        self.node_emb = Linear(node_dim, self.nhidden)
        self.edge_emb = Linear(self.edge_dim, self.nhidden)

        self.tabular_backbone = ModuleList()
        self.gnn_backbone = ModuleList()
        for i in range(num_layers):
            self.tabular_backbone.append(FTTransformerLayer(channels, nhead, self.ff_dim, dropout, activation, self.nhidden))
            #self.tabular_backbone.append(RCAttentionLayer(channels, nhead, self.ff_dim, dropout, activation))
            #self.tabular_backbone.append(RCAttentionLayer(channels, nhead))
            self.gnn_backbone.append(PNALayer(channels, self.nhidden, deg, reverse_mp))
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

        t_edge_attr = edge_attr
        t_target_edge_attr = target_edge_attr
        for layer in self.tabular_backbone:
            t_edge_attr = layer(t_edge_attr)
            t_target_edge_attr = layer(t_target_edge_attr)
        
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
    def __init__(self, channels: int, nhidden: int = 128, deg=None, reverse_mp: bool = False):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden
        self.reverse_mp = reverse_mp

        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        if not self.reverse_mp:
            self.gnn_conv = PNAConv(in_channels=self.nhidden, out_channels=self.nhidden,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=self.nhidden, towers=1, pre_layers=1, post_layers=1,
                        divide_input=False)
        else:
            self.gnn_conv = PNAConvHetero(n_hidden=self.nhidden, in_channels=self.nhidden, out_channels=self.nhidden,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=self.nhidden, towers=1, pre_layers=1, post_layers=1,
                        divide_input=False)
        self.gnn_norm = BatchNorm(self.nhidden)
        self.gnn_edge_update = Sequential(
                Linear(3 * self.nhidden, self.nhidden),
                ReLU(),
                Linear(self.nhidden, self.nhidden),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()
        for p in self.gnn_edge_update.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x_gnn, edge_index, edge_attr):
        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
        src, dst = edge_index
        edge_attr = edge_attr + self.gnn_edge_update(torch.cat([x_gnn[src], x_gnn[dst], edge_attr], dim=-1)) / 2
        return x_gnn, edge_attr
    
class RCBAttentionLayer(Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1, activation='relu'):
        super(RCBAttentionLayer, self).__init__()
        self.row_transformer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation=activation,  batch_first=True)
        self.row_norm = LayerNorm(hidden_dim)
        self.col_transformer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation=activation,  batch_first=True)
        self.col_norm = LayerNorm(hidden_dim)
    
    def reset_parameters(self):
        for p in self.row_transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.row_norm.reset_parameters()
        for p in self.col_transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.col_norm.reset_parameters()

    def forward(self, x, mask):
        # x: (batch_size, num_rows, num_cols, hidden_dim)
        batch_size, num_rows, num_cols, hidden_dim = x.shape
        col_mask = mask.unsqueeze(1).expand(batch_size, num_cols, num_rows).contiguous().view(batch_size * num_cols, num_rows).float()
        # convert 1 to 0 and 0 to -inf
        col_mask = (1 - col_mask) * -1e9
        #row_mask = mask.unsqueeze(2).expand(batch_size, num_rows, num_cols).contiguous().view(batch_size * num_rows, num_cols).float()

        # Apply row-wise attention
        row_input = x.view(batch_size * num_rows, num_cols, hidden_dim)
        # print(row_input)
        row_output = self.row_transformer(row_input) #, src_key_padding_mask=row_mask)  # (batch_size * num_rows, num_cols, hidden_dim)
        row_output = row_output.view(batch_size, num_rows, num_cols, hidden_dim)
        # print(row_output)

        # Apply column-wise attention
        col_input = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_cols, num_rows, hidden_dim)
        # # mask: (batch_size, num_rows)
        col_input = col_input.view(batch_size * num_cols, num_rows, hidden_dim)
        col_output = self.col_transformer(col_input, src_key_padding_mask=col_mask)  # (batch_size * num_cols, num_rows, hidden_dim)
        #col_output = self.col_transformer(col_input)  # (batch_size * num_cols, num_rows, hidden_dim)
        col_output = col_output.view(batch_size, num_cols, num_rows, hidden_dim)
        col_output = col_output.permute(0, 2, 1, 3).contiguous()  # Back to (batch_size, num_rows, num_cols, hidden_dim)
        # print(col_output)
        # import sys
        # sys.exit()
        # Combine row-wise and column-wise outputs (e.g., summation)
        combined_output = (self.row_norm(row_output) + self.col_norm(col_output)) / 2
        return combined_output

class RCAttentionLayer(Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1, activation='relu'):
        super(RCAttentionLayer, self).__init__()
        self.row_transformer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation=activation, batch_first=True)
        self.row_norm = LayerNorm(hidden_dim)
        self.col_transformer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation=activation, batch_first=True)
        self.col_norm = LayerNorm(hidden_dim)
    
    def reset_parameters(self):
        for p in self.row_transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.row_norm.reset_parameters()
        for p in self.col_transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.col_norm.reset_parameters()

    def forward(self, x):
        # x: (num_rows, num_cols, hidden_dim)
        # Apply row-wise attention
        row_input = x
        row_output = self.row_transformer(row_input)

        # Apply column-wise attention
        col_input = x.permute(1, 0, 2).contiguous()  # (num_cols, num_rows, hidden_dim)
        col_output = self.col_transformer(col_input)
        col_output = col_output.permute(1, 0, 2).contiguous()  # Back to (num_rows, num_cols, hidden_dim)

        # Combine row-wise and column-wise outputs (e.g., summation)
        combined_output = (self.row_norm(row_output) + self.col_norm(col_output)) / 2
        return combined_output

class FTTransformerLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.1, activation: str = 'relu', nhidden: int = 128):
        super().__init__()
        self.channels = channels
        self.nhidden = nhidden

        self.tab_conv = TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=feedforward_channels,
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
