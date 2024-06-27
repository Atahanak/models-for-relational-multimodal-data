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

import logging
logger = logging.getLogger(__name__)
import sys

class New(Module):
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
        encoder: StypeWiseFeatureEncoder = None,
        
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
        
        if encoder is None:
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
        else:
            self.encoder = encoder

        # fttransformer
        self.cls_embedding = Parameter(torch.empty(channels))
        
        # PNA
        self.node_emb = Linear(node_dim, nhidden)
        self.edge_emb = Linear(edge_dim, nhidden)
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

        #backbone
        self.backbone = ModuleList()
        for i in range(num_layers):
            #if i == (num_layers - 1):
            if True:
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.tab_norm.reset_parameters()
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.encoder.reset_parameters()
        for layer in self.backbone:
            layer.reset_parameters()
        #self.decoder.reset_parameters()

    def get_shared_params(self):
        param_groups = [
            self.encoder.parameters(),
            [self.cls_embedding],  # Wrap single parameters in a list
            self.node_emb.parameters(),
            #self.edge_emb.parameters(),
            self.backbone[:-1].parameters(),
        ]
        
        # Flatten the param groups into a single list
        flat_params = [param for group in param_groups for param in group]
        
        return flat_params

    def zero_grad_shared_params(self):
        for param in self.get_shared_params():
            if param.grad is not None:
                param.grad.data.zero_()

    def forward(self, x, edge_index, edge_attr, target_edge_index, target_edge_attr, lp=False) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        B = len(target_edge_attr)

        x_gnn = self.node_emb(x)
        # edge_attr, _ = self.encoder(edge_attr)
        # target_edge_attr, _ = self.encoder(target_edge_attr)
        
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        x_tab = torch.cat([x_cls, target_edge_attr], dim=1)
        x_tab = self.tab_norm(self.tab_conv(x_tab))

        x_edge = self.cls_embedding.repeat(edge_index.shape[1], 1, 1)
        edge_attr = torch.cat([x_edge, edge_attr], dim=1)
        edge_attr = self.tab_norm(self.tab_conv(edge_attr))
        # edge_attr, _ = edge_attr[:, 0, :], edge_attr[:, 1:, :]

        _, edge_attr = edge_attr[:, 0, :], edge_attr[:, 1:, :]
        edge_attr = edge_attr.view(-1, self.edge_dim)
        edge_attr = self.edge_emb(edge_attr)

        for layer in self.backbone:
            x_t = x_tab
            x_g = x_gnn
            x_tab, x_gnn, edge_attr = layer(x_tab, x_gnn, edge_index, edge_attr, target_edge_index, lp)
            x_tab = (x_tab + x_t)/2
            x_gnn = (x_gnn + x_g)/2


        # x_edge = self.cls_embedding.repeat(target_edge_index.shape[1], 1, 1)
        # target_edge_attr = torch.cat([x_edge, target_edge_attr], dim=1)
        # target_edge_attr = self.tab_norm(self.tab_conv(target_edge_attr))
        # # target_edge_attr, _ = target_edge_attr[:, 0, :], target_edge_attr[:, 1:, :]
        # _, target_edge_attr = target_edge_attr[:, 0, :], target_edge_attr[:, 1:, :]
        
        # target_edge_attr = target_edge_attr.view(-1, self.edge_dim)
        # target_edge_attr = self.edge_emb(target_edge_attr)
        # return target_edge_attr, x_gnn
        # x_tab, _ = x_tab[:, 0, :], x_tab[:, 1:, :]
        _, x_tab = x_tab[:, 0, :], x_tab[:, 1:, :]
        x_tab = x_tab.view(-1, self.edge_dim)
        x_tab = self.edge_emb(x_tab)
        return x_tab, x_gnn

class FTTransformerPNAFusedLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128, final_dropout: float = 0.5, deg=None):
        super().__init__()
        self.channels = channels
        #nhidden = int((nhidden // 5) * 5)
        self.nhidden = nhidden
        #fused_dim = channels + nhidden
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
                                deg=deg,
                                towers=1
        )
        self.gnn_norm = BatchNorm(nhidden)
        self.gnn_edge_update = Sequential(
                Linear(3 * self.nhidden, self.nhidden),
                ReLU(),
                Linear(self.nhidden, self.nhidden),
        )

        # fuse
        self.fuse = Sequential(
            LayerNorm(fused_dim),
            Linear(fused_dim, 4*fused_dim), 
            LeakyReLU(), Dropout(final_dropout), 
            Linear(4*fused_dim, 4*fused_dim), LeakyReLU(), 
            Dropout(final_dropout),
            Linear(4*fused_dim, fused_dim)
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

    def forward(self, x_tab, x_gnn, edge_index, edge_attr, target_edge_index, lp=False):
        x_tab = self.tab_norm(self.tab_conv(x_tab))
        x_tab_cls, x_tab_feat = x_tab[:, 0, :], x_tab[:, 1:, :]

        x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2

        # edge update
        src, dst = edge_index
        edge_attr = edge_attr + self.gnn_edge_update(torch.cat([x_gnn[src], x_gnn[dst], edge_attr], dim=-1)) / 2
        #logger.info(f"x_gnn {x_gnn.shape}")

        if not lp:
            x = torch.cat([x_tab_cls, x_gnn[target_edge_index[0]], x_gnn[target_edge_index[1]]], dim=-1)
            x = (x + self.fuse_norm(self.fuse(x))) / 2

            x_tab_cls = (x_tab_cls + x[:,:self.channels]) / 2
            x_tab = torch.cat([x_tab_cls.unsqueeze(1), x_tab_feat], dim=1)

            # TODO: pooling
            index = target_edge_index.flatten()
            logger.info(f"index {index.shape}")
            emb = torch.cat([x[:, self.channels:self.channels+self.nhidden], x[:, self.channels+self.nhidden:]], dim=0)
            logger.info(f"emb {emb.shape}")
            # Identify the unique indices and their corresponding inverse indices
            unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
            logger.info(f"uniq {unique_indices.shape}, inverse {inverse_indices.shape}")
            # Create a tensor of zeros to accumulate the sums
            summed_emb = torch.zeros((unique_indices.size(0), emb.size(1)), dtype=torch.float, device=emb.device)
            logger.info(f"summed {summed_emb.shape}")
            # Use scatter_add to sum the vectors corresponding to the same indices
            summed_emb.index_add_(0, inverse_indices, emb)
            logger.info(f"summed {summed_emb.shape}")
            # Count the occurrences of each index
            counts = torch.bincount(inverse_indices)
            logger.info(f"counts {counts.shape}")
            # Divide the summed vectors by their counts to get the average
            pooled_emb = summed_emb / counts.unsqueeze(1).float()
            logger.info(f"pooled {pooled_emb.shape}")
            sys.exit()

            x_gnn[unique_indeces] = (x_gnn[unique_indeces] + pooled_emb) / 2
        return x_tab, x_gnn, edge_attr
    
class FTTransformerPNAParallelLayer(Module):
    def __init__(self, channels: int, nhead: int, feedforward_channels: Optional[int] = None, dropout: float = 0.2, activation: str = 'relu', nhidden: int = 128, final_dropout: float = 0.5, deg=None):
        super().__init__()
        self.channels = channels
        #nhidden = int((nhidden // 5) * 5)
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
        for p in self.tab_conv.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.tab_norm.reset_parameters()
        self.gnn_conv.reset_parameters()
        self.gnn_norm.reset_parameters()

    def forward(self, x_tab, x_gnn, edge_index, edge_attr, target_edge_index=None, lp=False):
       x_tab = self.tab_norm(self.tab_conv(x_tab))
       x_gnn = (x_gnn + F.relu(self.gnn_norm(self.gnn_conv(x_gnn, edge_index, edge_attr)))) / 2
       src, dst = edge_index
       edge_attr = edge_attr + self.gnn_edge_update(torch.cat([x_gnn[src], x_gnn[dst], edge_attr], dim=-1)) / 2
       return x_tab, x_gnn, edge_attr


"""GNN model class definitions."""
import torch.nn as nn
from torch_geometric.nn import BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch
from ..gnn.decoder import LinkPredHead
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=128, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None, encoder=None):
        super().__init__()
        # n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.edge_dim = edge_dim

        self.encoder = encoder

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
        #                       Linear(25, n_classes))
        self.decoder = LinkPredHead(n_classes=n_classes, n_hidden=n_hidden, dropout=final_dropout)
        logger.info(f"decoder n_classes: {n_classes}, channels: {n_hidden}, dropout: {final_dropout}")

    # def forward(self, x, edge_index, edge_attr, target_edge_index, target_edge_attr):
    #     x = self.node_emb(x)
    #     #edge_attr, _ = self.encoder(edge_attr)
    #     #edge_attr = edge_attr.view(-1, self.edge_dim) 
    #     edge_attr = self.edge_emb(edge_attr)

    #     # target_edge_attr, _ = self.encoder(target_edge_attr)
    #     # target_edge_attr = target_edge_attr.view(-1, self.edge_dim) 
    #     target_edge_attr = self.edge_emb(target_edge_attr)

    #     for i in range(self.num_gnn_layers):
    #         x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
    #         if self.edge_updates: 
    #             src, dst = edge_index
    #             edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

    #     return target_edge_attr, x
    
        
    def forward(self, x, edge_index, edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        pos_edge_attr = self.edge_emb(pos_edge_attr)
        neg_edge_attr = self.edge_emb(neg_edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                src, dst = edge_index
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        # return self.mlp(out)
        return x, pos_edge_attr, neg_edge_attr
        #return self.decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
    
    def loss_fn(self, input1, input2):
        # input 1 is pos_preds and input_2 is neg_preds
        return -torch.log(input1 + 1e-15).mean() - torch.log(1 - input2 + 1e-15).mean()

