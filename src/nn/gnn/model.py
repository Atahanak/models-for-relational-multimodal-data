"""GNN model class definitions."""
import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing

from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)

class GINEConvHetero(nn.Module):
    def __init__(self, network, n_hidden):
        super().__init__()
        self.conv_forw = GINEConv(network, edge_dim=n_hidden)
        self.conv_back = GINEConv(network, edge_dim=n_hidden)

        self.lin = Linear(n_hidden*3, n_hidden)\
    
    def reset_parameters(self):
        self.conv_forw.reset_parameters()
        self.conv_back.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        edge_index_forw, edge_index_back = edge_index, edge_index.flipud()

        a_in  = self.conv_forw((x, None), edge_index_forw, edge_attr)
        a_out = self.conv_back((x, None), edge_index_back, edge_attr)

        return self.lin(torch.cat([x, a_in, a_out], dim=1))

class PNAConvHetero(nn.Module):
    def __init__(self, n_hidden, in_channels, out_channels,
                            aggregators, scalers, deg,
                            edge_dim, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False):
        super().__init__()
        self.conv_forw = PNAConv(in_channels=in_channels, out_channels=out_channels,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=edge_dim, towers=towers, pre_layers=pre_layers, post_layers=post_layers,
                            divide_input=divide_input)
        self.conv_back = PNAConv(in_channels=in_channels, out_channels=out_channels,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=edge_dim, towers=towers, pre_layers=pre_layers, post_layers=post_layers,
                            divide_input=divide_input)

        self.lin = Linear(n_hidden*3, n_hidden)
    
    def reset_parameters(self):
        self.conv_forw.reset_parameters()
        self.conv_back.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        edge_index_forw, edge_index_back = edge_index, edge_index.flipud()

        a_in  = self.conv_forw(x, edge_index_forw, edge_attr)
        a_out = self.conv_back(x, edge_index_back, edge_attr)

        return self.lin(torch.cat([x, a_in, a_out], dim=1))


class GINe(torch.nn.Module):
    def __init__(self, 
                num_features=1, 
                num_gnn_layers=2,
                n_hidden=100, 
                edge_updates=False, 
                edge_dim=None,
                reverse_mp=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.reverse_mp = reverse_mp

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            
            if self.reverse_mp:
                conv = GINEConvHetero(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(self.n_hidden, self.n_hidden)
                    ), n_hidden=self.n_hidden)
            else:
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(self.n_hidden, self.n_hidden)
                    ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))


    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr.view(edge_attr.shape[0], -1))

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        return x, edge_attr
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            if self.edge_updates: 
                for emlp in self.emlps:
                    emlp.reset_parameters()
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()

class PNAS(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=128, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None, reverse_mp=False):
        super().__init__()
        #n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.reverse_mp = reverse_mp

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if self.reverse_mp:
                conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=n_hidden, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False)
            else:
                conv = PNAConvHetero(n_hidden=n_hidden, in_channels=n_hidden, out_channels=n_hidden,
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

        # self.decoder = LinkPredHead(n_classes=n_classes, n_hidden=n_hidden, dropout=final_dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr.view(edge_attr.shape[0], -1))

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                src, dst = edge_index
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        return x, edge_attr
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=128, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None, encoder=None, reverse_mp=False):
        super().__init__()
        #n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.encoder = encoder
        self.reverse_mp = reverse_mp

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if self.reverse_mp:
                conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=n_hidden, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False)
            else:
                conv = PNAConvHetero(n_hidden=n_hidden, in_channels=n_hidden, out_channels=n_hidden,
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

        # self.decoder = LinkPredHead(n_classes=n_classes, n_hidden=n_hidden, dropout=final_dropout)

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

        return x, pos_edge_attr, neg_edge_attr
        #return self.decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
    
    def loss_fn(self, input1, input2):
        # input 1 is pos_preds and input_2 is neg_preds
        return -torch.log(input1 + 1e-15).mean() - torch.log(1 - input2 + 1e-15).mean()
