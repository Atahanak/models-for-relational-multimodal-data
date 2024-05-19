"""GNN model class definitions."""
import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch

from .decoder import LinkPredHead


class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=128, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            if self.edge_updates: 
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # self.decoder = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
        #                       Linear(25, n_classes))
        self.decoder = LinkPredHead(n_classes=n_classes, n_hidden=n_hidden, final_dropout=final_dropout)

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

        #x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        #x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        #out = x
        
        return self.decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            if self.edge_updates: 
                for emlp in self.emlps:
                    emlp.reset_parameters()
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.decoder.reset_parameters()
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

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
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
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
        self.decoder = LinkPredHead(n_classes=n_classes, n_hidden=n_hidden, final_dropout=final_dropout)

    def forward(self, x, edge_index, edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        src, dst = edge_index

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
        return self.decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
    
    def loss_fn(self, input1, input2):
        # input 1 is pos_preds and input_2 is neg_preds
        return -torch.log(input1 + 1e-15).mean() - torch.log(1 - input2 + 1e-15).mean()