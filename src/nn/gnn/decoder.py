import torch
import torch.nn as nn
from torch_geometric.nn import Linear

class LinkPredHead(torch.nn.Module):
    """Readout head for link prediction.

    Parameters
    ----------
    config : GNNConfig
        Architecture configuration
    """
    def __init__(self, n_classes=1, n_hidden=100, final_dropout=0.5) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.final_dropout = final_dropout

        self.mlp = nn.Sequential(
            Linear(self.n_hidden*3, self.n_hidden), 
            nn.ReLU(), nn.Dropout(self.final_dropout), 
            Linear(self.n_hidden, 25), nn.ReLU(), 
            nn.Dropout(self.final_dropout),
            Linear(25, self.n_classes)
        )
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for p in self.mlp.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            
    def forward(self, x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        #reshape s.t. each row in x corresponds to the concatenated src and dst node features for each edge
        x_pos = x[pos_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x_neg = x[neg_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()

        #concatenate the node feature vector with the corresponding edge features
        x_pos = torch.cat((x_pos, pos_edge_attr.view(-1, pos_edge_attr.shape[1])), 1)
        x_neg = torch.cat((x_neg, neg_edge_attr.view(-1, neg_edge_attr.shape[1])), 1)

        return (torch.sigmoid(self.mlp(x_pos)), torch.sigmoid(self.mlp(x_neg)))