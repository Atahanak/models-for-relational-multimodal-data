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
            
    def forward(self, x):
        return torch.sigmoid(self.mlp(x))