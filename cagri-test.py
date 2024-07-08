import os

from torch_frame.data import DataLoader
from torch_frame import stype, TensorFrame
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

import torch 
import torch.nn as nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML
from src.nn.gnn.model import GINe, PNA

import logging
import sys

def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )

# Create a logger
logger_setup()

batch_size = 256
channels = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = IBMTransactionsAML(
    root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv',
    split_type='temporal', 
    splits=[0.6, 0.2, 0.2], 
    khop_neighbors=[100, 100]
)
dataset.materialize()
train_dataset, val_dataset, test_dataset = dataset.split()


tensor_frame = dataset.tensor_frame
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=batch_size, shuffle=False, num_workers=4)

num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])

num_columns = num_numerical + num_categorical + 1
logging.info(f"num_numerical: {num_numerical}")
logging.info(f"num_categorical: {num_categorical}")
logging.info(f"num_columns: {num_columns}")


def graph_inputs(batch: TensorFrame, tensor_frame: TensorFrame):

    edges = batch.y[:,-3:]
    y = batch.y[:, 0]
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, 'train')
    edge_attr = tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    return node_feats, edge_index, edge_attr, y


class SupervisedTabGNN(nn.Module):
    def __init__(self, stype_encoder_dict):
        super().__init__()
        
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=channels,
                    col_stats=dataset.col_stats,
                    col_names_dict=dataset.tensor_frame.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
        ).to(device)

        self.graph_model = self.get_model()

    
    def forward(self, x, edge_index, edge_attr):
        
        edge_attr, _ = self.encoder(edge_attr)  


    

    
    def get_model(sample_batch, config, args):
        n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
        e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

        if args.model == "gin":
            model = GINe(
                    num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                    n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                    dropout=config.dropout, final_dropout=config.final_dropout
                    )
        elif args.model == "pna":
            if not isinstance(sample_batch, HeteroData):
                d = degree(sample_batch.edge_index[1], dtype=torch.long)
            else:
                index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
                d = degree(index, dtype=torch.long)
            deg = torch.bincount(d, minlength=1)
            model = PNA(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
                )
        
        return model


stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.timestamp: TimestampEncoder(),
    }



for batch in train_loader:
    batch_size = len(batch.y)
    node_feats, edge_index, edge_attr, y = graph_inputs(batch, tensor_frame)


def set_target_col(self: torch_frame.data.Dataset, pretrain: set[PretrainType],
                   col_to_stype: dict[str, torch_frame.stype], supervised_col: str) -> dict[str, torch_frame.stype]:
    # Handle supervised column
    if not pretrain:
        # !!! self.df['Is Laundering'] column stores strings "0" and "1".
        self.df['target'] = self.df[supervised_col].apply(lambda x: [float(x)]) + self.df['link'] 
        self.target_col = 'target'
        col_to_stype['target'] = torch_frame.relation
        return col_to_stype