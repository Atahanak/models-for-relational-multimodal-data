import os
from types import SimpleNamespace
import logging
import sys
import json
import wandb
import argparse
import tqdm

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

from src.datasets import IBMTransactionsAML
from src.nn.gnn.model import GINe, PNA
from src.nn.gnn.decoder import ClassifierHead
from sklearn.metrics import f1_score


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

def create_parser():
    parser = argparse.ArgumentParser()

    #Adaptations
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")

    #Model parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    #Misc
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")

    return parser

#%% 
parser = create_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define a model config dictionary and wandb logging at the same time
wandb.init(
    mode="disabled" if args.testing else "online",
    project="tab-gnn", #replace this with your wandb project name if you want to use wandb logging

    config={
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "data": args.data,
        "num_neighbors": args.num_neighs,
        "lr": 0.006213266113989207,
        "n_feats" : 1, 
        "n_hidden": 128,
        "n_gnn_layers": 3,
        "n_classes" : 2,
        "loss": "ce",
        "w_ce1": 1.0000182882773443,
        "w_ce2": 6.275014431494497,
        "dropout": 0.10527690625126304,
    }
)

config = wandb.config

# Create a logger
logger_setup()

dataset = IBMTransactionsAML(
    root=config.data,
    split_type='temporal', 
    splits=[0.6, 0.2, 0.2], 
    khop_neighbors=args.num_neighs
)
dataset.materialize()
train_dataset, val_dataset, test_dataset = dataset.split()


tensor_frame = dataset.tensor_frame 
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=config.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=config.batch_size, shuffle=False, num_workers=4)

num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])

num_columns = num_numerical + num_categorical + 1
config.num_columns = num_columns
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
    def __init__(self, stype_encoder_dict, config):
        super().__init__()
        
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=config.n_hidden,
                    col_stats=dataset.col_stats,
                    col_names_dict=dataset.tensor_frame.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
        )

        self.graph_model = self.get_graph_model(config)

        self.classifier = ClassifierHead(config.n_classes, config.n_hidden, dropout=config.dropout)

    
    def forward(self, x, edge_index, edge_attr):
        
        edge_attr, _ = self.encoder(edge_attr)  
        x, edge_attr = self.graph_model(x, edge_index, edge_attr)
        out = self.classifier(x, edge_index, edge_attr)

        return out

    def get_graph_model(self, config):
       
        n_feats = config.n_feats 
        e_dim = config.num_columns * config.n_hidden

        if config.model == "gin":
            model = GINe(num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_hidden=config.n_hidden, edge_updates=True, edge_dim=e_dim)

        # elif config.model == "pna":
        #     if not isinstance(sample_batch, HeteroData):
        #         d = degree(sample_batch.edge_index[1], dtype=torch.long)
        #     else:
        #         index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
        #         d = degree(index, dtype=torch.long)
        #     deg = torch.bincount(d, minlength=1)
        #     model = PNA(
        #         num_features=2, num_gnn_layers=config.n_gnn_layers, n_classes=2,
        #         n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=4,
        #         dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
        #         )
        
        return model


stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.timestamp: TimestampEncoder(),
    }


model = SupervisedTabGNN(stype_encoder_dict, config)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)



best_val_f1 = 0
for epoch in range(config.epochs):
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    for batch in tqdm.tqdm(train_loader, disable=not args.tqdm):
        optimizer.zero_grad()

        batch_size = len(batch.y)
        node_feats, edge_index, edge_attr, y = graph_inputs(batch, tensor_frame)
        pred = model(node_feats, edge_index, edge_attr)[:batch_size]

        preds.append(pred.argmax(dim=-1))
        ground_truths.append(y)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

    pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    f1 = f1_score(ground_truth, pred)
    wandb.log({"f1/train": f1}, step=epoch)
    logging.info(f'Train F1: {f1:.4f}')


    # #evaluate
    # val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
    # te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

    # wandb.log({"f1/validation": val_f1}, step=epoch)
    # wandb.log({"f1/test": te_f1}, step=epoch)
    # logging.info(f'Validation F1: {val_f1:.4f}')
    # logging.info(f'Test F1: {te_f1:.4f}')

    # if epoch == 0:
    #     wandb.log({"best_test_f1": te_f1}, step=epoch)
    # elif val_f1 > best_val_f1:
    #     best_val_f1 = val_f1
    #     wandb.log({"best_test_f1": te_f1}, step=epoch)
    #     if args.save_model:
    #         save_model(model, optimizer, epoch, args, data_config)
