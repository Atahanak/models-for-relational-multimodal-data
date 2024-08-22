import os
import logging
import sys
import argparse
from datetime import datetime
import os 
import os.path as osp 

from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
import torch 
import torch.nn as nn

from src.nn.gnn.model import GINe, PNAS
from src.nn.models import FTTransformer#, Trompt
from src.nn.gnn.decoder import ClassifierHead, NodeClassificationHead
from src.nn.decoder import MCMHead
from src.nn.models import TABGNN
from src.nn.models import TABGNNFused

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

    #Model parameters
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")
    parser.add_argument("--ports", action='store_true', help="Use ports in GNN training")
    parser.add_argument("--batch_size", default=200, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', type=int, default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')
    

    #Misc
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--output_path", default="/mnt/data/outputs/", type=str, help="Output path to save the best models", required=False)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--wandb_dir", default="/mnt/data/wandb/", type=str, help="Wandb directory to save the logs", required=False)
    parser.add_argument("--group", default="null", type=str, help="wandb group", required=False)

    return parser

def save_model(model, optimizer, epoch, config, ):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                },
            osp.join(config['experiment_path'], str(epoch)+ '.tar')
            )
    
class TT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.get_model(config)
        if config['task'] == 'edge_classification':
            self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
    
    def forward(self, x, edge_index, edge_attr):
        x, x_cls = self.model(x)
        if self.config['task'] == 'edge_classification':
            edge_attr, e_cls = self.model(edge_attr)
            out = self.classifier(x_cls, edge_index, e_cls)
        elif self.config['task'] == 'node_classification':
            out = self.classifier(x_cls)

        return out

    def get_model(self, config):
        n_feats = config['num_node_features']
        n_dim = n_feats*config['n_hidden'] 
        e_dim = config['num_edge_features'] * config['n_hidden']

        if config['model'] == "fttransformer":
            model = FTTransformer(
                channels=config["n_hidden"],
                num_layers=config['n_gnn_layers'],
                #nhead=config['nhead'],
            )
        elif config['model'] == "trompt":
            ValueError("TROMPT model not implemented yet!")
        else:
            raise ValueError("Invalid model name!")
        
        return model

class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_model = self.get_graph_model(config)
        if config['task'] == 'edge_classification':
            self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr = self.graph_model(x, edge_index, edge_attr)
        if self.config['task'] == 'edge_classification':
            out = self.classifier(x, edge_index, edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.classifier(x)
        return out

    def get_graph_model(self, config):
        
        n_feats = config['num_node_features']
        n_dim = n_feats 
        n_dim = n_feats*config['n_hidden'] 
        e_dim = config['num_edge_features']
        #e_dim = config['num_edge_features'] * config['n_hidden']

        if config['model'] == "gin":
            model = GINe(num_features=n_dim, num_gnn_layers=config['n_gnn_layers'], 
                         n_hidden=config['n_hidden'], 
                         edge_updates=config['emlps'], 
                         edge_dim=e_dim, 
                         reverse_mp=config['reverse_mp'])
        elif config['model'] == "pna":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = PNAS(
                num_features=n_dim,
                n_hidden=config['n_hidden'], 
                num_gnn_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                edge_updates=config['emlps'], 
                reverse_mp=config['reverse_mp'])
        else:
            raise ValueError("Invalid model name!")
        
        return model

class TABGNNS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        #self.emb = config['emb']
        self.model = self.get_model(config)
        if config['task'] == 'edge_classification':
            self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification-mcm_edge_table':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
            self.mcm = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3)
    
    def forward(self, x, edge_index, edge_attr):

        #x, edge_attr, target_edge_attr = self.model(x, edge_index, edge_attr, target_edge_attr)
        x, edge_attr = self.model(x, edge_index, edge_attr)

        if self.config['task'] == 'edge_classification':
            edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
            edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
            out = self.classifier(x, target_edge_index, target_edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.classifier(x)
        elif self.config['task'] == 'node_classification-mcm_edge_table':
            out = self.classifier(x)
            x_target = x[edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
            x_target = torch.cat((x_target, edge_attr), 1)
            out2 = self.mcm(x_target)
            return {"supervised": out, "mcm": out2}
        return out

    def get_model(self, config):
        
        n_feats = config['num_node_features']
        #n_dim = n_feats
        n_dim = n_feats*config['n_hidden'] 
        #e_dim = config['num_edge_features'] * config['n_hidden']
        e_dim = config['num_edge_features']

        if config['model'] == "tabgnn":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = TABGNN(
                node_dim=n_dim, 
                nhidden=config['n_hidden'], 
                channels=config['n_hidden'], 
                num_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                reverse_mp=config['reverse_mp'])
        else:
            raise ValueError("Invalid model name!")
        
        return model

class TABGNNFusedS(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        
        self.model = self.get_model(config)
        if config['task'] == 'edge_classification':
            self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification-mcm_edge_table':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
            self.mcm = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3)
    
    def forward(self, x, edge_index, edge_attr):
        edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]

        x, edge_attr = self.model(x, edge_index, edge_attr, target_edge_index, target_edge_attr)
        if self.config['task'] == 'edge_classification':
            out = self.classifier(x, target_edge_index, target_edge_attr)
        elif self.config['task'] == 'node_classification':
            #out = self.classifier(x[:, 0, :])
            out = self.classifier(x)
        elif self.config['task'] == 'node_classification-mcm_edge_table':
            out = self.classifier(x)
            x_target = x[edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
            x_target = torch.cat((x_target, edge_attr), 1)
            out2 = self.mcm(x_target)
            return {"supervised": out, "mcm": out2}
        return out

    def get_model(self, config):
        
        n_feats = config['num_node_features']
        n_dim = n_feats*config['n_hidden'] 
        #e_dim = config['num_edge_features'] * config['n_hidden']
        e_dim = config['num_edge_features']

        if config['model'] == "tabgnnfused":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = TABGNNFused(
                node_dim=n_dim, 
                #encoder=self.encoder,
                nhidden=config['n_hidden'], 
                channels=config['n_hidden'], 
                num_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                reverse_mp=config['reverse_mp'])
        else:
            raise ValueError("Invalid model name!")
        
        return model