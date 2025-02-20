import os
import logging
import sys
import argparse
import os 
import os.path as osp 

import torch 
import torch.nn as nn

from nn.gnn import GINe, PNAS, CPNA, CPNATAB
from src.nn.models import FTTransformer
from src.nn.gnn.decoder import ClassifierHead, NodeClassificationHead
from src.nn.decoder import MCMHead
from src.nn.models import TABGNN
from src.nn.models import TABGNNFused, TABGNNInterleaved

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
    parser.add_argument("--n_hidden", default=32, type=int, help="Number of hidden units in the GNN model", required=False)
    parser.add_argument("--n_gnn_layers", default=2, type=int, help="Number of GNN layers in the model", required=False)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--freeze", action="store_true", help="freeze model parameters for tabular backbone", required=False)

    #Misc
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--output_path", default="/mnt/data/outputs/", type=str, help="Output path to save the best models", required=False)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--load_model", default=None, type=str, help="Load model.")
    parser.add_argument("--checkpoint", action='store_true', help="Load checkpoint.")
    parser.add_argument("--wandb_dir", default="/mnt/data/wandb/", type=str, help="Wandb directory to save the logs", required=False)
    parser.add_argument("--group", default="null", type=str, help="wandb group", required=False)
    parser.add_argument("--task", default="edge_classification", type=str, help="Task to be performed by the model", required=False)

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
            self.decoder = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
    
    def forward(self, x, edge_index, edge_attr):
        x, x_cls = self.model(x)
        if self.config['task'] == 'edge_classification':
            edge_attr, e_cls = self.model(edge_attr)
            out = self.decoder(x_cls, edge_index, e_cls)
        elif self.config['task'] == 'node_classification':
            out = self.decoder(x_cls)

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
        self.batch_size = config['batch_size']
        self.node_encoder = config['node_encoder']
        self.edge_encoder = config['edge_encoder']
        self.model = self.get_graph_model(config)
        if config['task'] == 'edge_classification':
            if config['model'] == "cpna" or self.config['model'] == "cpnatab":
                self.decoder = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'], e_hidden=config['num_edge_features'] * config['n_hidden'])
            else:
                self.decoder = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            if config['model'] == "cpna" or self.config['model'] == "cpnatab":
                self.decoder = NodeClassificationHead(config['n_classes'], config['num_edge_features'] * config['n_hidden'], dropout=config['dropout'])
            else:
                self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'mcm_edge_table':
            if config['model'] == "cpna" or self.config['model'] == "cpnatab":
                self.decoder = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=config['num_edge_features']+2)
            else:
                self.decoder = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3)
        if config['load_model'] is not None and config['checkpoint']:
            logging.info(f"Loading decoder from {config['load_model']}")
            self.decoder.load_state_dict(torch.load(config['load_model'] + 'decoder')) 

    def forward(self, x, edge_index, edge_attr):
        x, _ = self.node_encoder(x)
        edge_attr, _ = self.edge_encoder(edge_attr)
        x, edge_attr = self.model(x, edge_index, edge_attr)
        if self.config['model'] == "cpna" or self.config['model'] == "cpnatab":
            edge_attr = edge_attr.reshape(-1, self.config['num_edge_features'] * self.config['n_hidden'])
            edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        else:
            edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]

        if self.config['task'] == 'edge_classification':
            out = self.decoder(x, target_edge_index, target_edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.decoder(x)
        elif self.config['task'] == 'mcm_edge_table':
            # edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
            # edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
            x_target = x[target_edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
            x_target = torch.cat((x_target, target_edge_attr), 1)
            out = self.decoder(x_target)

        return out

    def get_graph_model(self, config):
        
        n_feats = config['num_node_features']
        n_dim = n_feats*config['n_hidden'] 
        #n_dim = n_feats 
        e_dim = config['num_edge_features'] * config['n_hidden']
        #e_dim = config['num_edge_features']

        if config['model'] == "gin":
            model = GINe(num_features=n_feats, num_gnn_layers=config['n_gnn_layers'], 
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
            if config['load_model'] is not None:
                logging.info(f"Loading model from {config['load_model']}")
                model.load_state_dict(torch.load(config['load_model'] + 'model'))
        elif config['model'] == "cpna":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = CPNA(
                num_features=n_dim,
                n_hidden=config['n_hidden'], 
                num_gnn_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                edge_updates=config['emlps'], 
                reverse_mp=config['reverse_mp'])
            if config['load_model'] is not None:
                logging.info(f"Loading model from {config['load_model']}")
                model.load_state_dict(torch.load(config['load_model'] + 'model'))
        elif config['model'] == "cpnatab":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = CPNATAB(
                num_features=n_dim,
                n_hidden=config['n_hidden'], 
                num_gnn_layers=config['n_gnn_layers'], 
                edge_dim=e_dim,
                deg=in_degree_histogram,
                edge_updates=config['emlps'], 
                reverse_mp=config['reverse_mp'])
            if config['load_model'] is not None:
                logging.info(f"Loading model from {config['load_model']}")
                model.load_state_dict(torch.load(config['load_model'] + 'model')) 
        else:
            raise ValueError("Invalid model name!")
        
        return model

class TABGNNS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.node_encoder = config['node_encoder']
        self.edge_encoder = config['edge_encoder']

        self.model = self.get_model(config)
        if config['task'] == 'edge_classification':
            self.decoder = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        # elif config['task'] == 'node_classification-mcm_edge_table':
        #     self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        #     self.mcm = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3)
        elif config['task'] == 'mcm_edge_table':
            self.decoder = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3) 
        if config['load_model'] is not None and config['checkpoint']:
            logging.info(f"Loading decoder from {config['load_model']}")
            self.decoder.load_state_dict(torch.load(config['load_model'] + 'decoder')) 
    
    def forward(self, x, edge_index, edge_attr):

        #x, edge_attr, target_edge_attr = self.model(x, edge_index, edge_attr, target_edge_attr)
        x, _ = self.node_encoder(x)
        edge_attr, _ = self.edge_encoder(edge_attr)
        x, edge_attr = self.model(x, edge_index, edge_attr)

        if self.config['task'] == 'edge_classification':
            edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
            edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
            out = self.decoder(x, target_edge_index, target_edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.decoder(x)
        # elif self.config['task'] == 'node_classification-mcm_edge_table':
        #     out = self.decoder(x)
        #     x_target = x[edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
        #     x_target = torch.cat((x_target, edge_attr), 1)
        #     out2 = self.mcm(x_target)
        #     return {"supervised": out, "mcm": out2}
        elif self.config['task'] == 'mcm_edge_table':
            edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
            edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
            x_target = x[target_edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
            x_target = torch.cat((x_target, target_edge_attr), 1)
            out = self.decoder(x_target)
        return out

    def get_model(self, config):
        
        n_feats = config['num_node_features']
        #n_dim = n_feats
        n_dim = n_feats*config['n_hidden'] 
        e_dim = config['num_edge_features'] * config['n_hidden']
        #e_dim = config['num_edge_features']

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
        elif config["model"] == "tabgnninterleaved":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = TABGNNInterleaved(
                node_dim=n_dim, 
                nhidden=config['n_hidden'], 
                channels=config['n_hidden'], 
                num_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                reverse_mp=config['reverse_mp'])
        else:
            raise ValueError("Invalid model name!")
        if config['load_model'] is not None:
            logging.info(f"Loading model from {config['load_model']}")
            model.load_state_dict(torch.load(config['load_model'] + 'model'))
        
        return model

class TABGNNFusedS(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.node_encoder = config['node_encoder']
        self.edge_encoder = config['edge_encoder']
        self.model = self.get_model(config)

        if config['task'] == 'edge_classification':
            self.decoder = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        # elif config['task'] == 'node_classification-mcm_edge_table':
        #     self.decoder = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        #     self.mcm = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3)
        elif config['task'] == 'mcm_edge_table':
            self.decoder = MCMHead(config['n_hidden'], config['masked_num_numerical_edge'], config['masked_categorical_ranges_edge'], w=3) 
        if config['load_model'] is not None and config['checkpoint']:
            logging.info(f"Loading decoder from {config['load_model']}")
            self.decoder.load_state_dict(torch.load(config['load_model'] + 'decoder')) 
    
    def forward(self, x, edge_index, edge_attr):

        edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
        x, _ = self.node_encoder(x)
        edge_attr, _ = self.edge_encoder(edge_attr)
        target_edge_attr, _ = self.edge_encoder(target_edge_attr)
        x, edge_attr, target_edge_attr  = self.model(x, edge_index, edge_attr, target_edge_index, target_edge_attr)
        if self.config['task'] == 'edge_classification':
            out = self.decoder(x, target_edge_index, target_edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.decoder(x[:, 0, :])
        # elif self.config['task'] == 'node_classification-mcm_edge_table':
        #     out = self.decoder(x)
        #     x_target = x[edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
        #     x_target = torch.cat((x_target, edge_attr), 1)
        #     out2 = self.mcm(x_target)
        #     return {"supervised": out, "mcm": out2}
        elif self.config['task'] == 'mcm_edge_table':
            x_target = x[target_edge_index.T].reshape(-1, 2 * self.config["n_hidden"])
            x_target = torch.cat((x_target, target_edge_attr), 1)
            out = self.decoder(x_target)
        return out

    def get_model(self, config):
        
        n_feats = config['num_node_features']
        n_dim = n_feats*config['n_hidden'] 
        e_dim = config['num_edge_features'] * config['n_hidden']

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
            if config['load_model'] is not None:
                logging.info(f"Loading model from {config['load_model']}")
                model.load_state_dict(torch.load(config['load_model'] + 'model'))
        else:
            raise ValueError("Invalid model name!")
        
        return model