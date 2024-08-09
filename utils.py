import os
import logging
import sys
import argparse
from tqdm import tqdm
from datetime import datetime
import os 
import os.path as osp 

from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
import torch 
import torch.nn as nn
from sklearn.metrics import f1_score

from src.nn.gnn.model import GINe, PNAS
from src.nn.gnn.decoder import ClassifierHead, NodeClassificationHead
from src.nn.models import TABGNN
from src.nn.models import TABGNNFused
from src.utils.batch_processing import graph_inputs, node_inputs

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
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
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

    return parser

@torch.no_grad()
def evaluate(loader, dataset, tensor_frame, model, device, args, mode):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    for batch in tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        batch_size = len(batch.y)
        node_feats, edge_index, edge_attr, y = graph_inputs(dataset, batch, tensor_frame, mode=mode, args=args)
        
        # # convert tensorframe to tensor
        # feats = []
        # from datetime import datetime
        # for stype in edge_attr.stypes:
        #     feat = edge_attr.feat_dict[stype]
        #     if feat.dim() == 3:
        #         years_in_seconds = (feat[:, :, 0]) * 365 * 24 * 3600
        #         months_in_seconds = (feat[:, :, 1]) * 30 * 24 * 3600
        #         days_in_seconds = (feat[:, :, 2]) * 24 * 3600
        #         hours_in_seconds = feat[:, :, 3] * 3600
        #         minutes_in_seconds = feat[:, :, 4] * 60
        #         seconds = feat[:, :, 5]

        #         # Sum all components to get the UNIX timestamp
        #         feat = years_in_seconds + months_in_seconds + days_in_seconds + hours_in_seconds + minutes_in_seconds + seconds
        #     elif feat.dim() == 1:
        #         feat = feat.unsqueeze(1)
        #     feats.append(feat)
        # edge_attr = torch.cat(feats, dim=1)

        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
        with torch.no_grad():
            pred = model(node_feats, edge_index, edge_attr)[:batch_size]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(y)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    model.train()
    return f1

def save_model(model, optimizer, epoch, config, ):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                },
            osp.join(config['experiment_path'], str(epoch)+ '.tar')
            )
    
def create_experiment_path(config):
    """
    Get unique experiment id
    """
    run_str = '{date:%m-%d_%H:%M:%S.%f}'.format(date=datetime.now())
    config['experiment_path'] = osp.join(config['output_path'], 'experiments', run_str)
    os.makedirs(config['experiment_path'])
    return

class GNN(nn.Module):
    def __init__(self, col_stats, col_names_dict, stype_encoder_dict, config):
        super().__init__()
        self.config = config
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=config['n_hidden'],
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
        )
        self.graph_model = self.get_graph_model(config)
        if config['task'] == 'edge_classification':
            self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
        elif config['task'] == 'node_classification':
            self.classifier = NodeClassificationHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])

    def forward(self, x, edge_index, edge_attr):
        #edge_attr, _ = self.encoder(edge_attr)  
        x, edge_attr = self.graph_model(x, edge_index, edge_attr)
        if self.config['task'] == 'edge_classification':
            out = self.classifier(x, edge_index, edge_attr)
        elif self.config['task'] == 'node_classification':
            out = self.classifier(x)
        return out

    def get_graph_model(self, config):
        
        n_feats = 2 if config['ego'] else 1
        #e_dim = config['num_columns'] * config['n_hidden']
        e_dim = config['num_columns']

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
                num_features=n_feats, 
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
    def __init__(self, col_stats, col_names_dict, stype_encoder_dict, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=config['n_hidden'],
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
        )
        self.model = self.get_model(config)
        self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
    
    def forward(self, x, edge_index, edge_attr):
        edge_attr, _ = self.encoder(edge_attr)
        edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
        x, edge_attr, target_edge_attr = self.model(x, edge_index, edge_attr, target_edge_attr)
        out = self.classifier(x, target_edge_index, target_edge_attr)
        return out

    def get_model(self, config):
        
        n_feats = 2 if config['ego'] else 1
        e_dim = config['num_columns'] * config['n_hidden']

        if config['model'] == "tabgnn":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = TABGNN(
                node_dim=n_feats, 
                encoder=self.encoder,
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
    def __init__(self, col_stats, col_names_dict, stype_encoder_dict, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=config['n_hidden'],
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
        )
        self.model = self.get_model(config)
        self.classifier = ClassifierHead(config['n_classes'], config['n_hidden'], dropout=config['dropout'])
    
    def forward(self, x, edge_index, edge_attr):
        edge_attr, _ = self.encoder(edge_attr)
        edge_attr, target_edge_attr = edge_attr[self.batch_size:, :], edge_attr[:self.batch_size, :]
        edge_index, target_edge_index = edge_index[:, self.batch_size:], edge_index[:, :self.batch_size]
        x, edge_attr, target_edge_attr = self.model(x, edge_index, edge_attr, target_edge_index, target_edge_attr)
        out = self.classifier(x, target_edge_index, target_edge_attr)
        return out

    def get_model(self, config):
        
        n_feats = 2 if config['ego'] else 1
        e_dim = config['num_columns'] * config['n_hidden']

        if config['model'] == "tabgnnfused":
            if config['in_degrees'] is None:
                raise ValueError("In degrees are not provided for PNA model!")
            in_degrees = config['in_degrees']
            max_in_degree = int(max(in_degrees))
            in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
            in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
            model = TABGNNFused(
                node_dim=n_feats, 
                encoder=self.encoder,
                nhidden=config['n_hidden'], 
                channels=config['n_hidden'], 
                num_layers=config['n_gnn_layers'], 
                edge_dim=e_dim, 
                deg=in_degree_histogram,
                reverse_mp=config['reverse_mp'])
        else:
            raise ValueError("Invalid model name!")
        
        return model