import os
import logging
import sys
import argparse
import tqdm
from datetime import datetime
import os 
import os.path as osp 

from torch_frame import TensorFrame
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
import torch 
import torch.nn as nn

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
    parser.add_argument("--output_path", default="./outputs", type=str, help="Output path to save the best models", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")

    return parser


def graph_inputs(dataset, batch: TensorFrame, tensor_frame: TensorFrame, mode='train'):

    edges = batch.y[:,-3:]
    y = batch.y[:, 0].to(torch.long)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, mode)
    edge_attr = tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    return node_feats, edge_index, edge_attr, y


@torch.no_grad()
def evaluate(loader, dataset, tensor_frame, model, device, args, mode):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        batch.to(device)
        batch_size = len(batch.y)
        node_feats, edge_index, edge_attr, y = graph_inputs(dataset, batch, tensor_frame, mode=mode)
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
            osp.join(config.experiment_path, epoch+ '.tar')
            )
    

def create_experiment_path(config):
    """
    Get unique experiment id
    """
    run_str = '{date:%m-%d_%H:%M:%S.%f}'.format(date=datetime.now())
    config.experiment_path = osp.join(config.output_path, 'experiments', run_str)
    os.makedirs(config.experiment_path)
    return


class SupervisedTabGNN(nn.Module):
    def __init__(self, col_stats, col_names_dict, stype_encoder_dict, config):
        super().__init__()
        
        self.encoder = StypeWiseFeatureEncoder(
                    out_channels=config.n_hidden,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
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
        else:
            raise ValueError("Invalid model name!")
        
        return model