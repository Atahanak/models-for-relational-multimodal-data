import logging
import wandb
from tqdm import tqdm as tqdm

from torch_frame.data import DataLoader
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
import torch
from torch_geometric.utils import degree

from src.datasets import EthereumPhishing, EllipticBitcoin
from sklearn.metrics import f1_score
from utils import *

@torch.no_grad()
def evaluate(loader, dataset, tensor_frame, model, device, args, mode):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    for batch in tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        batch_size = len(batch.y)
        node_feats, edge_index, edge_attr, y, mask = dataset.get_graph_inputs(batch, mode=mode, args=args)

        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
        with torch.no_grad():
            pred = model(node_feats, edge_index, edge_attr)[:batch_size]
            if mask is not None:
                pred = pred[mask]
                y = y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(y)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    model.train()
    return f1

# workaround for CUDA invalid configuration bug
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

parser = create_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
config={
    "epochs": args.n_epochs,
    "batch_size": args.batch_size,
    "model": args.model,
    "data": args.data,
    "output_path" : args.output_path,
    "num_neighbors": args.num_neighs,
    "emlps": args.emlps, 
    "reverse_mp": args.reverse_mp,
    "ego": args.ego,
    "ports": args.ports,
    #"lr": 0.0004,
    #"lr": 5e-4,
    "lr": 0.0006116418195373612,
    "n_hidden": 32,
    "n_gnn_layers": 2,
    "n_classes" : 2,
    "loss": "ce",
    "w_ce1": 1.,
    #"w_ce2": 6.,
    "w_ce2": 9.23,
    #"w_ce2": 40.97,
    "dropout": 0.083,
    #"dropout": 0.10527690625126304,
    "task": "node_classification"
}

#define a model config dictionary and wandb logging at the same time
print("Testing: ", args.testing)
wandb.init(
    dir=args.wandb_dir,
    mode="disabled" if args.testing else "online",
    project="rel-mm-supervised", #replace this with your wandb project name if you want to use wandb logging
    entity="cse3000",
    config=config
)

create_experiment_path(config) # type: ignore
# Create a logger
logger_setup()

if 'ethereum-phishing-transaction-network' in config['data']:
    dataset = EthereumPhishing(
        root=config['data'],
        split_type='temporal_daily', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        channels=config['n_hidden']
    )
elif 'elliptic_bitcoin_dataset' in config['data']:
    dataset = EllipticBitcoin(
        root=config['data'],
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        channels=config['n_hidden']
    )
else:
    raise ValueError("Invalid data name!")
nodes = dataset.nodes
edges = dataset.edges

train_dataset, val_dataset, test_dataset = nodes.split()

if config['model'] == 'pna' or config['model'] == 'tabgnn' or config['model'] == 'tabgnnfused':
    edge_index = edges.train_graph.edge_index
    num_nodes = edges.train_graph.num_nodes
    config["in_degrees"] = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)

tensor_frame = nodes.tensor_frame 
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)

num_numerical = len(edges.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in edges.tensor_frame.col_names_dict else 0
num_categorical = len(edges.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in edges.tensor_frame.col_names_dict else 0
num_timestamp = len(edges.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in edges.tensor_frame.col_names_dict else 0
num_columns = num_numerical + num_categorical + num_timestamp
config['num_edge_features'] = num_columns
logging.info(f"Number of edge features: {num_columns}")

num_numerical = len(nodes.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in nodes.tensor_frame.col_names_dict else 0
num_categorical = len(nodes.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in nodes.tensor_frame.col_names_dict else 0
num_timestamp = len(nodes.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in nodes.tensor_frame.col_names_dict else 0
config['num_node_features'] = num_numerical + num_categorical + num_timestamp
logging.info(f"Number of node features: {config['num_node_features']}")


stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.timestamp: TimestampEncoder(),
}

if config['model'] == 'pna' or config['model'] == 'gin':
    model = GNN(
                edges.col_stats, 
                edges.tensor_frame.col_names_dict, 
                stype_encoder_dict, 
                config
            ).to(device)
elif config['model'] == 'tabgnn':
    model = TABGNNS(
                edges.col_stats, 
                edges.tensor_frame.col_names_dict, 
                stype_encoder_dict, 
                config
            ).to(device)
elif config['model'] == 'tabgnnfused':
    model = TABGNNFusedS(
                edges.col_stats, 
                edges.tensor_frame.col_names_dict, 
                stype_encoder_dict, 
                config
            ).to(device)
else:
    raise ValueError("Invalid model name!")

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config['w_ce1'], config['w_ce2']]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

best_val_f1 = 0
for epoch in range(config['epochs']):
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    for batch in tqdm(train_loader, disable=not args.tqdm):

        optimizer.zero_grad()
        batch_size = len(batch.y)

        node_feats, edge_index, edge_attr, y, mask = dataset.get_graph_inputs(batch, mode='train', args=args)
        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

        pred = model(node_feats, edge_index, edge_attr)[:batch_size]
        
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        # print(pred)
        # print(y)

        preds.append(pred.argmax(dim=-1))
        ground_truths.append(y)
        loss = loss_fn(pred, y.view(-1).to(torch.long))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()
            
    pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    f1 = f1_score(ground_truth, pred)
    wandb.log({"f1/train": f1}, step=epoch)
    wandb.log({"Loss": total_loss/total_examples}, step=epoch)
    logging.info(f'Train F1: {f1:.4f}, Epoch: {epoch}')
    logging.info(f'Train Loss: {total_loss/total_examples:.4f}')

    # evaluate
    val_f1 = evaluate(val_loader, dataset, edges.tensor_frame, model, device, args, 'val')
    te_f1 = evaluate(test_loader, dataset, edges.tensor_frame, model, device, args, 'test')

    wandb.log({"f1/validation": val_f1}, step=epoch)
    wandb.log({"f1/test": te_f1}, step=epoch)
    logging.info(f'Validation F1: {val_f1:.4f}')
    logging.info(f'Test F1: {te_f1:.4f}')

    if epoch == 0:
        wandb.log({"best_test_f1": te_f1}, step=epoch)
    elif val_f1 > best_val_f1:
        best_val_f1 = val_f1
        wandb.log({"best_test_f1": te_f1}, step=epoch)
        # if args.save_model:
        #     save_model(model, optimizer, epoch, config)





