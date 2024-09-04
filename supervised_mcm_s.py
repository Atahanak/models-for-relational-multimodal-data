import logging
import os
import wandb
from tqdm import tqdm as tqdm

from torch_frame.data import DataLoader
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
import torch
from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML, EthereumPhishing, EllipticBitcoin, OgbnArxiv
from src.datasets.util.mask import PretrainType
from src.utils.loss import SSLoss
from sklearn.metrics import f1_score
from utils import create_parser, logger_setup, save_model
from utils import TT, GNN, TABGNNS, TABGNNFusedS

# workaround for CUDA invalid configuration bug
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch.multiprocessing.set_sharing_strategy('file_system')

def train(epoch, loader, dataset, model, device, args, mode, config):

    total_loss = total_examples = 0
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = acc = 1e-12

    preds = []
    ground_truths = []

    for batch in tqdm(train_loader, disable=not args.tqdm):

        optimizer.zero_grad()

        if 'mcm' in config['task'] and 'ethereum-phishing-transaction-network' in config['data']:
            node_feats, edge_index, edge_attr, y, mask = dataset.get_mcm_inputs(batch, mode='train', args=args)
        else:
            node_feats, edge_index, edge_attr, y, mask = dataset.get_graph_inputs(batch, mode='train', args=args)
        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
        batch_size = y.size(0)

        pred = model(node_feats, edge_index, edge_attr)
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        
        if 'mcm' in config['task']:
            # print("pred: ", pred[0].shape)
            # print("y: ", y.shape)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(pred[1][:batch_size], pred[0][:batch_size], y)
            acc += loss_c[2]
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss = t_loss
            total_loss += loss.item()
            total_examples += (t_c + t_n)
        elif 'classification' in config['task']:
            preds.append(pred[:batch_size].argmax(dim=-1))
            ground_truths.append(y)
            loss = loss_fn(pred[:batch_size], y.view(-1).to(torch.long))
            total_loss += loss.item() * pred.numel()
            total_examples += pred.numel()
        
        loss.backward()
        optimizer.step()

            
    if 'mcm' in config['task']:
        wandb.log({"loss_mcm": total_loss/total_examples}, step=epoch)
        wandb.log({"train_acc": acc / t_c}, step=epoch)
        wandb.log({"loss_c": loss_c_accum / t_c}, step=epoch)
        wandb.log({"loss_n": loss_n_accum / t_n}, step=epoch)
        wandb.log({"train_rmse": loss_n_accum / t_n}, step=epoch)

        logging.info(f'Train Loss: {total_loss/total_examples:.4f}')
        logging.info(f'Train Acc: {acc/t_c:.4f}')
        logging.info(f'Loss C: {loss_c_accum / t_c:.4f}')
        logging.info(f'Loss N: {loss_n_accum / t_n:.4f}')
        logging.info(f'Train RMSE: {loss_n_accum / t_n:.4f}')
        return loss_n_accum / t_n, acc / total_examples
    elif 'classification' in config['task']:
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        if config['n_classes'] == 2:
            f1 = f1_score(ground_truth, pred)
        else:
            f1 = f1_score(ground_truth, pred, average='weighted')

        wandb.log({"loss": total_loss/total_examples}, step=epoch)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}, Epoch: {epoch}')
        return f1   

@torch.no_grad()
def evaluate(epoch, loader, dataset, model, device, args, mode, config):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = acc = rmse = 1e-12
    for batch in tqdm(loader, disable=not args.tqdm):
        if 'mcm' in config['task'] and 'ethereum-phishing-transaction-network' in config['data']:
            node_feats, edge_index, edge_attr, y, mask = dataset.get_mcm_inputs(batch, mode='train', args=args)
        else:
            node_feats, edge_index, edge_attr, y, mask = dataset.get_graph_inputs(batch, mode='train', args=args)
        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
        batch_size = y.size(0)

        pred = model(node_feats, edge_index, edge_attr) 
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        with torch.no_grad():
            if 'mcm' in config['task']:
                _, loss_c, loss_n = ssloss.mcm_loss(pred[1][:batch_size], pred[0][:batch_size], y)
                acc += loss_c[2]
                t_c += loss_c[1]
                t_n += loss_n[1]
                loss_c_accum += loss_c[0].item()
                loss_n_accum += loss_n[0].item()
            elif 'classification' in config['task']:
                preds.append(pred[:batch_size].argmax(dim=-1))
                ground_truths.append(y)

    model.train()
    if 'mcm' in config['task']:
        wandb.log({f"{mode}_rmse": loss_n_accum / t_n}, step=epoch)
        wandb.log({f"{mode}_accuracy": acc / t_c}, step=epoch)
        return loss_n_accum / t_n, acc / t_c
    elif 'classification' in config['task']:
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        if config['n_classes'] == 2:
            f1 = f1_score(ground_truth, pred)
        else:
            f1 = f1_score(ground_truth, pred, average='weighted')
        wandb.log({f"{mode}_f1": f1}, step=epoch)
        logging.info(f'{mode} f1: {f1:.4f}')
        acc = (pred == ground_truth).mean()
        wandb.log({f"{mode}_acc": acc}, step=epoch)
        logging.info(f'{mode} acc: {acc:.4f}')
        return f1

parser = create_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
config={
    "epochs": args.epochs,
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
    "n_hidden": args.n_hidden,
    "n_gnn_layers": args.n_gnn_layers,
    "n_classes" : 2,
    "loss": "ce",
    "w_ce1": 1.,
    #"w_ce2": 6.,
    "w_ce2": 9.23,
    "loss_weights": [1., 9.23],
    #"w_ce2": 40.97,
    "dropout": 0.083,
    #"dropout": 0.10527690625126304,
    "task": args.task,
    "load_model": args.load_model,
}

#define a model config dictionary and wandb logging at the same time
print("Testing: ", args.testing)
wandb.init(
    dir=args.wandb_dir,
    mode="disabled" if args.testing else "online",
    project="lol", #replace this with your wandb project name if you want to use wandb logging
    entity="cse3000",
    group=args.group,
    config=config
)

config['experiment_path'] = args.wandb_dir + wandb.run.id + '/'
if args.save_model:
    os.mkdir(config['experiment_path'])

print(config['experiment_path'])
logger_setup()

if 'ethereum-phishing-transaction-network' in config['data']:
    dataset = EthereumPhishing(
        root=config['data'],
        pretrain={PretrainType.MASK, PretrainType.LINK_PRED},
        split_type='temporal_daily', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden'],
        use_cutoffs=True
    )
    config['lr'] = 0.0008
    config['dropout'] = 0.123
    config['w_ce2'] = 1.16
    config['n_gnn_layers'] = 2
    # config['n_hidden'] = 32
    # config['task'] = 'node_classification-mcm_edge_table'
elif 'elliptic_bitcoin_dataset' in config['data']:
    dataset = EllipticBitcoin(
        root=config['data'],
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
    config['task'] = 'node_classification'
elif 'ibm-transactions-for-anti-money-laundering-aml' in config['data']:
    dataset = IBMTransactionsAML(
        root=config['data'],
        split_type='temporal_daily', 
        pretrain = {PretrainType.MASK, PretrainType.LINK_PRED} if 'mcm' in config['task'] else {},
        splits=[0.6, 0.2, 0.2], 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        channels=config['n_hidden'],
    )
elif 'ogbn_arxiv' in config['data']:
    dataset = OgbnArxiv(
        root=config['data'],
        split_type='temporal', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
    config['task'] = 'node_classification'
    config['n_classes'] = 40
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
else:
    raise ValueError("Invalid data name!")
nodes = dataset.nodes
edges = dataset.edges
if config['load_model'] is not None:
    logging.info(f"Loading encoders from {config['load_model']}")
    nodes.encoder.load_state_dict(torch.load(config['load_model']+'node_encoder'))
    edges.encoder.load_state_dict(torch.load(config['load_model']+'edge_encoder'))

if 'node_classification' in config['task']:
    train_dataset, val_dataset, test_dataset = nodes.split()
else:
    train_dataset, val_dataset, test_dataset = edges.split()

if config['model'] == 'pna' or config['model'] == 'tabgnn' or config['model'] == 'tabgnnfused':
    edge_index = edges.train_graph.edge_index
    num_nodes = edges.train_graph.num_nodes
    config["in_degrees"] = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)

train_loader = DataLoader(train_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)
logging.info(f"Train loader size: {len(train_loader)}")
logging.info(f"Val loader size: {len(val_loader)}")
logging.info(f"Test loader size: {len(test_loader)}")

num_misc = len(edges.tensor_frame.col_names_dict[stype.relation]) if stype.relation in edges.tensor_frame.col_names_dict else 0
num_numerical = len(edges.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in edges.tensor_frame.col_names_dict else 0
num_categorical = len(edges.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in edges.tensor_frame.col_names_dict else 0
num_timestamp = len(edges.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in edges.tensor_frame.col_names_dict else 0
num_columns = num_numerical + num_categorical + num_timestamp + num_misc
config['num_edge_features'] = num_columns
config['masked_num_numerical_edge'] = len(edges.masked_numerical_columns)
config['masked_num_categorical_edge'] = len(edges.masked_categorical_columns)
config['masked_categorical_ranges_edge'] = [len(edges.col_stats[col][StatType.COUNT][0]) for col in edges.tensor_frame.col_names_dict[stype.categorical] if col in edges.masked_categorical_columns] if stype.categorical in edges.tensor_frame.col_names_dict else []
logging.info(f"Number of edge features: {num_columns}")

num_misc = len(nodes.tensor_frame.col_names_dict[stype.relation]) if stype.relation in nodes.tensor_frame.col_names_dict else 0
num_numerical = len(nodes.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in nodes.tensor_frame.col_names_dict else 0
num_categorical = len(nodes.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in nodes.tensor_frame.col_names_dict else 0
num_timestamp = len(nodes.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in nodes.tensor_frame.col_names_dict else 0
config['num_node_features'] = num_numerical + num_categorical + num_timestamp + num_misc
config['masked_num_numerical_node'] = len(nodes.masked_numerical_columns)
config['masked_num_categorical_node'] = len(nodes.masked_categorical_columns)
config['masked_categorical_ranges_node'] = [len(nodes.col_stats[col][StatType.COUNT][0]) for col in nodes.tensor_frame.col_names_dict[stype.categorical] if col in nodes.masked_categorical_columns] if stype.categorical in nodes.tensor_frame.col_names_dict else []
logging.info(f"Number of node features: {config['num_node_features']}")

if config['model'] == 'pna' or config['model'] == 'gin':
    model = GNN(
                config
            ).to(device)
elif config['model'] == 'tabgnn':
    model = TABGNNS(
                config
            ).to(device)
elif config['model'] == 'tabgnnfused':
    model = TABGNNFusedS(
                config
            ).to(device)
elif config['model'] == 'fttransformer':
    model = TT(
                config
            ).to(device)
else:
    raise ValueError("Invalid model name!")

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(config['loss_weights']).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

if 'edge_table' in config['task']:
    ssloss = SSLoss(device, config['masked_num_numerical_edge'])
elif 'node_table' in config['task']:
    ssloss = SSLoss(device, config['masked_num_numerical_node'])

# freeze tabular layers
if args.freeze:
    print("Freezing tabular layers")
    for param in model.model.tabular_backbone.parameters():
        param.requires_grad = False

if 'mcm' in config['task']:
    best_m = [1000, 0]
else:
    best_m = 0
for epoch in range(config['epochs'], config['epochs']*2):
    train_m = train(epoch, train_loader, dataset, model, device, args, 'train', config)
    val_m = evaluate(epoch, val_loader, dataset, model, device, args, 'val', config)
    te_m = evaluate(epoch, test_loader, dataset, model, device, args, 'test', config)

    if 'mcm' in config['task']:
        if val_m[0] < best_m[0] and val_m[1] > best_m[1]:
            best_m = val_m
            wandb.log({"best_test_acc": te_m[1]}, step=epoch)
            logging.info(f'Best test acc: {te_m[1]:.4f}')
            wandb.log({"best_test_rmse": te_m[0]}, step=epoch)
            logging.info(f'Best test rmse: {te_m[0]:.4f}')

            if args.save_model:
                logging.info(f"Saving model to {config['experiment_path']}")
                torch.save(nodes.encoder.state_dict(), config['experiment_path']+'node_encoder')
                torch.save(edges.encoder.state_dict(), config['experiment_path']+'edge_encoder')
                torch.save(model.model.state_dict(), config['experiment_path']+'model')


    if 'classification' in config['task']:
        if val_m > best_m:
            best_m = val_m
            wandb.log({"best_test_f1": te_m}, step=epoch)