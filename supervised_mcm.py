import logging
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
from utils import create_parser, create_experiment_path, logger_setup, save_model
from utils import TT, GNN, TABGNNS, TABGNNFusedS

# workaround for CUDA invalid configuration bug
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@torch.no_grad()
def evaluate(epoch, loader, dataset, model, device, args, mode, config):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    acc = t_n = t_c = rmse = 1e-12
    for batch in tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        batch_size = len(batch.y)
        node_feats, edge_index, edge_attr, y, y_mcm, mask = dataset.get_mcm_inputs(batch, mode=mode, args=args)

        node_feats, edge_index, edge_attr, y, y_mcm = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), y_mcm.to(device)
        with torch.no_grad():
            pred = model(node_feats, edge_index, edge_attr)
            pred, mcm_pred = pred['supervised'][:batch_size], pred['mcm']
            if mask is not None:
                pred = pred[mask]
                y = y[mask]
            t_loss, loss_c, loss_n = ssloss.mcm_loss(mcm_pred[1], mcm_pred[0], y_mcm)
            acc += loss_c[2]
            t_c += loss_c[1]
            t_n += loss_n[1]
            rmse += loss_n[0].item()
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(y)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    wandb.log({f"{mode}_rmse": rmse / t_n}, step=epoch)
    wandb.log({f"{mode}_accuracy": acc / t_c}, step=epoch)
    if config['n_classes'] == 2:
        f1 = f1_score(ground_truth, pred)
    else:
        f1 = f1_score(ground_truth, pred, average='weighted')
    acc = (pred == ground_truth).mean()
    wandb.log({f"{mode}_acc": acc}, step=epoch)
    logging.info(f'{mode} acc: {acc:.4f}')
    model.train()
    return f1

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
    "loss_weights": [1., 9.23],
    #"w_ce2": 40.97,
    "dropout": 0.083,
    #"dropout": 0.10527690625126304,
    "task": "node_classification-mcm_edge_table"
}

#define a model config dictionary and wandb logging at the same time
print("Testing: ", args.testing)
wandb.init(
    dir=args.wandb_dir,
    mode="disabled" if args.testing else "online",
    project="rel-mm-supervised-mcm", #replace this with your wandb project name if you want to use wandb logging
    entity="cse3000",
    config=config
)

create_experiment_path(config) # type: ignore
# Create a logger
logger_setup()
if 'ethereum-phishing-transaction-network' in config['data']:
    dataset = EthereumPhishing(
        root=config['data'],
        pretrain={PretrainType.MASK, PretrainType.LINK_PRED},
        split_type='temporal_daily', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
    config['lr'] = 0.0008
    config['dropout'] = 0.123
    config['w_ce2'] = 1.16
    config['n_gnn_layers'] = 2
    config['n_hidden'] = 32
elif 'elliptic_bitcoin_dataset' in config['data']:
    dataset = EllipticBitcoin(
        root=config['data'],
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
elif 'ibm-transactions-for-anti-money-laundering-aml' in config['data']:
    dataset = IBMTransactionsAML(
        root=config['data'],
        split_type='temporal_daily', 
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
config['masked_categorical_ranges_edge'] = [len(nodes.col_stats[col][StatType.COUNT][0]) for col in nodes.tensor_frame.col_names_dict[stype.categorical] if col in nodes.masked_categorical_columns] if stype.categorical in nodes.tensor_frame.col_names_dict else []
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

best_val_f1 = 0
for epoch in range(config['epochs']):
    total_loss = total_examples = 0
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = acc = 1e-12

    preds = []
    ground_truths = []

    for batch in tqdm(train_loader, disable=not args.tqdm):

        optimizer.zero_grad()
        batch_size = len(batch.y)

        node_feats, edge_index, edge_attr, y, y_mcm, mask = dataset.get_mcm_inputs(batch, mode='train', args=args)
        node_feats, edge_index, edge_attr, y, y_mcm = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), y_mcm.to(device)

        pred = model(node_feats, edge_index, edge_attr)
        pred, mcm_pred = pred['supervised'][:batch_size], pred['mcm']
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        
        t_loss, loss_c, loss_n = ssloss.mcm_loss(mcm_pred[1], mcm_pred[0], y_mcm)
        preds.append(pred.argmax(dim=-1))
        ground_truths.append(y)
        loss = loss_fn(pred, y.view(-1).to(torch.long))
        loss += t_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pred.numel()
        acc += loss_c[2]
        t_c += loss_c[1]
        t_n += loss_n[1]
        loss_c_accum += loss_c[0].item()
        loss_n_accum += loss_n[0].item()
        total_examples += pred.numel()
            
    pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    if config['n_classes'] == 2:
        f1 = f1_score(ground_truth, pred)
    else:
        f1 = f1_score(ground_truth, pred, average='weighted')

    wandb.log({"f1/train": f1}, step=epoch)
    wandb.log({"Loss": total_loss/total_examples}, step=epoch)
    wandb.log({"train_acc": acc / total_examples}, step=epoch)
    wandb.log({"loss_c": loss_c_accum / t_c}, step=epoch)
    wandb.log({"loss_n": loss_n_accum / t_n}, step=epoch)
    wandb.log({"train_rmse": loss_n_accum / t_n}, step=epoch)

    logging.info(f'Train F1: {f1:.4f}, Epoch: {epoch}')
    logging.info(f'Train Loss: {total_loss/total_examples:.4f}')
    logging.info(f'Train Acc: {acc:.4f}')
    logging.info(f'Loss C: {loss_c_accum / t_c:.4f}')
    logging.info(f'Loss N: {loss_n_accum / t_n:.4f}')
    logging.info(f'Train RMSE: {loss_n_accum / t_n:.4f}')

    # evaluate
    val_f1 = evaluate(epoch, val_loader, dataset, model, device, args, 'val', config)
    te_f1 = evaluate(epoch, test_loader, dataset, model, device, args, 'test', config)

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





