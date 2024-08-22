import sys
import logging
import wandb
from tqdm import tqdm as tqdm

#from torch_frame.data import DataLoader
from torch.utils.data import DataLoader
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
import torch
from torch_geometric.utils import degree
logging.getLogger('torch').setLevel(logging.WARNING)

from src.datasets import IBMTransactionsAML, EthereumPhishing, EllipticBitcoin, OgbnArxiv, MusaeGitHub, LastFMAsia, WikiSquirrel, WikiChameleon, Facebook
from sklearn.metrics import f1_score
from utils import *

# workaround for CUDA invalid configuration bug
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@torch.no_grad()
def evaluate(loader, dataset, model, device, args, mode, config):
    model.eval()

    '''Evaluates the model performance '''
    preds = []
    ground_truths = []
    for batch in tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        batch_size = len(batch[0])
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
    if config['n_classes'] == 2:
        f1 = f1_score(ground_truth, pred)
    else:
        f1 = f1_score(ground_truth, pred, average='micro')
    acc = (pred == ground_truth).mean()
    # wandb.log({f"{mode}_acc": acc}, step=epoch)
    # logging.info(f'{mode} acc: {acc:.4f}')

    model.train()
    return f1, acc

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
    "task": "edge_classification"
}

#define a model config dictionary and wandb logging at the same time
print("Testing: ", args.testing)
wandb.init(
    dir=args.wandb_dir,
    mode="disabled" if args.testing else "online",
    project="rel-mm-supervised-mcm", #replace this with your wandb project name if you want to use wandb logging
    group=args.group,
    entity="cse3000",
    config=config
)

config['experiment_path'] = args.wandb_dir
logger_setup()

if 'ethereum-phishing-transaction-network' in config['data']:
    config['lr'] = 0.0008
    config['dropout'] = 0.123
    config['w_ce2'] = 1.16
    config['n_gnn_layers'] = 2
    config['n_hidden'] = 32
    config['task'] = 'node_classification'
    dataset = EthereumPhishing(
        root=config['data'],
        split_type='temporal_daily', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
elif 'elliptic_bitcoin_dataset' in config['data']:
    config['task'] = 'node_classification'
    dataset = EllipticBitcoin(
        root=config['data'],
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
elif 'ibm-transactions-for-anti-money-laundering-aml' in config['data']:
    config['task'] = 'edge_classification'
    config['lr'] = 0.0005
    config['dropout'] = 0.10527690625126304
    config['w_ce2'] = 6
    config['n_gnn_layers'] = 2
    config['n_hidden'] = 64
    dataset = IBMTransactionsAML(
        root=config['data'],
        split_type='temporal_daily', 
        splits=[0.6, 0.2, 0.2], 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        channels=config['n_hidden'],
    )
elif 'ogbn_arxiv' in config['data']:
    config['task'] = 'node_classification'
    config['n_classes'] = 40
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
    dataset = OgbnArxiv(
        root=config['data'],
        split_type='temporal', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
elif 'git_web_ml' in config['data']:
    config['task'] = 'node_classification'
    config['n_gnn_layers'] = 4
    dataset = MusaeGitHub(
        root=config['data'],
        split_type='random', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        splits=[0.7, 0.15, 0.15],
        channels=config['n_hidden']
    )
    config['emb'] = dataset.emb
elif 'squirrel' in config['data']:
    config['task'] = 'node_classification'
    config['n_hidden'] = 64
    config['n_gnn_layers'] = 4
    print(args.num_neighs)
    dataset = WikiSquirrel(
        root=config['data'],
        split_type='random', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        splits=[0.6, 0.2, 0.2],
        channels=config['n_hidden']
    )
    config['n_classes'] = dataset.num_classes
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
elif 'chameleon' in config['data']:
    config['task'] = 'node_classification'
    config['num_gnn_layers'] = 2
    dataset = WikiChameleon(
        root=config['data'],
        split_type='random', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        splits=[0.6, 0.2, 0.2],
        channels=config['n_hidden']
    )
    #config['dropout'] = 0.5
    config['n_classes'] = dataset.num_classes
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
elif 'facebook' in config['data']:
    config['task'] = 'node_classification'
    config['num_gnn_layers'] = 2
    dataset = Facebook(
        root=config['data'],
        split_type='random', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        splits=[0.6, 0.2, 0.2],
        channels=config['n_hidden']
    )
    #config['dropout'] = 0.5
    config['n_classes'] = dataset.num_classes
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
elif 'lastfm_asia' in config['data']:
    config['task'] = 'node_classification'
    config['n_hidden'] = 8
    dataset = LastFMAsia(
        root=config['data'],
        split_type='random', 
        khop_neighbors=args.num_neighs,
        ports=args.ports,
        ego=args.ego,
        channels=config['n_hidden']
    )
    config['n_classes'] = dataset.num_classes
    config['loss_weights'] = [1 for _ in range(config['n_classes'])]
else:
    raise ValueError("Invalid data name!")
nodes = dataset.nodes
edges = dataset.edges

if config['task'] == 'node_classification':
    train_dataset, val_dataset, test_dataset = dataset.split()
else:
    train_dataset, val_dataset, test_dataset = edges.split()

if config['model'] == 'pna' or config['model'] == 'tabgnn' or config['model'] == 'tabgnnfused':
    edge_index = edges.train_graph.edge_index
    num_nodes = edges.train_graph.num_nodes
    config["in_degrees"] = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)

#train_loader = DataLoader(train_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=True, num_workers=4)
#val_loader = DataLoader(val_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)
#test_loader = DataLoader(test_dataset.tensor_frame, batch_size=config['batch_size'], shuffle=False, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

num_misc = len(edges.tensor_frame.col_names_dict[stype.relation]) if stype.relation in edges.tensor_frame.col_names_dict else 0
num_numerical = len(edges.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in edges.tensor_frame.col_names_dict else 0
num_categorical = len(edges.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in edges.tensor_frame.col_names_dict else 0
num_timestamp = len(edges.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in edges.tensor_frame.col_names_dict else 0
num_columns = num_numerical + num_categorical + num_timestamp + num_misc
config['num_edge_features'] = num_columns
logging.info(f"Number of edge features: {num_columns}")

# num_misc = len(nodes.tensor_frame.col_names_dict[stype.relation]) if stype.relation in nodes.tensor_frame.col_names_dict else 0
# num_numerical = len(nodes.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in nodes.tensor_frame.col_names_dict else 0
# num_categorical = len(nodes.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in nodes.tensor_frame.col_names_dict else 0
# num_timestamp = len(nodes.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in nodes.tensor_frame.col_names_dict else 0
#config['num_node_features'] = num_numerical + num_categorical + num_timestamp + num_misc
config['num_node_features'] = nodes.num_features
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
    #model = torch.compile(model)
else:
    raise ValueError("Invalid model name!")

#loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config['w_ce1'], config['w_ce2']]).to(device))
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(config['loss_weights']).to(device))

# cast model parameters to bflat16
# model = model.half()
# print number of parameters
num_params = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_params}")
# print number of emb parameters
num_emb_params = sum(p.numel() for p in dataset.emb.parameters())
logging.info(f"Number of emb parameters: {num_emb_params}")

logging.info(f'Total number of parameters {num_params + num_emb_params}')
def parameter_breakdown(model):
    total_params = 0
    print(f"{'Layer':<40} {'Shape':<30} {'Params':<15}")
    print("-" * 90)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name:<40} {str(list(param.shape)):<30} {layer_params:<15}")

    print(f"\nTotal Trainable Parameters: {total_params:,}")

parameter_breakdown(model)

# pass mode and dataset.emb (type torch.Embedding) parameters to the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': dataset.emb.parameters()}
], lr=config['lr'])
# import sys
# sys.exit()

from torch.profiler import profile, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

best_val_f1 = 0
best_te_f1 = 0
best_val_acc = 0
best_te_acc = 0
for epoch in range(config['epochs']):
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    for batch in tqdm(train_loader, disable=not args.tqdm):

        optimizer.zero_grad()
        batch_size = len(batch[1])

        node_feats, edge_index, edge_attr, y, mask = dataset.get_graph_inputs(batch, mode='train', args=args)
        # print gpu memory allocation before and after the operation
        #print(torch.cuda.memory_summary(device=device, abbreviated=False))
        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
        #print(torch.cuda.memory_summary(device=device, abbreviated=False))
        #print shapes of all the tensors
        # print(f"Node Feats: {node_feats.shape}")
        # print(f"Edge Index: {edge_index.shape}")
        # print(f"Edge Attr: {edge_attr.shape}")
        # print(f"Y: {y.shape}")

        # with autocast():
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        pred = model(node_feats, edge_index, edge_attr)[:batch_size]
        #print(torch.cuda.memory_summary(device=device, abbreviated=False))
        if mask is not None:
            pred = pred[mask]
            y = y[mask]

        preds.append(pred.argmax(dim=-1))
        ground_truths.append(y)
        loss = loss_fn(pred.float(), y.view(-1).to(torch.long))
        
        loss.backward()
        optimizer.step()
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5))
        #sys.exit()
        # display memory usage for each operation

        # print(torch.cuda.memory_summary(device=device, abbreviated=False))
        # # Scales the loss and performs backpropagation
        # scaler.scale(loss).backward()
        
        # # Update model parameters
        # scaler.step(optimizer)
        
        # # Update the scale for next iteration
        # scaler.update()

        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()
            
    pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    logging.info(f'Train Loss: {total_loss/total_examples:.4f}')
    if config['n_classes'] == 2:
        f1 = f1_score(ground_truth, pred)
    else:
        f1 = f1_score(ground_truth, pred, average='micro')
    acc = (pred == ground_truth).mean()
    wandb.log({"acc/train": acc}, step=epoch)
    wandb.log({"f1/train": f1}, step=epoch)
    wandb.log({"Loss": total_loss/total_examples}, step=epoch)
    logging.info(f'Train F1: {f1:.4f}, Epoch: {epoch}')
    logging.info(f'Train Acc: {acc:.4f}')

    # evaluate
    val_f1, val_acc = evaluate(val_loader, dataset, model, device, args, 'val', config)
    #val_f1, val_acc = 0, 0
    logging.info(f'Validation Acc: {val_acc:.4f}')
    logging.info(f'Validation F1: {val_f1:.4f}')
    wandb.log({"acc/validation": val_acc}, step=epoch)
    wandb.log({"f1/validation": val_f1}, step=epoch)

    te_f1, te_acc = evaluate(test_loader, dataset, model, device, args, 'test', config)
    wandb.log({"acc/test": te_acc}, step=epoch)
    wandb.log({"f1/test": te_f1}, step=epoch)
    logging.info(f'Test Acc: {te_acc:.4f}')
    logging.info(f'Test F1: {te_f1:.4f}')

    if val_f1 > best_val_f1:
    #if te_f1 > best_te_f1:
        best_val_f1 = val_f1
        best_te_f1 = te_f1
        wandb.log({"best_test_f1": te_f1}, step=epoch)
        # if args.save_model:
        #     save_model(model, optimizer, epoch, config)
    logging.info(f'Best Test F1: {best_te_f1:.4f}')

    if val_acc > best_val_acc:
    #if te_acc > best_te_acc:
        best_val_acc = val_acc
        best_te_acc = te_acc
        wandb.log({"best_test_acc": te_acc}, step=epoch)
    logging.info(f'Best Acc: {best_te_acc:.4f}')




