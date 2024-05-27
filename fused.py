# %%
import os
import numpy as np
import random

import torch
import torch.nn.functional as F

from src.datasets.util.mask import PretrainType
from torch_frame.data import DataLoader
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame import NAStrategy
from torch_frame import TensorFrame

from src.datasets import IBMTransactionsAML
from src.nn.models import FTTransformerGINeFused
from src.utils.loss import SSLoss
from src.utils.metric import SSMetric

from tqdm import tqdm
import wandb

from icecream import ic
import sys

torch.set_float32_matmul_precision('high')

# %%
seed = 42
batch_size = 200
lr = 2e-4
eps = 1e-8
weight_decay = 1e-3
epochs = 3

compile = False
data_split = [0.6, 0.2, 0.2]
split_type = 'temporal'

khop_neighbors = [100, 100]
pos_sample_prob = 1
num_neg_samples = 64
channels = 128
num_layers = 3
dropout = 0.5

pretrain = {PretrainType.MASK, PretrainType.MASK_VECTOR, PretrainType.LINK_PRED}
#pretrain = 'lp'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'testing': False,
    'batch_size': batch_size,
    'seed': seed,
    'device': device,
    'lr': lr,
    'eps': eps,
    'epochs': epochs,
    'compile': compile,
    'data_split': data_split,
    'pos_sample_prob': pos_sample_prob,
    'channels': channels,
    'split_type': split_type,
    'num_neg_samples': num_neg_samples,
    'pretrain': pretrain,
    'khop_neighbors': khop_neighbors,
    'num_layers': num_layers,
    'dropout': dropout,
    'weight_decay': weight_decay,
}

# %%
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

# %%
wandb.login()
run = wandb.init(
    # dir="/mnt/data/",
    mode="disabled" if args['testing'] else "online",
    project=f"rel-mm", 
    name=f"model=FTTransformerGINeFused,dataset=IBM-AML_Hi_Sm,objective=MCM+MV+LP,only-last-layer-fuse,fusenorm,correct",
    #name=f"debug-fused",
    config=args
)

# %%
dataset = IBMTransactionsAML(
    root='/scratch/imcauliffe/AML_dataset_preprocessed/HI-Small_Trans-c.csv',
    # root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv', 
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    pretrain=pretrain, 
    split_type=split_type, 
    splits=data_split, 
    khop_neighbors=khop_neighbors
)
ic(dataset)
dataset.materialize()
dataset.df.head(5)
train_dataset, val_dataset, test_dataset = dataset.split()
ic(len(train_dataset), len(val_dataset), len(test_dataset))

# %%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
g = torch.Generator()
g.manual_seed(seed)
tensor_frame = dataset.tensor_frame
train_tensor_frame = train_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_tensor_frame = val_dataset.tensor_frame
val_loader = DataLoader(val_tensor_frame, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
test_tensor_frame = test_dataset.tensor_frame
test_loader = DataLoader(test_tensor_frame, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

# %%
num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
num_columns = num_numerical + num_categorical
ic(num_numerical, num_categorical, num_columns)

ssloss = SSLoss(device, num_numerical)
ssmetric = SSMetric(device)

# %%
stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.timestamp: TimestampEncoder(na_strategy=NAStrategy.OLDEST_TIMESTAMP),
}
encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=dataset.col_stats,
            col_names_dict=train_tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
)

# %%
def create_empty_rows(col_names_dict, num_rows):
    feat_dict = {}
    for stype, col_names in col_names_dict.items():
        if stype == stype.categorical:
            feat_dict[stype] = torch.full((num_rows, len(col_names)), -1)
        elif stype == stype.numerical:
            feat_dict[stype] = torch.full((num_rows, len(col_names)), float('NaN'))
        elif stype == stype.timestamp:
            #feat_dict[stype] = torch.full((num_rows, len(col_names)), -1)
            #hack to get the timestamp encoder to work
            row_values = [[1970, 0, 0, 3, 0, 0, 0]]
            feat_dict[stype] = torch.tensor([row_values] * num_rows)
    return feat_dict

def inputs(tf: TensorFrame, pos_sample_prob=1, train=True):   
    edges = tf.y
    batch_size = len(edges)
    # ic(edges[:, 2:])
    # ic(edges[:, :2])
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges[:, 2:], train)

    edge_data = tensor_frame.__getitem__(idx)
    #ic(edge_data.feat_dict[stype.timestamp][0:5])

    edge_attr, col_names = encoder(edge_data)
    edge_attr = edge_attr.view(-1, len(col_names) * channels)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0] + 1 # add interaction node
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index+1 for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))
    # ic(edge_index.shape, edge_attr.shape)

    # sample positive edges
    positions = torch.arange(batch_size)
    num_samples = int(len(positions) * pos_sample_prob)
    if len(positions) > 0 and num_samples > 0:
        drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), num_samples, replacement=False)
    else:
        drop_idxs = torch.tensor([]).long()
    drop_edge_ind = positions[drop_idxs]

    mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
    mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
    
    # add interaction node to the graph
    unique_nodes_ids = torch.unique(edges[:, 2:].t()[0:2].flatten())
    local_unique_nodes_ids = torch.tensor([n_id_map[node.item()] for node in unique_nodes_ids], dtype=torch.long)
    # ic(unique_nodes_ids.shape)
    int_edges = torch.stack([local_unique_nodes_ids, torch.tensor([0] * local_unique_nodes_ids.shape[0])], dim=0)
    int_edge_attr = TensorFrame(create_empty_rows(tensor_frame.col_names_dict, local_unique_nodes_ids.shape[0]), tensor_frame.col_names_dict)
    input_edge_index = torch.cat([int_edges, edge_index[:, ~mask]], dim=1)
    input_edge_attr  = torch.cat([encoder(int_edge_attr)[0].view(-1, len(col_names) * channels), edge_attr[~mask]], dim=0)
    # ic(input_edge_index.shape, input_edge_attr.shape)

    pos_edge_index = edge_index[:, mask]
    pos_edge_attr  = edge_attr[mask]

    # generate/sample negative edges
    neg_edges = []
    neg_edge_attr = []
    nodeset = set(range(edge_index.max()+1))
    for i, edge in enumerate(pos_edge_index.t()):
        src, dst = edge[0], edge[1]

        # Chose negative examples in a smart way
        unavail_mask = (edge_index == src).any(dim=0) | (edge_index == dst).any(dim=0)
        unavail_nodes = torch.unique(edge_index[:, unavail_mask])
        unavail_nodes = set(unavail_nodes.tolist())
        avail_nodes = nodeset - unavail_nodes
        avail_nodes = torch.tensor(list(avail_nodes))
        # Finally, emmulate np.random.choice() to chose randomly amongst available nodes
        indices = torch.randperm(len(avail_nodes))[:num_neg_samples]
        neg_nodes = avail_nodes[indices]
        
        # Generate num_neg_samples/2 negative edges with the same source but different destinations
        num_neg_samples_half = int(num_neg_samples/2)
        neg_dsts = neg_nodes[:num_neg_samples_half]  # Selecting num_neg_samples/2 random destination nodes for the source
        neg_edges_src = torch.stack([src.repeat(num_neg_samples_half), neg_dsts], dim=0)
        
        # Generate num_neg_samples/2 negative edges with the same destination but different sources
        neg_srcs = neg_nodes[num_neg_samples_half:]  # Selecting num_neg_samples/2 random source nodes for the destination
        neg_edges_dst = torch.stack([neg_srcs, dst.repeat(num_neg_samples_half)], dim=0)

        # Add these negative edges to the list
        neg_edges.append(neg_edges_src)
        neg_edges.append(neg_edges_dst)
        # Replicate the positive edge attribute for each of the negative edges generated from this edge
        pos_attr = pos_edge_attr[i].unsqueeze(0)  # Get the attribute of the current positive edge
        
        replicated_attr = pos_attr.repeat(num_neg_samples, 1)  # Replicate it num_neg_samples times (for each negative edge)
        neg_edge_attr.append(replicated_attr)
    
    input_edge_index = input_edge_index.to(device)
    input_edge_attr = input_edge_attr.to(device)
    pos_edge_index = pos_edge_index.to(device)
    pos_edge_attr = pos_edge_attr.to(device)
    node_feats = node_feats.to(device)
    neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
    neg_edge_attr = torch.cat(neg_edge_attr, dim=0).to(device)
    return node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr

def lp_inputs(tf: TensorFrame, pos_sample_prob=1, train=True):
    
    edges = tf.y[:, 2:]
    batch_size = len(edges)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, train)
    #del edges

    edge_data = tensor_frame.__getitem__(idx)
    edge_attr, col_names = encoder(edge_data)
    edge_attr = edge_attr.view(-1, len(col_names) * channels)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    # sample positive edges
    positions = torch.arange(batch_size)
    num_samples = int(len(positions) * pos_sample_prob)
    if len(positions) > 0 and num_samples > 0:
        drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), num_samples, replacement=False)
    else:
        drop_idxs = torch.tensor([]).long()
    drop_edge_ind = positions[drop_idxs]

    mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
    mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
    input_edge_index = edge_index[:, ~mask]
    input_edge_attr  = edge_attr[~mask]

    pos_edge_index = edge_index[:, mask]
    pos_edge_attr  = edge_attr[mask]

    # generate/sample negative edges
    neg_edges = []
    neg_edge_attr = []
    nodeset = set(range(edge_index.max()+1))
    for i, edge in enumerate(pos_edge_index.t()):
        src, dst = edge[0], edge[1]

        # Chose negative examples in a smart way
        unavail_mask = (edge_index == src).any(dim=0) | (edge_index == dst).any(dim=0)
        unavail_nodes = torch.unique(edge_index[:, unavail_mask])
        unavail_nodes = set(unavail_nodes.tolist())
        avail_nodes = nodeset - unavail_nodes
        avail_nodes = torch.tensor(list(avail_nodes))
        # Finally, emmulate np.random.choice() to chose randomly amongst available nodes
        indices = torch.randperm(len(avail_nodes))[:num_neg_samples]
        neg_nodes = avail_nodes[indices]
        
        # Generate num_neg_samples/2 negative edges with the same source but different destinations
        num_neg_samples_half = int(num_neg_samples/2)
        neg_dsts = neg_nodes[:num_neg_samples_half]  # Selecting num_neg_samples/2 random destination nodes for the source
        neg_edges_src = torch.stack([src.repeat(num_neg_samples_half), neg_dsts], dim=0)
        
        # Generate num_neg_samples/2 negative edges with the same destination but different sources
        neg_srcs = neg_nodes[num_neg_samples_half:]  # Selecting num_neg_samples/2 random source nodes for the destination
        neg_edges_dst = torch.stack([neg_srcs, dst.repeat(num_neg_samples_half)], dim=0)

        # Add these negative edges to the list
        neg_edges.append(neg_edges_src)
        neg_edges.append(neg_edges_dst)
        # Replicate the positive edge attribute for each of the negative edges generated from this edge
        pos_attr = pos_edge_attr[i].unsqueeze(0)  # Get the attribute of the current positive edge
        
        replicated_attr = pos_attr.repeat(num_neg_samples, 1)  # Replicate it num_neg_samples times (for each negative edge)
        neg_edge_attr.append(replicated_attr)
    
    input_edge_index = input_edge_index.to(device)
    input_edge_attr = input_edge_attr.to(device)
    pos_edge_index = pos_edge_index.to(device)
    pos_edge_attr = pos_edge_attr.to(device)
    node_feats = node_feats.to(device)
    neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
    neg_edge_attr = torch.cat(neg_edge_attr, dim=0).to(device)
    return node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr


def optimizer_step(optimizers, losses):
    if len(optimizers) != len(losses):
        raise ValueError("Number of optimizers and losses should be the same")
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
        losses = [losses]
    #for i, (optimizer, loss) in enumerate(zip(optimizers, losses)):
    for i, (optimizer, loss) in enumerate(zip(optimizers, losses)):
        optimizer.zero_grad()
        loss.backward(retain_graph=(i < len(optimizers)-1))
        optimizer.step()

def train(epoc: int, model, optimizer, scheduler) -> float:
    model.train()
    loss_accum = total_count = 0
    loss_accum = loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    with tqdm(train_loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            # Get the updated learning rate and log it
            #updated_lr = optimizer.param_groups[0]['lr']
            #wandb.log({"lr": updated_lr})
            node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf)
            tf = tf.to(device)
            num_pred, cat_pred, pos_pred, neg_pred = model(tf, node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            loss = link_loss + t_loss
            #optimizer_step(optimizer, [link_loss, t_loss])
            optimizer_step(optimizer, [loss])
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #scheduler.step()

            loss_accum += (loss.item() * len(pos_pred))
            total_count += len(pos_pred)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(pos_pred)
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}')
            del pos_pred
            del neg_pred
            del num_pred
            del cat_pred
            del edge_index
            del edge_attr
            del tf
            del loss
            del link_loss
            del t_loss
            del loss_c
            del loss_n
            wandb.log({"train_loss": loss_accum/total_count, "train_loss_lp": loss_lp_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n})
        #optimizer.state.clear()
    return {'loss': loss_accum / total_count}

@torch.no_grad()
def test(loader: DataLoader, model, dataset_name) -> float:
    model.eval()
    mrrs = []
    hits1 = []
    hits2 = []
    hits5 = []
    hits10 = []
    loss_accum = 0
    total_count = 0
    accum_acc = accum_l2 = 0
    loss_accum = loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    t_n = t_c = 0

    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf, train=False)
            tf = tf.to(device)
            num_pred, cat_pred, pos_pred, neg_pred = model(tf, node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            loss = link_loss + t_loss
            
            t_c += loss_c[1]
            t_n += loss_n[1] 
            loss_c_accum += loss_c[0]
            loss_n_accum += loss_n[0]
            loss_lp_accum += link_loss * len(pos_pred)
            loss_accum += float(loss) * len(pos_pred)
            total_count += len(pos_pred)
            mrr_score, hits = ssmetric.mrr(pos_pred, neg_pred, [1,2,5,10], num_neg_samples)
            mrrs.append(mrr_score)
            hits1.append(hits['hits@1'])
            hits2.append(hits['hits@2'])
            hits5.append(hits['hits@5'])
            hits10.append(hits['hits@10'])
            for i, ans in enumerate(tf.y):
                # ans --> [val, idx]
                if ans[1] > (num_numerical-1):
                    accum_acc += (cat_pred[int(ans[1])-num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - num_pred[i][int(ans[1])]) #rmse
            t.set_postfix(
                loss=f'{loss_accum/total_count:.4f}',
                mrr=f'{np.mean(mrrs):.4f}',
                hits1=f'{np.mean(hits1):.4f}',
                hits2=f'{np.mean(hits2):.4f}',
                hits5=f'{np.mean(hits5):.4f}',
                hits10=f'{np.mean(hits10):.4f}',
                accuracy=f'{accum_acc / t_c:.4f}',
                rmse=f'{torch.sqrt(accum_l2 / t_n):.4f}', 
                loss_mcm=f'{(loss_c_accum/t_c) + (loss_n_accum/t_n):.4f}',
                loss_c = f'{loss_c_accum/t_c:.4f}', 
                loss_n = f'{loss_n_accum/t_n:.4f}',
                loss_lp = f'{loss_lp_accum/total_count:.4f}',
            )
            del tf
            del pos_pred
            del neg_pred
            del num_pred
            del cat_pred
            del edge_index
            del edge_attr
            wandb.log({
                f"{dataset_name}_loss": loss_accum/total_count,
                f"{dataset_name}_loss_c": loss_c_accum/t_c,
                f"{dataset_name}_loss_n": loss_n_accum/t_n,
                f"{dataset_name}_loss_lp": loss_lp_accum/total_count,
            })
        mrr_score = np.mean(mrrs)
        hits1 = np.mean(hits1)
        hits2 = np.mean(hits2)
        hits5 = np.mean(hits5)
        hits10 = np.mean(hits10)
        accuracy = accum_acc / t_c
        rmse = torch.sqrt(accum_l2 / t_n)
        wandb.log({
            f"{dataset_name}_mrr": mrr_score,
            f"{dataset_name}_hits@1": hits1,
            f"{dataset_name}_hits@2": hits2,
            f"{dataset_name}_hits@5": hits5,
            f"{dataset_name}_hits@10": hits10,
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_rmse": rmse,
        })
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10, "accuracy": accuracy, "rmse": rmse}

# %%
model = FTTransformerGINeFused(
    channels=channels,
    out_channels=None,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    edge_dim=channels*train_dataset.tensor_frame.num_cols,
    num_layers=num_layers, 
    dropout=dropout,
    pretrain=True
)
model = torch.compile(model, dynamic=True) if compile else model
model.to(device)
learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
ic(learnable_params)
wandb.log({"learnable_params": learnable_params})

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
params_lp = [
        {'params': [param for name, param in model.named_parameters() if 'tab_conv' not in name and 'mcm_decoder' not in name and not any(nd in name for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [param for name, param in model.named_parameters() if 'tab_conv' not in name and 'mcm_decoder' not in name and any(nd in name for nd in no_decay)], 'weight_decay': 0.0},
]
#params_lp = [param for name, param in model.named_parameters() if 'tab_conv' not in name and 'mcm_decoder' not in name]
params_mcm = [
        {'params': [param for name, param in model.named_parameters() if 'gnn_conv' not in name and 'lp_decoder' not in name and not any(nd in name for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [param for name, param in model.named_parameters() if 'gnn_conv' not in name and 'lp_decoder' not in name and any(nd in name for nd in no_decay)], 'weight_decay': 0.0},
]
#params_mcm = [param for name, param in model.named_parameters() if 'lp_decoder' not in name and 'gnn_conv' not in name]

optimizer_mcm = torch.optim.AdamW(params_mcm, lr=lr, eps=eps, weight_decay=weight_decay)
optimizer_lp = torch.optim.AdamW(params_lp, lr=lr, eps=eps, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_mcm, base_lr=lr, max_lr=2*lr, step_size_up=2000, cycle_momentum=False)

# train_metric = test(train_loader, model, "tr")
# val_metric = test(val_loader, model, "val")
# test_metric = test(test_loader, model, "test")
# ic(
#         train_metric,
#         val_metric, 
#         test_metric
# )

torch.autograd.set_detect_anomaly(False)
for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, [optimizer], scheduler)
    #train_loss = train(epoch, model, [optimizer_lp, optimizer_mcm], scheduler)
    train_metric = test(train_loader, model, "tr")
    val_metric = test(val_loader, model, "val")
    test_metric = test(test_loader, model, "test")
    ic(
        train_loss, 
        train_metric, 
        val_metric, 
        test_metric
    )
# Create a directory to save models
save_dir = '.cache/saved_models'
run_id = wandb.run.id
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, f'latest_model_run_{run_id}.pth')

# Save the model after each epoch, replacing the old model
torch.save(model.state_dict(), model_save_path)
ic(f'Model saved to {model_save_path}')
# %%
wandb.finish()


