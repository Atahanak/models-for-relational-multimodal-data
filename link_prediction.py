# %%
import os

import torch
import torch.nn.functional as F
import numpy as np
import random

from torch_frame.data import DataLoader
from torch_frame import TensorFrame
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from torch_geometric.data import Data
from transformers import get_inverse_sqrt_schedule

from src.datasets import IBMTransactionsAML
from src.nn.gnn.model import GINe
from src.utils.loss import lp_loss
from src.utils.metric import mrr

from tqdm import tqdm
import wandb
from icecream import ic
import sys

torch.set_float32_matmul_precision('high')

# %%
seed = 42
batch_size = 200
lr = 5e-4
eps = 1e-8
epochs = 1

compile = False
data_split = [0.6, 0.2, 0.2]
split_type = 'temporal'

khop_neighbors = [100, 100]
pos_sample_prob = 0.15
num_neg_samples = 64
channels = 128

pretrain = 'lp'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'testing': True,
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
    mode="disabled" if args['testing'] else "online",
    project=f"rel-mm", 
    #name="model=GINe,dataset=IBM-AML_Hi_Sm,objective=lp,khop_neighs=[100,100],channels=32,epoch=3",
    name="debug-temporal-LOL-channels256-LOL",
    config=args
)

# %%
dataset = IBMTransactionsAML(
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv', 
    root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
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

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.timestamp: TimestampEncoder(),
}
encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=dataset.col_stats,
            col_names_dict=train_tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
)
def lp_inputs(tf: TensorFrame, pos_sample_prob=0.15, train=True):
    
    edges = tf.y
    batch_size = len(edges)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, train)

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
# batch = next(iter(train_loader))
# node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(batch)
# ic( node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
# ic( node_feats.shape, edge_index.shape, edge_attr.shape, input_edge_index.shape, input_edge_attr.shape, pos_edge_index.shape, pos_edge_attr.shape, neg_edge_index.shape, neg_edge_attr.shape)

# %%
def train(epoc: int, model, optimizer) -> float:
    model.train()
    loss_accum = total_count = 0

    with tqdm(train_loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            node_feats, _, _, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf)
            pred, neg_pred = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            loss = lp_loss(pred, neg_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_accum += float(loss) * len(pred)
            total_count += len(pred)
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}')
            del pred
            del tf
        wandb.log({"train_loss": loss_accum/total_count})
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
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            node_feats, _, _, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf, train=False)
            pred, neg_pred = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            loss = lp_loss(pred, neg_pred)
            loss_accum += float(loss) * len(pred)
            total_count += len(pred)
            mrr_score, hits = mrr(pred, neg_pred, [1,2,5,10], num_neg_samples)
            mrrs.append(mrr_score)
            hits1.append(hits['hits@1'])
            hits2.append(hits['hits@2'])
            hits5.append(hits['hits@5'])
            hits10.append(hits['hits@10'])
            t.set_postfix(
                loss=f'{loss_accum/total_count:.4f}',
                mrr=f'{np.mean(mrrs):.4f}',
                hits1=f'{np.mean(hits1):.4f}',
                hits2=f'{np.mean(hits2):.4f}',
                hits5=f'{np.mean(hits5):.4f}',
                hits10=f'{np.mean(hits10):.4f}'
            )
        mrr_score = np.mean(mrrs)
        hits1 = np.mean(hits1)
        hits2 = np.mean(hits2)
        hits5 = np.mean(hits5)
        hits10 = np.mean(hits10)
        wandb.log({
            f"{dataset_name}_loss": loss_accum/total_count,
            f"{dataset_name}_mrr": mrr_score,
            f"{dataset_name}_hits@1": hits1,
            f"{dataset_name}_hits@2": hits2,
            f"{dataset_name}_hits@5": hits5,
            f"{dataset_name}_hits@10": hits10
        })
        del tf
        del pred
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10}

# %%
torch.autograd.set_detect_anomaly(True)
model = GINe(num_features=1, num_gnn_layers=3, edge_dim=train_dataset.tensor_frame.num_cols*channels, n_classes=1)
model = torch.compile(model, dynamic=True) if compile else model
model.to(device)
learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
ic(learnable_params)
wandb.log({"learnable_params": learnable_params})

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0, timescale=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# train_metric = test(train_loader, model, "train")
# val_metric = test(val_loader, model, "val")
# test_metric = test(test_loader, model, "test")
# ic(
#         train_metric, 
#         val_metric, 
#         test_metric
# )

for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, optimizer)
    train_metric = test(train_loader, model, "train")
    val_metric = test(val_loader, model, "val")
    test_metric = test(test_loader, model, "test")
    ic(
        train_loss, 
        train_metric, 
        val_metric, 
        test_metric
    )

# %%
wandb.finish()


