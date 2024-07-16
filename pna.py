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
from torch_frame.data.stats import StatType

from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML
from src.nn.gnn import PNA
from src.utils.loss import SSLoss
from src.utils.metric import SSMetric
from src.nn.gnn.decoder import LinkPredHead
from src.utils.batch_processing import lp_inputs

from tqdm import tqdm
import wandb

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Specify the log message format
    datefmt='%Y-%m-%d %H:%M:%S',  # Specify the date format
    handlers=[
        #logging.FileHandler('app.log'),  # Log messages to a file
        logging.StreamHandler()  # Also output log messages to the console
    ]
)
logger = logging.getLogger(__name__)
import sys

torch.set_float32_matmul_precision('high')

seed = 42
batch_size = 200
lr = 2e-4
eps = 1e-8
weight_decay = 1e-3
epochs = 30

compile = False
data_split = [0.6, 0.2, 0.2]
split_type = 'temporal'

khop_neighbors = [100, 100]
pos_sample_prob = 1
num_neg_samples = 64
channels = 128
num_layers = 3
dropout = 0.5

pretrain = {PretrainType.LINK_PRED}
testing = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'testing': testing,
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

wandb.login()
run = wandb.init(
    #dir="/mnt/data/",
    dir="/takyildiz/scratch/",
    mode="disabled" if args['testing'] else "online",
    project=f"exp", 
    name=f"PNA",
    group=f"pna,lp",
    entity="cse3000",
    #name=f"debug-fused",
    config=args
)

dataset = IBMTransactionsAML(
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv' if not testing else '/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    root='/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv' if not testing else '/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    pretrain={PretrainType.LINK_PRED},
    mask_type="replace",
    split_type=split_type, 
    splits=data_split, 
    khop_neighbors=khop_neighbors
)
dataset.materialize()
dataset.df.head(5)
train_dataset, val_dataset, test_dataset = dataset.split()

tensor_frame = dataset.tensor_frame
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=batch_size, shuffle=False, num_workers=4)
num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])

num_columns = num_numerical + num_categorical + 1
logger.info(f"num_numerical: {num_numerical}")
logger.info(f"num_categorical: {num_categorical}")
logger.info(f"num_columns: {num_columns}")

# def lp_inputs(tf: TensorFrame, tensor_frame):
    
#     edges = tf.y[:,-3:]
#     batch_size = len(edges)
#     khop_source, khop_destination, idx = dataset.sample_neighbors(edges, 'train')

#     edge_attr = tensor_frame.__getitem__(idx)

#     nodes = torch.unique(torch.cat([khop_source, khop_destination]))
#     num_nodes = nodes.shape[0]
#     node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

#     n_id_map = {value.item(): index for index, value in enumerate(nodes)}
#     local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
#     local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
#     edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

#     drop_edge_ind = torch.tensor([x for x in range(int(batch_size))])
#     mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
#     mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
#     input_edge_index = edge_index[:, ~mask]
#     input_edge_attr  = edge_attr[~mask]

#     pos_edge_index = edge_index[:, mask]
#     pos_edge_attr  = edge_attr[mask]

#     # generate/sample negative edges
#     neg_edges = []
#     target_dict = pos_edge_attr.feat_dict
#     for key, value in pos_edge_attr.feat_dict.items():
#         attr = []
#         # duplicate each row of the tensor by num_neg_samples times repeated values must be contiguous
#         for r in value:
#             if key == stype.timestamp:
#                 attr.append(r.repeat(num_neg_samples, 1, 1))
#             else:
#                 attr.append(r.repeat(num_neg_samples, 1))
#         target_dict[key] = torch.cat([target_dict[key], torch.cat(attr, dim=0)], dim=0)
#     target_edge_attr = TensorFrame(target_dict, pos_edge_attr.col_names_dict)

#     nodeset = set(range(edge_index.max()+1))
#     for i, edge in enumerate(pos_edge_index.t()):
#         src, dst = edge[0], edge[1]

#         # Chose negative examples in a smart way
#         # unavail_mask = (edge_index == src).any(dim=0) | (edge_index == dst).any(dim=0)
#         # unavail_nodes = torch.unique(edge_index[:, unavail_mask])
#         # unavail_nodes = set(unavail_nodes.tolist())
#         # avail_nodes = nodeset - unavail_nodes
#         # avail_nodes = torch.tensor(list(avail_nodes))
#         # # Finally, emmulate np.random.choice() to chose randomly amongst available nodes
#         # indices = torch.randperm(len(avail_nodes))[:num_neg_samples]
#         # neg_nodes = avail_nodes[indices]

#         # Create a mask of unavailable nodes
#         unavail_mask = torch.isin(edge_index.flatten(), torch.tensor([src, dst]))
#         unavail_nodes = edge_index.flatten()[unavail_mask].unique()
#         # Create a mask for all nodes
#         all_nodes = torch.arange(max(nodeset) + 1)
#         avail_mask = ~torch.isin(all_nodes, unavail_nodes)
#         # Get available nodes
#         avail_nodes = all_nodes[avail_mask]
#         # Randomly select negative samples from available nodes
#         neg_nodes = avail_nodes[torch.randint(high=len(avail_nodes), size=(num_neg_samples,))]
        
#         # Generate num_neg_samples/2 negative edges with the same source but different destinations
#         num_neg_samples_half = int(num_neg_samples/2)
#         neg_dsts = neg_nodes[:num_neg_samples_half]  # Selecting num_neg_samples/2 random destination nodes for the source
#         neg_edges_src = torch.stack([src.repeat(num_neg_samples_half), neg_dsts], dim=0)
        
#         # Generate num_neg_samples/2 negative edges with the same destination but different sources
#         neg_srcs = neg_nodes[num_neg_samples_half:]  # Selecting num_neg_samples/2 random source nodes for the destination
#         neg_edges_dst = torch.stack([neg_srcs, dst.repeat(num_neg_samples_half)], dim=0)

#         # Add these negative edges to the list
#         neg_edges.append(neg_edges_src)
#         neg_edges.append(neg_edges_dst)
    
#     input_edge_index = input_edge_index.to(device)
#     input_edge_attr = input_edge_attr.to(device)
#     #pos_edge_index = pos_edge_index.to(device)
#     #pos_edge_attr = pos_edge_attr.to(device)
#     node_feats = node_feats.to(device)
#     if len(neg_edges) > 0:
#         #neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
#         neg_edge_index = torch.cat(neg_edges, dim=1)
#     target_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).to(device)
#     target_edge_attr = target_edge_attr.to(device)
#     return node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr

def train_lp(loader, epoc, encoder, model, lp_decoder, optimizer, scheduler) -> float:
    encoder.train()
    model.train()
    lp_decoder.train()
    total_count = 0
    loss_lp_accum = 0

    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, num_neg_samples)
            node_feats = node_feats.to(device)
            input_edge_index = input_edge_index.to(device)
            input_edge_attr = input_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]

            pos_edge_attr = target_edge_attr[:batch_size,:]
            pos_edge_attr, _ = encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, num_columns * channels) 

            neg_edge_attr = target_edge_attr[batch_size:,:]
            neg_edge_attr, _ = encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, num_columns * channels) 

            input_edge_attr, _ = encoder(input_edge_attr)
            input_edge_attr = input_edge_attr.view(-1, num_columns * channels) 

            x, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            link_loss.backward()
            optimizer.step()

            total_count += len(tf.y)
            loss_lp_accum += link_loss.item() * len(tf.y)
            t.set_postfix(loss_lp=f'{loss_lp_accum/total_count:.4f}')
            wandb.log({"train_loss_lp": loss_lp_accum/total_count})
    return {'loss': loss_lp_accum / total_count} 

@torch.no_grad()
def eval_lp(loader: DataLoader, encoder, model, lp_decoder, dataset_name) -> float:
    encoder.eval()
    model.eval()
    lp_decoder.eval()
    mrrs = []
    hits1 = []
    hits2 = []
    hits5 = []
    hits10 = []
    loss_accum = 0
    total_count = 0
    loss_accum = loss_lp_accum = total_count = 0
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, num_neg_samples)
            node_feats = node_feats.to(device)
            input_edge_index = input_edge_index.to(device)
            input_edge_attr = input_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            pos_edge_attr, _ = encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, num_columns * channels) 

            neg_edge_attr = target_edge_attr[batch_size:,:]
            neg_edge_attr, _ = encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, num_columns * channels) 

            input_edge_attr, _ = encoder(input_edge_attr)
            input_edge_attr = input_edge_attr.view(-1, num_columns * channels)

            x, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            loss = ssloss.lp_loss(pos_pred, neg_pred)
            
            loss_lp_accum += loss * len(pos_pred)
            loss_accum += float(loss) * len(pos_pred)
            total_count += len(pos_pred)
            mrr_score, hits = ssmetric.mrr(pos_pred, neg_pred, [1,2,5,10], num_neg_samples)
            mrrs.append(mrr_score)
            hits1.append(hits['hits@1'])
            hits2.append(hits['hits@2'])
            hits5.append(hits['hits@5'])
            hits10.append(hits['hits@10'])
            t.set_postfix(
                mrr=f'{np.mean(mrrs):.4f}',
                hits1=f'{np.mean(hits1):.4f}',
                hits2=f'{np.mean(hits2):.4f}',
                hits5=f'{np.mean(hits5):.4f}',
                hits10=f'{np.mean(hits10):.4f}',
                loss_lp = f'{loss_lp_accum/total_count:.4f}',
            )
            wandb.log({
                f"{dataset_name}_loss_lp": loss_lp_accum/total_count,
            })
        mrr_score = np.mean(mrrs)
        hits1 = np.mean(hits1)
        hits2 = np.mean(hits2)
        hits5 = np.mean(hits5)
        hits10 = np.mean(hits10)
        wandb.log({
            f"{dataset_name}_mrr": mrr_score,
            f"{dataset_name}_hits@1": hits1,
            f"{dataset_name}_hits@2": hits2,
            f"{dataset_name}_hits@5": hits5,
            f"{dataset_name}_hits@10": hits10,
        })
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10}

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.timestamp: TimestampEncoder(),
}
encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
).to(device)
ssloss = SSLoss(device, num_numerical)
ssmetric = SSMetric(device)

edge_index = dataset.train_graph.edge_index
num_nodes = dataset.train_graph.num_nodes
in_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
max_in_degree = int(in_degrees.max())
in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
model = PNA(
    num_features=1, 
    num_gnn_layers=num_layers, 
    edge_dim=train_dataset.tensor_frame.num_cols*channels, 
    n_classes=1, 
    deg=in_degree_histogram,
    edge_updates=True,
)
model = torch.compile(model, dynamic=True) if compile else model
model.to(device)
num_categorical = [len(dataset.col_stats[col][StatType.COUNT][0]) for col in dataset.tensor_frame.col_names_dict[stype.categorical]] if stype.categorical in dataset.tensor_frame.col_names_dict else 0
lp_decoder = LinkPredHead(n_classes=1, n_hidden=channels, dropout=dropout).to(device)

learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in lp_decoder.parameters() if p.requires_grad)
logger.info(f"learnable_params: {learnable_params}")
wandb.log({"learnable_params": learnable_params})

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in lp_decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in lp_decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=2*lr, step_size_up=2000, cycle_momentum=False)

save_dir = '/takyildiz/scratch/saved_models'
run_id = wandb.run.id
os.makedirs(save_dir, exist_ok=True)
best_lp = 0

for epoch in range(1, epochs + 1):
    logger.info(f"Epoch {epoch}:")
    lp_loss = train_lp(train_loader, epoch, encoder, model, lp_decoder, optimizer, scheduler)
    logger.info(f"loss_lp: {lp_loss}")

    val_lp = eval_lp(val_loader, encoder, model, lp_decoder, "val")
    logger.info(f"val_lp: {val_lp}")

    test_lp = eval_lp(test_loader, encoder, model, lp_decoder, "test")
    logger.info(f"test_lp: {test_lp}")
    if test_lp['mrr'] > best_lp and not testing:
        model_save_path = os.path.join(save_dir, f'{run_id}_mrr.pth')
        best_lp = test_lp['mrr']
        torch.save(model.state_dict(), model_save_path)
        logger.info(f'Best MRR model saved to {model_save_path}')

wandb.finish()