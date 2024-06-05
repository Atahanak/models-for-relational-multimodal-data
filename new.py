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

from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML
from src.nn.models import FTTransformerPNAFused
from src.utils.loss import SSLoss
from src.utils.metric import SSMetric
from src.nn.weighting.MoCo import MoCoLoss

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
epochs = 30

compile = False
data_split = [0.6, 0.2, 0.2]
split_type = 'temporal'

khop_neighbors = [100, 100]
pos_sample_prob = 0.5
num_neg_samples = 64
channels = 128
num_layers = 3
dropout = 0.5

pretrain = {PretrainType.MASK, PretrainType.LINK_PRED}
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

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

wandb.login()
run = wandb.init(
    dir="/mnt/data/",
    mode="disabled" if args['testing'] else "online",
    project=f"rel-mm-fix", 
    name=f"pos_sample_prob={pos_sample_prob}",
    #group=f"last-layer-notfused,moco2",
    entity="cse3000",
    #name=f"debug-fused",
    config=args
)

dataset = IBMTransactionsAML(
    #root='/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv', 
    root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv', 
    #root='/home/takyildiz/cse3000/data/Over-Sampled_Tiny_Trans-c.csv', 
    #root='/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    #root='/home/dragomir/Downloads/dummy-100k-random-c.csv', 
    #root='/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv', 
    pretrain=pretrain,
    mask_type="replace",
    split_type=split_type, 
    splits=data_split, 
    khop_neighbors=khop_neighbors
)
ic(dataset)
dataset.materialize()
dataset.df.head(5)
train_dataset, val_dataset, test_dataset = dataset.split()
ic(len(train_dataset), len(val_dataset), len(test_dataset))

edge_index = dataset.train_graph.edge_index
num_nodes = dataset.train_graph.num_nodes

# Compute the in-degree for each node
in_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)

# Compute the maximum in-degree
max_in_degree = int(in_degrees.max())

# Create a histogram tensor for in-degrees
in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())

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

num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
num_columns = num_numerical + num_categorical
ic(num_numerical, num_categorical, num_columns)

ssloss = SSLoss(device, num_numerical)
ssmetric = SSMetric(device)

model = FTTransformerPNAFused(
    channels=channels,
    out_channels=None,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    edge_dim=channels*train_dataset.tensor_frame.num_cols,
    num_layers=num_layers, 
    dropout=dropout,
    pretrain=True,
    deg=in_degree_histogram
)
model = torch.compile(model, dynamic=True) if compile else model
model.to(device)

def lp_inputs(tf: TensorFrame, pos_sample_prob=0.15):
    
    edges = tf.y[:, 2:]
    batch_size = len(edges)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, train)

    edge_attr = tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    if pos_sample_prob == 0:
        return node_feats.to(device), edge_index.to(device), edge_attr.to(device) 
    # sample positive edges
    positions = torch.arange(batch_size)
    num_samples = int(len(positions) * pos_sample_prob)
    if len(positions) > 0 and num_samples > 0:
        drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), num_samples, replacement=False)
    else:
        drop_idxs = torch.tensor([]).long()
    drop_edge_ind = positions[drop_idxs]
    #drop_edge_ind = torch.tensor([x for x in range(int(batch_size/2),batch_size)])

    mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
    mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
    input_edge_index = edge_index[:, ~mask]
    input_edge_attr  = edge_attr[~mask]

    pos_edge_index = edge_index[:, mask]
    pos_edge_attr  = edge_attr[mask]

    # generate/sample negative edges
    neg_edges = []
    #ic(pos_edge_attr.feat_dict)
    neg_dict = {}
    for key, value in pos_edge_attr.feat_dict.items():
        #ic(key, value.shape)
        attr = []
        # duplicate each row of the tensor by num_neg_samples times repeated values must be contiguous
        for r in value:
            #ic(r.shape)
            if key == stype.timestamp:
                attr.append(r.repeat(num_neg_samples, 1, 1))
            else:
                attr.append(r.repeat(num_neg_samples, 1))
        neg_dict[key] = torch.cat(attr, dim=0)
    #ic(neg_dict)
    neg_edge_attr = TensorFrame(neg_dict, pos_edge_attr.col_names_dict)

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
        # pos_attr = pos_edge_attr[i]#.unsqueeze(0)  # Get the attribute of the current positive edge
        
        # replicated_attr = pos_attr.repeat(num_neg_samples, 1)  # Replicate it num_neg_samples times (for each negative edge)
        # neg_edge_attr.append(replicated_attr)
    
    input_edge_index = input_edge_index.to(device)
    input_edge_attr = input_edge_attr.to(device)
    pos_edge_index = pos_edge_index.to(device)
    pos_edge_attr = pos_edge_attr.to(device)
    node_feats = node_feats.to(device)
    if len(neg_edges) > 0:
        neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
    neg_edge_attr = neg_edge_attr.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
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
            node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf, 1)
            tf = tf.to(device)
            # ic(node_feats.shape, mask.shape, input_edge_index.shape, input_edge_attr.num_rows, pos_edge_index.shape, pos_edge_attr.num_rows, neg_edge_index.shape, neg_edge_attr.num_rows)
            # ic(tf.y[~mask].shape)
            # sys.exit()
            _, x_gnn = model(node_feats, input_edge_index, input_edge_attr)
            pos_edge_attr, _ = model.encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, model.edge_dim)
            pos_edge_attr = model.edge_emb(pos_edge_attr)
            neg_edge_attr, _ = model.encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, model.edge_dim)
            neg_edge_attr = model.edge_emb(neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            x_tab, _ = model(node_feats, edge_index, edge_attr)
            num_pred, cat_pred = mcm_decoder(x_tab[:, 0, :])

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            #link_loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            #t_loss.backward()
            moco_loss = mocoloss.loss([link_loss, t_loss])
            optimizer.step()

            loss = link_loss.item() + t_loss.item()
            loss_accum += (loss * len(pos_pred))
            total_count += len(pos_pred)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(pos_pred)
            #t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_lp=f'{loss_lp_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}')
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_lp=f'{loss_lp_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}', moco_loss=f'{moco_loss[0]:.4f},{moco_loss[1]:.4f}')
            wandb.log({"train_loss": loss_accum/total_count, "train_loss_lp": loss_lp_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n})
    return {'loss': loss_accum / total_count}

@torch.no_grad()
def eval_mcm(loader: DataLoader, model, mcm_decoder, dataset_name) -> float:
    model.eval()
    total_count = 0
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    t_n = t_c = 0
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            node_feats, edge_index, edge_attr = lp_inputs(tf, pos_sample_prob=0)
            tf.y = tf.y.to(device)
            x_tab, _ = model(node_feats, edge_index, edge_attr, len(tf.y))
            num_pred, cat_pred = mcm_decoder(x_tab[:, 0, :])
            _, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1] 
            loss_c_accum += loss_c[0]
            loss_n_accum += loss_n[0]
            total_count += len(num_pred)
            for i, ans in enumerate(tf.y):
                if ans[1] > (num_numerical-1):
                    accum_acc += (cat_pred[int(ans[1])-num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - num_pred[i][int(ans[1])]) #rmse
            t.set_postfix(
                accuracy=f'{accum_acc / t_c:.4f}',
                rmse=f'{torch.sqrt(accum_l2 / t_n):.4f}', 
                loss_mcm=f'{(loss_c_accum/t_c) + (loss_n_accum/t_n):.4f}',
                loss_c = f'{loss_c_accum/t_c:.4f}', 
                loss_n = f'{loss_n_accum/t_n:.4f}',
            )
            wandb.log({
                f"{dataset_name}_loss_mcm": (loss_c_accum/t_c) + (loss_n_accum/t_n),
                f"{dataset_name}_loss_c": loss_c_accum/t_c,
                f"{dataset_name}_loss_n": loss_n_accum/t_n,
            })
        accuracy = accum_acc / t_c
        rmse = torch.sqrt(accum_l2 / t_n)
        wandb.log({
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_rmse": rmse,
        })
        return {"accuracy": accuracy, "rmse": rmse}

@torch.no_grad()
def eval_lp(loader: DataLoader, model, lp_decoder, dataset_name) -> float:
    model.eval()
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
            node_feats, _, _, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr = lp_inputs(tf, pos_sample_prob=1)
            _, x_gnn = model(node_feats, input_edge_index, input_edge_attr, 0)
            pos_edge_attr, _ = model.encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, model.edge_dim)
            pos_edge_attr = model.edge_emb(pos_edge_attr)
            neg_edge_attr, _ = model.encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, model.edge_dim)
            neg_edge_attr = model.edge_emb(neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
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

mocoloss = MoCoLoss(model, 2, device, beta=0.999, beta_sigma=0.1, gamma=0.999, gamma_sigma=0.1, rho=0.05)
#mocoloss = MoCoLoss(model, 3, device, beta=0.999, beta_sigma=0.1, gamma=0.999, gamma_sigma=0.1, rho=0.05)
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

from src.nn.decoder import MCMHead
from src.nn.gnn.decoder import LinkPredHead
from torch_frame.data.stats import StatType
num_categorical = [len(dataset.col_stats[col][StatType.COUNT][0]) for col in dataset.tensor_frame.col_names_dict[stype.categorical]] if stype.categorical in dataset.tensor_frame.col_names_dict else 0
mcm_decoder = MCMHead(channels, num_numerical, num_categorical).to(device)
lp_decoder = LinkPredHead(n_hidden=channels, dropout=dropout).to(device)

# train_mcm = eval_mcm(train_loader, model, mcm_decoder, "tr")
# train_lp = eval_lp(train_loader, model, lp_decoder, "tr")

# val_mcm = eval_mcm(val_loader, model, mcm_decoder, "val")
# val_lp = eval_lp(val_loader, model, lp_decoder, "val")

# test_mcm = eval_mcm(test_loader, model, mcm_decoder, "test")
# test_lp = eval_lp(test_loader, model, lp_decoder, "test")
# ic(
#     train_mcm,
#     train_lp,
#     val_mcm,
#     val_lp,
#     test_mcm,
#     test_lp
# )

torch.autograd.set_detect_anomaly(False)
for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, optimizer, scheduler)
    #train_loss = train(epoch, model, [optimizer], scheduler)
    #train_loss = train(epoch, model, [optimizer_lp, optimizer_mcm], scheduler)
    #train_metric = test(train_loader, model, "tr")
    val_mcm = eval_mcm(val_loader, model, mcm_decoder, "val")
    val_lp = eval_lp(val_loader, model, lp_decoder, "val")
    test_mcm = eval_mcm(test_loader, model, mcm_decoder, "test")
    test_lp = eval_lp(test_loader, model, lp_decoder, "test")
    ic(
        train_loss,
        val_mcm,
        val_lp,
        test_mcm,
        test_lp
    )
# Create a directory to save models
#save_dir = '/scratch/takyildiz/.cache/saved_models'
#run_id = wandb.run.id
#os.makedirs(save_dir, exist_ok=True)
#model_save_path = os.path.join(save_dir, f'latest_model_run_{run_id}.pth')
#
## Save the model after each epoch, replacing the old model
#torch.save(model.state_dict(), model_save_path)
#ic(f'Model saved to {model_save_path}')
# %%
wandb.finish()


