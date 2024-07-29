# %%
import os
import numpy as np
import random

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from src.datasets.util.mask import PretrainType
from torch_frame.data import DataLoader
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.data.stats import StatType

from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML
from src.nn.models import TABGNN
from src.nn.decoder import MCMHead
from src.nn.gnn.decoder import LinkPredHead
from src.utils.loss import SSLoss
from src.utils.metric import SSMetric
from src.nn.weighting.MoCo import MoCoLoss

from tqdm.auto import tqdm

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

# Create a logger
logger = logging.getLogger(__name__)
import sys
import time

torch.set_num_threads(4)

seed = 42
batch_size = 200
lr = 2e-4
eps = 1e-8
weight_decay = 1e-3
epochs = 50

compile = False
data_split = [0.6, 0.2, 0.2]
split_type = 'temporal'

khop_neighbors = [100, 100]
pos_sample_prob = 1
num_neg_samples = 64
channels = 128
num_layers = 3
dropout = 0.5

pretrain = {PretrainType.MASK, PretrainType.LINK_PRED}
#pretrain = {PretrainType.LINK_PRED}

testing = True

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
dataset = IBMTransactionsAML(
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv',# if not testing else '/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv',# if not testing else '/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    #root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Medium_Trans-c.csv',# if not testing else '/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    #root='/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv' if not testing else '/scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
    pretrain=pretrain,
    mask_type="replace",
    split_type=split_type, 
    splits=data_split, 
    khop_neighbors=khop_neighbors
)
dataset.materialize()
train_dataset, val_dataset, test_dataset = dataset.split()

# %%
batch_size = 200
tensor_frame = dataset.tensor_frame
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=batch_size, shuffle=False, num_workers=4)
num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])

num_columns = num_numerical + num_categorical + 1
logger.info(f"num_numerical: {num_numerical}")
logger.info(f"num_categorical: {num_categorical}")
logger.info(f"num_columns: {num_columns}")

# %%
def train_mcm(dataset, loader, epoc: int, encoder, model, mcm_decoder, optimizer, scheduler) -> float:
    model.train()
    loss_accum = total_count = 0
    loss_accum = loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for tf in loader:
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = mcm_inputs(tf, dataset)
            tf = tf.to(device)
            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x[target_edge_index.T].reshape(-1, 2 * channels)#.relu()
            x_target = torch.cat((x_target, target_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]

            optimizer.zero_grad()
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            t_loss.backward()
            optimizer.step()

            loss_accum += (t_loss.item() * len(tf.y))
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
    return {'loss': loss_accum / total_count}

def train_lp(dataset, loader, epoc: int, encoder, model, lp_decoder, optimizer, scheduler) -> float:
    encoder.train()
    model.train()
    lp_decoder.train()
    total_count = 0
    loss_lp_accum = 0

    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset)
            tf = tf.to(device)
            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            neg_edge_attr = target_edge_attr[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            link_loss.backward()
            optimizer.step()

            total_count += len(tf.y)
            loss_lp_accum += link_loss.item() * len(tf.y)
            t.set_postfix(loss_lp=f'{loss_lp_accum/total_count:.4f}')
            # wandb.log({"train_loss_lp": loss_lp_accum/total_count})
            #wandb.log({"lr": scheduler.get_last_lr()[0]})
    return {'loss': loss_lp_accum / total_count} 

@torch.no_grad()
def eval_mcm(dataset, loader: DataLoader, encoder, model, mcm_decoder, dataset_name) -> float:
    encoder.eval()
    model.eval()
    mcm_decoder.eval()
    total_count = 0
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    t_n = t_c = 0
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = mcm_inputs(tf, dataset)
            tf = tf.to(device)
            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x[target_edge_index.T].reshape(-1, 2 * channels)#.relu()
            x_target = torch.cat((x_target, target_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]
            _, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1] 
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
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
def eval_lp(dataset, loader: DataLoader, encoder, model, lp_decoder, dataset_name) -> float:
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
            node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset)
            tf = tf.to(device)
            input_edge_attr, _ = encoder(input_edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, edge_attr, target_edge_attr = model(node_feats, input_edge_index, input_edge_attr, target_edge_attr)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            neg_edge_attr = target_edge_attr[batch_size:,:]

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



@torch.no_grad()
def eval(dataset, loader, encoder, model, lp_decoder, mcm_decoder, dataset_name):
    encoder.eval()
    model.eval()
    lp_decoder.eval()
    mcm_decoder.eval()
    mrrs = []
    hits1 = []
    hits2 = []
    hits5 = []
    hits10 = []
    loss_accum = 0
    total_count = 0
    loss_accum = loss_lp_accum = total_count = 0
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = t_c = t_n = 0
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset)
            input_edge_attr, _ = encoder(input_edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, edge_attr, target_edge_attr = model(node_feats, input_edge_index, input_edge_attr, target_edge_attr)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            neg_edge_attr = target_edge_attr[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            x_target = x_gnn[pos_edge_index.T].reshape(-1, 2 * channels)#.relu()
            x_target = torch.cat((x_target, pos_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred] 

            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            moco_loss = mocoloss.loss([link_loss, t_loss])

            loss_accum += ((link_loss.item()*moco_loss[0]+(t_loss.item()*moco_loss[1])) * len(tf.y))
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(tf.y)
            mrr_score, hits = ssmetric.mrr(pos_pred, neg_pred, [1,2,5,10], num_neg_samples)
            mrrs.append(mrr_score)
            hits1.append(hits['hits@1'])
            hits2.append(hits['hits@2'])
            hits5.append(hits['hits@5'])
            hits10.append(hits['hits@10'])

            for i, ans in enumerate(tf.y):
                if ans[1] > (num_numerical-1):
                    accum_acc += (cat_pred[int(ans[1])-num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - num_pred[i][int(ans[1])]) #rmse

            t.set_postfix(
                mrr=f'{np.mean(mrrs):.4f}',
                hits1=f'{np.mean(hits1):.4f}',
                hits2=f'{np.mean(hits2):.4f}',
                hits5=f'{np.mean(hits5):.4f}',
                hits10=f'{np.mean(hits10):.4f}',
                loss_lp = f'{loss_lp_accum/total_count:.4f}',
                accuracy=f'{accum_acc / t_c:.4f}',
                rmse=f'{torch.sqrt(accum_l2 / t_n):.4f}', 
                loss_mcm=f'{(loss_c_accum/t_c) + (loss_n_accum/t_n):.4f}',
                loss_c = f'{loss_c_accum/t_c:.4f}', 
                loss_n = f'{loss_n_accum/t_n:.4f}'
            )
        mrr_score = np.mean(mrrs)
        hits1 = np.mean(hits1)
        hits2 = np.mean(hits2)
        hits5 = np.mean(hits5)
        hits10 = np.mean(hits10)
        accuracy = accum_acc / t_c
        rmse = torch.sqrt(accum_l2 / t_n)
        wandb.log({
            f"{dataset_name}_loss_mcm": (loss_c_accum/t_c) + (loss_n_accum/t_n),
            f"{dataset_name}_loss_c": loss_c_accum/t_c,
            f"{dataset_name}_loss_n": loss_n_accum/t_n,
            f"{dataset_name}_loss_lp": loss_lp_accum/total_count,
            f"{dataset_name}_mrr": mrr_score,
            f"{dataset_name}_hits@1": hits1,
            f"{dataset_name}_hits@2": hits2,
            f"{dataset_name}_hits@5": hits5,
            f"{dataset_name}_hits@10": hits10,
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_rmse": rmse,
        })
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10}, {"accuracy": accuracy, "rmse": rmse}

num_categorical = [len(dataset.col_stats[col][StatType.COUNT][0]) for col in dataset.tensor_frame.col_names_dict[stype.categorical]] if stype.categorical in dataset.tensor_frame.col_names_dict else 0
mcm_decoder = MCMHead(channels, num_numerical, num_categorical, w=3).to(device)
lp_decoder = LinkPredHead(n_classes=1, n_hidden=channels, dropout=dropout).to(device)

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

edge_index = dataset.train_graph.edge_index
num_nodes = dataset.train_graph.num_nodes
in_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
max_in_degree = int(in_degrees.max())
in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
model = TABGNN(
    encoder=encoder,
    channels=channels,
    edge_dim=channels*dataset.tensor_frame.num_cols,
    num_layers=num_layers, 
    dropout=dropout,
    deg=in_degree_histogram
)
model = torch.compile(model, dynamic=True) if compile else model
model.to(device)

model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
lp_decoder_params = sum(p.numel() for p in lp_decoder.parameters() if p.requires_grad)
mcm_decoder_params = sum(p.numel() for p in mcm_decoder.parameters() if p.requires_grad)
logger.info(f"model_params: {model_params}")
logger.info(f"encoder_params: {encoder_params}")
logger.info(f"lp_decoder_params: {lp_decoder_params}")
logger.info(f"mcm_decoder_params: {mcm_decoder_params}")
learnable_params = model_params + encoder_params + lp_decoder_params + mcm_decoder_params
logger.info(f"learnable_params: {learnable_params}")
# wandb.log({"learnable_params": learnable_params})

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # {'params': [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    # {'params': [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in mcm_decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in mcm_decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in lp_decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in lp_decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/3, max_lr=3*lr, step_size_up=2000, cycle_momentum=False)
ssloss = SSLoss(device, num_numerical)
ssmetric = SSMetric(device)
mocoloss = MoCoLoss(model, 2, device, beta=0.999, beta_sigma=0.1, gamma=0.999, gamma_sigma=0.1, rho=0.05)

save_dir = '/mnt/data/.cache/saved_models'
#save_dir = '/scratch/takyildiz/.cache/saved_models'
# run_id = wandb.run.id
os.makedirs(save_dir, exist_ok=True)
best_lp = 0
best_acc = 0
best_rmse = 2

# %%
# def mcm_inputs(tf: TensorFrame, dataset):
#     batch_size = len(tf.y)
#     edges = tf.y[:,-3:]
#     khop_source, khop_destination, idx = dataset.sample_neighbors(edges, 'train')

#     edge_attr = dataset.tensor_frame.__getitem__(idx)

#     nodes = torch.unique(torch.cat([khop_source, khop_destination]))
#     num_nodes = nodes.shape[0]
#     node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

#     n_id_map = {value.item(): index for index, value in enumerate(nodes)}
#     local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
#     local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
#     edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

#     drop_edge_ind = torch.tensor([x for x in range(batch_size)])
#     mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
#     mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
#     # input_edge_index = edge_index[:, ~mask]
#     # input_edge_attr  = edge_attr[~mask]
#     input_edge_index = edge_index
#     input_edge_attr  = edge_attr
#     target_edge_index = edge_index[:, mask]
#     target_edge_attr  = edge_attr[mask]
#     return node_feats.to(device), input_edge_index.to(device), input_edge_attr.to(device), target_edge_index.to(device), target_edge_attr.to(device)  

# def lp_inputs(tf: TensorFrame, dataset):
#     edges = tf.y[:,-3:]
#     batch_size = len(edges)
#     start = time.time()
#     khop_source, khop_destination, idx = dataset.sample_neighbors(edges, 'train')
#     # logger.info(f"sample_neighbors time: {time.time()-start}")
    
#     edge_attr = dataset.tensor_frame.__getitem__(idx)

#     nodes = torch.unique(torch.cat([khop_source, khop_destination]))
#     num_nodes = nodes.shape[0]
#     node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

#     start = time.time()
#     import numpy as np

#     n_id_map = {value.item(): index for index, value in enumerate(nodes)}
#     vectorized_map = np.vectorize(lambda x: n_id_map[x])

#     khop_combined = torch.cat((khop_source, khop_destination))
#     local_khop_combined = torch.LongTensor(vectorized_map(khop_combined.numpy()))

#     local_khop_source, local_khop_destination = local_khop_combined.split(khop_source.size(0))
#     edge_index = torch.stack((local_khop_source, local_khop_destination))
#     # n_id_map = {value.item(): index for index, value in enumerate(nodes)}
#     # local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
#     # local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
#     # edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))
#     #edge_index = torch.stack((local_khop_source, local_khop_destination))
#     # logger.info(f"new_edge_index time: {time.time()-start}")
#     start = time.time()

#     # drop_edge_ind = torch.tensor([x for x in range(int(batch_size))])
#     # mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
#     # mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
#     input_edge_index = edge_index[:, batch_size:]
#     input_edge_attr  = edge_attr[batch_size:]

#     pos_edge_index = edge_index[:, :batch_size]
#     pos_edge_attr  = edge_attr[:batch_size]
#     #logger.info(f"sample_neighbors time: {time.time()-start}")

#     #start = time.time()
#     # generate/sample negative edges
#     # neg_edges = []
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
#     # logger.info(f"target_edge_attr time: {time.time()-start}")

#     start = time.time()
#     # nodeset = set(range(edge_index.max()+1))
#     # # logger.info(f"edge_index: {edge_index}")
#     # # logger.info(f"pos_edge_index: {pos_edge_index}")
#     # # # Write to a file
#     # # import json
#     # # with open('edge_index.json', 'w') as f:
#     # #     json.dump(edge_index.tolist(), f)
#     # # with open('pos_edge_index.json', 'w') as f:
#     # #     json.dump(pos_edge_index.tolist(), f)
#     # # sys.exit()
#     # for i, edge in enumerate(pos_edge_index.t()):
#     #     src, dst = edge[0], edge[1]

#     #     # # # Chose negative examples in a smart way
#     #     # unavail_mask = (edge_index == src).any(dim=0) | (edge_index == dst).any(dim=0)
#     #     # unavail_nodes = torch.unique(edge_index[:, unavail_mask])
#     #     # unavail_nodes = set(unavail_nodes.tolist())
#     #     # avail_nodes = nodeset - unavail_nodes
#     #     # avail_nodes = torch.tensor(list(avail_nodes))
#     #     # # Finally, emmulate np.random.choice() to chose randomly amongst available nodes
#     #     # indices = torch.randperm(len(avail_nodes))[:num_neg_samples]
#     #     # neg_nodes = avail_nodes[indices]
        
#     #     # # Create a mask of unavailable nodes
#     #     # unavail_mask = torch.isin(edge_index.flatten(), torch.tensor([src, dst])).view(edge_index.shape).any(dim=0)
        
#     #     # # Get unique unavailable nodes
#     #     # unavail_nodes = torch.unique(edge_index[:, unavail_mask])
        
#     #     # # Convert to set for fast set operations
#     #     # unavail_nodes_set = set(unavail_nodes.tolist())
        
#     #     # # Determine available nodes by set difference
#     #     # avail_nodes = list(nodeset - unavail_nodes_set)
        
#     #     # # Convert available nodes back to tensor
#     #     # avail_nodes = torch.tensor(avail_nodes, dtype=torch.long)
        
#     #     # # Randomly select negative samples from available nodes
#     #     # indices = torch.randperm(len(avail_nodes))[:num_neg_samples]
#     #     # neg_nodes = avail_nodes[indices]

#     #     # Create a mask of unavailable nodes
#     #     unavail_mask = torch.isin(edge_index.flatten(), torch.tensor([src, dst]))
#     #     unavail_nodes = edge_index.flatten()[unavail_mask].unique()

#     #     # Create a mask for all nodes
#     #     all_nodes = torch.arange(max(nodeset) + 1)
#     #     avail_mask = ~torch.isin(all_nodes, unavail_nodes)

#     #     # Get available nodes
#     #     avail_nodes = all_nodes[avail_mask]

#     #     # Randomly select negative samples from available nodes
#     #     neg_nodes = avail_nodes[torch.randint(high=len(avail_nodes), size=(num_neg_samples,))]

#     #     # Generate num_neg_samples/2 negative edges with the same source but different destinations
#     #     num_neg_samples_half = int(num_neg_samples/2)
#     #     neg_dsts = neg_nodes[:num_neg_samples_half]  # Selecting num_neg_samples/2 random destination nodes for the source
#     #     neg_edges_src = torch.stack([src.repeat(num_neg_samples_half), neg_dsts], dim=0)
        
#     #     # Generate num_neg_samples/2 negative edges with the same destination but different sources
#     #     neg_srcs = neg_nodes[num_neg_samples_half:]  # Selecting num_neg_samples/2 random source nodes for the destination
#     #     neg_edges_dst = torch.stack([neg_srcs, dst.repeat(num_neg_samples_half)], dim=0)

#     #     # Add these negative edges to the list
#     #     neg_edges.append(neg_edges_src)
#     #     neg_edges.append(neg_edges_dst)
    
#     input_edge_index = input_edge_index.to(device)
#     input_edge_attr = input_edge_attr.to(device)
#     #pos_edge_index = pos_edge_index.to(device)
#     #pos_edge_attr = pos_edge_attr.to(device)
#     node_feats = node_feats.to(device)
    
#     # if len(neg_edges) > 0:
#     #     #neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
#     #     neg_edge_index = torch.cat(neg_edges, dim=1)

#     neg_edge_index = negative_sampling.generate_negative_samples(edge_index.tolist(), pos_edge_index.tolist(), num_neg_samples)
#     neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)

#     target_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).to(device)
#     target_edge_attr = target_edge_attr.to(device)
#     #logger.info(f"negative edges: {time.time()-start}")
#     #sys.exit()
#     return node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr


from src.utils.batch_processing import mcm_inputs, lp_inputs
# %%
def train(dataset, loader, epoc: int, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler):
    encoder.train()
    model.train()
    lp_decoder.train()
    mcm_decoder.train()
    loss_accum = total_count = 0
    loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    count = 0
    ave = 0
    l_time = 0
    f_time = 0
    b_time = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for tf in loader:
            with record_function("pre-processing"):
                batch_size = len(tf.y)
                node_feats, edge_index, edge_attr, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset)
                #tf.y = tf.y.to(device)
                node_feats = node_feats.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                input_edge_index = input_edge_index.to(device)
                target_edge_index = target_edge_index.to(device)
                target_edge_attr = target_edge_attr.to(device)

            with record_function("forward"):
                edge_attr, _ = encoder(edge_attr)
                input_edge_attr = edge_attr[batch_size:,:,:]
                target_edge_attr, _ = encoder(target_edge_attr)
                x_gnn, _, target_edge_attr_lp = model(node_feats, input_edge_index, input_edge_attr, target_edge_attr)
                pos_edge_index = target_edge_index[:, :batch_size]
                neg_edge_index = target_edge_index[:, batch_size:]
                pos_edge_attr = target_edge_attr_lp[:batch_size,:]
                neg_edge_attr = target_edge_attr_lp[batch_size:,:]
                pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

                x_gnn, _, target_edge_attr_mcm = model(node_feats, edge_index, edge_attr, target_edge_attr)
                x_target = x_gnn[pos_edge_index.T].reshape(-1, 2 * channels)#.relu()
                pos_edge_attr = target_edge_attr_mcm[:batch_size,:]
                x_target = torch.cat((x_target, pos_edge_attr), 1)
                num_pred, cat_pred = mcm_decoder(x_target)
                num_pred = num_pred.cpu()
                cat_pred = [x.cpu() for x in cat_pred]
            
            with record_function("backward"):
                optimizer.zero_grad()
                link_loss = ssloss.lp_loss(pos_pred, neg_pred)
                t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
                loss = link_loss + t_loss
                loss.backward()
                #moco_loss = mocoloss.loss([link_loss, t_loss])
                optimizer.step()
            # scheduler.step()

            #loss_accum += ((link_loss.item()*moco_loss[0]+(t_loss.item()*moco_loss[1])) * len(tf.y))
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(tf.y)
            if count == 10:
                break
            count += 1
            prof.step()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return {'loss': loss_accum / total_count} 

# %%
logger.info(f"Benchmarking TABGNN...")
loss = train(dataset, train_loader, 0, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler)

import sys
sys.exit()

# %%
from src.nn.models import TABGNNFused
model = TABGNNFused(
    encoder=encoder,
    channels=channels,
    edge_dim=channels*dataset.tensor_frame.num_cols,
    num_layers=num_layers, 
    dropout=dropout,
    deg=in_degree_histogram
)
model.to(device)
def train_mcm(dataset, loader, epoc: int, encoder, model, mcm_decoder, optimizer, scheduler) -> float:
    model.train()
    loss_accum = total_count = 0
    loss_accum = loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    count = 0
    ave = 0
    l_time = 0
    f_time = 0
    b_time = 0
    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            start = time.time()
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = mcm_inputs(tf, dataset, 'train')
            node_feats = node_feats.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)
            l_time = l_time + time.time()-start
            start = time.time()
            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr)
            x_target = x[target_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
            x_target = torch.cat((x_target, target_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]
            f_time = f_time + time.time()-start
            start = time.time()

            optimizer.zero_grad()
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            t_loss.backward()
            optimizer.step()

            loss_accum += (t_loss.item() * len(tf.y))
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            b_time = b_time + time.time()-start
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}')
            # wandb.log({"train_loss": loss_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n})
            if count == 100:
                logger.info(f"inputs time: {l_time/count}")
                logger.info(f"forward time: {f_time/count}")
                logger.info(f"backward time: {b_time/count}")
                break
            count += 1
    return {'loss': loss_accum / total_count}

def train_lp(dataset, loader, epoc: int, encoder, model, lp_decoder, optimizer, scheduler) -> float:
    encoder.train()
    model.train()
    lp_decoder.train()
    total_count = 0
    loss_lp_accum = 0
    count = 0
    ave = 0
    l_time = 0
    f_time = 0
    b_time = 0
    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            start = time.time()
            batch_size = len(tf.y)
            node_feats, _, _, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], 'train')
            node_feats = node_feats.to(device)
            neigh_edge_index = neigh_edge_index.to(device)
            neigh_edge_attr = neigh_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)
            l_time = l_time + time.time()-start
            start = time.time()
            neigh_edge_attr, _ = encoder(neigh_edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, _, target_edge_attr = model(node_feats, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr, True)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            neg_edge_attr = target_edge_attr[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            f_time = f_time + time.time()-start
            start = time.time()
            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            link_loss.backward()
            optimizer.step()

            total_count += len(tf.y)
            loss_lp_accum += link_loss.item() * len(tf.y)

            b_time = b_time + time.time()-start
            t.set_postfix(loss_lp=f'{loss_lp_accum/total_count:.4f}')
            # wandb.log({"train_loss_lp": loss_lp_accum/total_count})
            #wandb.log({"lr": scheduler.get_last_lr()[0]})
            if count == 100:
                logger.info(f"inputs time: {l_time/count}")
                logger.info(f"forward time: {f_time/count}")
                logger.info(f"backward time: {b_time/count}")
                break
            count += 1
    return {'loss': loss_lp_accum / total_count} 
def trainf(dataset, loader, epoc: int, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler, moo):
    encoder.train()
    model.train()
    lp_decoder.train()
    mcm_decoder.train()
    ave = 0
    l_time = 0
    f_time = 0
    b_time = 0
    count = 0
    if moo == "moco":
        mocoloss = MoCoLoss(model, 2, device, beta=0.999, beta_sigma=0.1, gamma=0.999, gamma_sigma=0.1, rho=0.05)
    loss_accum = total_count = 0
    loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0
    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            start = time.time()
            batch_size = len(tf.y)
            node_feats, edge_index, edge_attr, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], 'train')
            node_feats = node_feats.to(device)
            neigh_edge_index = neigh_edge_index.to(device)
            neigh_edge_attr = neigh_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            l_time = l_time + time.time()-start
            
            start = time.time()
            edge_attr, _ = encoder(edge_attr)
            neigh_edge_attr = edge_attr[batch_size:,:,:]
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, _, target_edge_attr_lp = model(node_feats, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr, True)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr_lp[:batch_size,:]
            neg_edge_attr = target_edge_attr_lp[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            x_gnn, _, target_edge_attr_mcm = model(node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr)
            x_target = x_gnn[pos_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
            pos_edge_attr = target_edge_attr_mcm[:batch_size,:]
            x_target = torch.cat((x_target, pos_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            f_time = f_time + time.time()-start
            # print(f"forward time: {time.time()-start}")

            start = time.time()
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            if moo == "moco":
                moco_loss = mocoloss.loss([link_loss, t_loss])
                loss_accum += ((link_loss.item()*moco_loss[0]+(t_loss.item()*moco_loss[1])) * len(tf.y))
            else:
                loss = link_loss + t_loss
                loss.backward()
                loss_accum += ((link_loss.item()+(t_loss.item())) * len(tf.y))
            optimizer.step()
            # scheduler.step()

            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(tf.y)
            b_time = b_time + time.time()-start
            # print(f"backward time: {time.time()-start}")
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_lp=f'{loss_lp_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}')
            # wandb.log({"epoch": epoc, "train_loss": loss_accum/total_count, "train_loss_lp": loss_lp_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n})
            #wandb.log({"lr": scheduler.get_last_lr()[0]})
            if count == 100:
                logger.info(f"inputs time: {l_time/count}")
                logger.info(f"forward time: {f_time/count}")
                logger.info(f"backward time: {b_time/count}")
                break
            count += 1
    return {'loss': loss_accum / total_count} 

# %%
# loss = train_mcm(dataset, train_loader, 0, encoder, model, mcm_decoder, optimizer, scheduler)
logger.info(f"Benchmarking FUSED...")
loss = trainf(dataset, train_loader, 0, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler, 'sum')
logger.info(f"Benchmarking FUSED mcm...")
loss = train_mcm(dataset, train_loader, 0, encoder, model, mcm_decoder, optimizer, scheduler)
logger.info(f"Benchmarking FUSED lp...")
loss = train_lp(dataset, train_loader, 0, encoder, model, lp_decoder, optimizer, scheduler)


