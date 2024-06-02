#!/usr/bin/env python
# coding: utf-8
import os

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
#

# In[6]:


import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.datasets import Yandex
from torch_frame.data import DataLoader
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    TimestampEncoder,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet
)
from icecream import ic
from tqdm import tqdm

from src.datasets import IBMTransactionsAML


seed = 42
batch_size = 512
numerical_encoder_type = 'linear'
model_type = 'fttransformer'
channels = 256
num_layers = 4

compile = True
lr = 1e-3
epochs = 10


dataset = IBMTransactionsAML(root=os.getcwd() + '/data/Over-Sampled_Tiny_Trans-c.csv')
ic(dataset)
dataset.materialize()
is_classification = dataset.task_type.is_classification
ic(is_classification)
dataset.df.head(5)


# In[7]:


torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[8]:


train_dataset, val_dataset, test_dataset = dataset.split()


# In[9]:


train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_tensor_frame, batch_size=batch_size, shuffle=False)
ic(len(train_loader), len(val_loader), len(test_loader))


# In[10]:


# print an example batch
ic(next(iter(train_loader)).feat_dict)
ic(next(iter(train_loader)).y)


# In[11]:


if numerical_encoder_type == 'linear':
    numerical_encoder = LinearEncoder()
elif numerical_encoder_type == 'linear_bucket':
    numerical_encoder = LinearBucketEncoder()
elif numerical_encoder_type == 'periodic':
    numerical_encoder = LinearPeriodicEncoder()
else:
    raise ValueError(f'Unknown numerical encoder type: {numerical_encoder_type}')

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder,
    stype.timestamp: TimestampEncoder(),
}

if is_classification:
    output_channels = dataset.num_classes
else:
    output_channels = 1


# In[12]:


if model_type == 'fttransformer':
    model = FTTransformer(
        channels=channels,
        out_channels=output_channels,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict
    ).to(device)
elif model_type == 'resnet':
    model = ResNet(
        channels=channels,
        out_channels=output_channels,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict
    ).to(device)
else:
    raise ValueError(f'Unknown model type: {model_type}')

model = torch.compile(model, dynamic=True) if compile else model
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def train(epoc: int) -> float:
    model.train()
    loss_accum = total_count = 0

    with tqdm(train_loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            if is_classification:
                loss = F.cross_entropy(pred, tf.y)
            else:
                loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}')
    return loss_accum / total_count

@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    accum = total_count = 0
    confusion_matrix = [[0 for _ in range(dataset.num_classes)] for _ in range(dataset.num_classes)]
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            total_count += len(tf.y)
            if is_classification:
                pred_class = pred.argmax(dim=-1)
                #update confusion matrix
                for r, p in zip(tf.y, pred_class):
                    confusion_matrix[r][p] += 1
                #display confusion matrix
                #t.set_postfix(confusion_matrix=confusion_matrix)
                accum += float((tf.y == pred_class).sum())
                t.set_postfix(accuracy=f'{accum/total_count:.4f}')
            else:
                accum += float(F.mse_loss(pred.view(-1), tf.y.view(-1), reduction='sum'))

        if is_classification:
            accuracy = accum / total_count
            return [confusion_matrix, accuracy]
        else:
            rmse = (accum / total_count) **0.5
            return rmse


# In[13]:


if is_classification:
    metric = 'Acc'
    best_val_metric = (None, 0)
    best_test_metric = (None, 0)
else:
    metric = 'RMSE'
    best_val_metric = float('inf')
    best_test_metric = float('inf')

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    ic(val_metric)
    test_metric = test(test_loader)
    ic(test_metric)

    if is_classification and val_metric[1] > best_val_metric[1]:
        best_val_metric = val_metric
        best_test_metric = test_metric
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    ic(train_loss, train_metric, val_metric, test_metric)

ic(best_val_metric, best_test_metric)

