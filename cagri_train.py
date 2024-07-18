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

from src.datasets import IBMTransactionsAML
from sklearn.metrics import f1_score
from cagri_utils import *

parser = create_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define a model config dictionary and wandb logging at the same time
wandb.init(
    mode="disabled" if args.testing else "online",
    project="rel-mm-supervised", #replace this with your wandb project name if you want to use wandb logging

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
        "lr": 5e-4,
        "n_hidden": 64,
        "n_gnn_layers": 2,
        "n_classes" : 2,
        "loss": "ce",
        "w_ce1": 1.,
        "w_ce2": 6.,
        "dropout": 0.10527690625126304,
    }
)

config = wandb.config
create_experiment_path(config) # type: ignore
# Create a logger
logger_setup()

dataset = IBMTransactionsAML(
    root=config.data,
    split_type='temporal', 
    splits=[0.6, 0.2, 0.2], 
    khop_neighbors=args.num_neighs
)
dataset.materialize()
train_dataset, val_dataset, test_dataset = dataset.split()


tensor_frame = dataset.tensor_frame 
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset.tensor_frame, batch_size=config.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=config.batch_size, shuffle=False, num_workers=4)

num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])

num_columns = num_numerical + num_categorical + 1
config.num_columns = num_columns
logging.info(f"num_numerical: {num_numerical}")
logging.info(f"num_categorical: {num_categorical}")
logging.info(f"num_columns: {num_columns}")


stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.timestamp: TimestampEncoder(),
    }


model = SupervisedTabGNN(
            dataset.col_stats, 
            dataset.tensor_frame.col_names_dict, 
            stype_encoder_dict, 
            config).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)



best_val_f1 = 0
for epoch in range(config.epochs):
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    for batch in train_loader:

        optimizer.zero_grad()
        batch_size = len(batch.y)

        node_feats, edge_index, edge_attr, y = graph_inputs(dataset, batch, tensor_frame, mode='train', args=args)
        node_feats, edge_index, edge_attr, y = node_feats.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

        pred = model(node_feats, edge_index, edge_attr)[:batch_size]

        preds.append(pred.argmax(dim=-1))
        ground_truths.append(y)
        loss = loss_fn(pred, y.to(torch.long))
        
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

    #evaluate
    val_f1 = evaluate(val_loader, dataset, tensor_frame, model, device, args, 'val')
    te_f1 = evaluate(test_loader, dataset, tensor_frame, model, device, args, 'test')

    wandb.log({"f1/validation": val_f1}, step=epoch)
    wandb.log({"f1/test": te_f1}, step=epoch)
    logging.info(f'Validation F1: {val_f1:.4f}')
    logging.info(f'Test F1: {te_f1:.4f}')

    if epoch == 0:
        wandb.log({"best_test_f1": te_f1}, step=epoch)
    elif val_f1 > best_val_f1:
        best_val_f1 = val_f1
        wandb.log({"best_test_f1": te_f1}, step=epoch)
        if args.save_model:
            save_model(model, optimizer, epoch, config)





