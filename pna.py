import os
import re
import time
from typing import Optional, Tuple, Set

import numpy as np
import torch

from src.datasets.util.mask import PretrainType
from torch_frame.data import DataLoader
from torch_frame import stype
from torch_frame.data.stats import StatType

from torch_geometric.utils import degree

from src.datasets import IBMTransactionsAML
from src.datasets import EthereumPhishingTransactions
from src.nn.gnn import PNA
from src.nn.decoder import MCMHead
from src.nn.gnn.decoder import LinkPredHead
from src.utils.loss import SSLoss
from src.utils.metric import SSMetric
from src.nn.weighting.MoCo import MoCoLoss
from src.utils.batch_processing import mcm_inputs, lp_inputs


from tqdm.auto import tqdm
import wandb

import logging
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

import fire

# workaround for CUDA invalid configuration bug
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_numerical = num_categorical = num_columns = 0
ssloss = ssmetric = None
args = None

def train_mcm(dataset, loader, epoch: int, encoder, model, mcm_decoder, optimizer, scheduler) -> float:
    model.train()
    loss_accum = total_count = 0
    loss_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 1e-12

    with tqdm(loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = mcm_inputs(tf, dataset, 'train', args["ego"])
            node_feats = node_feats.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)
            
            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x[target_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
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
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}')
        wandb.log({"train_loss": loss_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n}, step=epoch)
    return {'loss': loss_accum / total_count}

def train_lp(dataset, loader, epoch: int, encoder, model, lp_decoder, optimizer, scheduler) -> float:
    encoder.train()
    model.train()
    lp_decoder.train()
    total_count = 0
    loss_lp_accum = 0

    with tqdm(loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, _, _, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], "train", args["ego"])
            node_feats = node_feats.to(device)
            input_edge_index = input_edge_index.to(device)
            input_edge_attr = input_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]

            pos_edge_attr = target_edge_attr[:batch_size,:]
            pos_edge_attr, _ = encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, num_columns * args["channels"]) 

            neg_edge_attr = target_edge_attr[batch_size:,:]
            neg_edge_attr, _ = encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, num_columns * args["channels"]) 

            input_edge_attr, _ = encoder(input_edge_attr)
            input_edge_attr = input_edge_attr.view(-1, num_columns * args["channels"]) 

            x, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            link_loss.backward()
            optimizer.step()

            total_count += len(tf.y)
            loss_lp_accum += link_loss.item() * len(tf.y)
            t.set_postfix(loss_lp=f'{loss_lp_accum/total_count:.4f}')
        wandb.log({"train_loss_lp": loss_lp_accum/total_count}, step=epoch)
    return {'loss': loss_lp_accum / total_count} 

@torch.no_grad()
def eval_mcm(epoch, dataset, loader: DataLoader, encoder, model, mcm_decoder, dataset_name) -> float:
    encoder.eval()
    model.eval()
    mcm_decoder.eval()
    total_count = 0
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = total_count = t_c = t_n = 1e-12
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr = mcm_inputs(tf, dataset, dataset_name, args["ego"])
            node_feats = node_feats.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            edge_attr, _ = encoder(edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x, edge_attr, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x[target_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
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
        accuracy = accum_acc / t_c
        rmse = torch.sqrt(accum_l2 / t_n)
        wandb.log({
            f"{dataset_name}_loss_mcm": (loss_c_accum/t_c) + (loss_n_accum/t_n),
            f"{dataset_name}_loss_c": loss_c_accum/t_c,
            f"{dataset_name}_loss_n": loss_n_accum/t_n,
            f"{dataset_name}_accuracy": accuracy,
            f"{dataset_name}_rmse": rmse,
        }, step=epoch)
        return {"accuracy": accuracy, "rmse": rmse}

@torch.no_grad()
def eval_lp(epoch, dataset, loader: DataLoader, encoder, model, lp_decoder, dataset_name) -> float:
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
            node_feats, _, _, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], dataset_name, args["ego"])
            node_feats = node_feats.to(device)
            input_edge_index = input_edge_index.to(device)
            input_edge_attr = input_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = target_edge_attr[:batch_size,:]
            pos_edge_attr, _ = encoder(pos_edge_attr)
            pos_edge_attr = pos_edge_attr.view(-1, num_columns * args["channels"]) 

            neg_edge_attr = target_edge_attr[batch_size:,:]
            neg_edge_attr, _ = encoder(neg_edge_attr)
            neg_edge_attr = neg_edge_attr.view(-1, num_columns * args["channels"]) 

            input_edge_attr, _ = encoder(input_edge_attr)
            input_edge_attr = input_edge_attr.view(-1, num_columns * args["channels"])

            x, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            pos_pred, neg_pred = lp_decoder(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
            loss = ssloss.lp_loss(pos_pred, neg_pred)
            
            loss_lp_accum += loss * len(pos_pred)
            loss_accum += float(loss) * len(pos_pred)
            total_count += len(pos_pred)
            mrr_score, hits = ssmetric.mrr(pos_pred, neg_pred, [1,2,5,10], args["num_neg_samples"])
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
        mrr_score = np.mean(mrrs)
        hits1 = np.mean(hits1)
        hits2 = np.mean(hits2)
        hits5 = np.mean(hits5)
        hits10 = np.mean(hits10)
        wandb.log({
            f"{dataset_name}_loss_lp": loss_lp_accum/total_count,
            f"{dataset_name}_mrr": mrr_score,
            f"{dataset_name}_hits@1": hits1,
            f"{dataset_name}_hits@2": hits2,
            f"{dataset_name}_hits@5": hits5,
            f"{dataset_name}_hits@10": hits10,
        }, step=epoch)
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10}

def train(dataset, loader, epoch: int, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler, moo):
    encoder.train()
    model.train()
    lp_decoder.train()
    mcm_decoder.train()
    if moo == "moco":
        mocoloss = MoCoLoss(model, 2, device, beta=0.999, beta_sigma=0.1, gamma=0.999, gamma_sigma=0.1, rho=0.05)
    loss_accum = total_count = 0
    loss_lp_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 1e-12
    with tqdm(loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, edge_index, edge_attr, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], 'train', args["ego"])
            node_feats = node_feats.to(device)
            neigh_edge_index = neigh_edge_index.to(device)
            neigh_edge_attr = neigh_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            if "2f" in moo:
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                edge_attr, _ = encoder(edge_attr)
                neigh_edge_attr = edge_attr[batch_size:,:,:]
            else:
                neigh_edge_attr, _ = encoder(neigh_edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, _, t_attr = model(node_feats, neigh_edge_index, neigh_edge_attr, target_edge_attr)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = t_attr[:batch_size,:]
            neg_edge_attr = t_attr[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            if "2f" in moo:
                x_gnn, _, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x_gnn[pos_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
            x_target = torch.cat((x_target, pos_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)

            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]

            optimizer.zero_grad()
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
        
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            if "moco" in moo:
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
            t.set_postfix(loss=f'{loss_accum/total_count:.4f}', loss_lp=f'{loss_lp_accum/total_count:.4f}', loss_c=f'{loss_c_accum/t_c:.4f}', loss_n=f'{loss_n_accum/t_n:.4f}')
        wandb.log({"train_loss": loss_accum/total_count, "train_loss_lp": loss_lp_accum/total_count, "train_loss_c": loss_c_accum/t_c, "train_loss_n": loss_n_accum/t_n}, step=epoch)
    return {'loss': loss_accum / total_count} 

@torch.no_grad()
def eval(epoch, dataset, loader, encoder, model, lp_decoder, mcm_decoder, dataset_name, moo):
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
    loss_c_accum = loss_n_accum = t_c = t_n = 1e-12
    with tqdm(loader, desc=f'Evaluating') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, edge_index, edge_attr, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset, args["num_neg_samples"], dataset_name, args["ego"])
            node_feats = node_feats.to(device)
            neigh_edge_index = neigh_edge_index.to(device)
            neigh_edge_attr = neigh_edge_attr.to(device)
            target_edge_index = target_edge_index.to(device)
            target_edge_attr = target_edge_attr.to(device)

            if "2f" in moo:
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                edge_attr, _ = encoder(edge_attr)
                neigh_edge_attr = edge_attr[batch_size:,:,:]
            else:
                neigh_edge_attr, _ = encoder(neigh_edge_attr)
            target_edge_attr, _ = encoder(target_edge_attr)
            x_gnn, _, t_edge_attr = model(node_feats, neigh_edge_index, neigh_edge_attr, target_edge_attr)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            pos_edge_attr = t_edge_attr[:batch_size,:]
            neg_edge_attr = t_edge_attr[batch_size:,:]
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            if "2f" in moo:
                edge_index.to(device)
                edge_attr.to(device)
                x_gnn, _, target_edge_attr = model(node_feats, edge_index, edge_attr, target_edge_attr)
            x_target = x_gnn[pos_edge_index.T].reshape(-1, 2 * args["channels"])#.relu()
            x_target = torch.cat((x_target, pos_edge_attr), 1)
            num_pred, cat_pred = mcm_decoder(x_target)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred] 
            link_loss = ssloss.lp_loss(pos_pred, neg_pred)
            t_loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            # moco_loss = mocoloss.loss([link_loss, t_loss])

            #loss_accum += ((link_loss.item()*moco_loss[0]+(t_loss.item()*moco_loss[1])) * len(tf.y))
            loss_accum += ((link_loss.item()+(t_loss.item())) * len(tf.y))
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            loss_lp_accum += link_loss.item() * len(tf.y)
            mrr_score, hits = ssmetric.mrr(pos_pred, neg_pred, [1,2,5,10], args["num_neg_samples"])
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
        }, step=epoch)
        return {"mrr": mrr_score, "hits@1": hits1, "hits@2": hits2, "hits@5": hits5, "hits@10": hits10}, {"accuracy": accuracy, "rmse": rmse}

def parse_checkpoint(checkpoint: str) -> list[str, int]:
    """
    Parse the checkpoint file to extract the run identifier and the epoch number.

    Args:
        checkpoint (str): Path to the checkpoint file.

    Returns:
        Tuple[str, int]: A tuple containing the run identifier and the epoch number.

    Raises:
        ValueError: If the checkpoint file does not exist or has an invalid format.
    """

    if not os.path.isfile(checkpoint):
        raise ValueError('Checkpoint file does not exist')

    pattern = r"^run_(?P<run_id>[a-zA-Z0-9]+)_epoch_(?P<epoch>\d+)\.pth$"
    logger.info(f'checkpoint: {os.path.basename(checkpoint)}')
    match = re.match(pattern, os.path.basename(checkpoint))
    if match:
        run_id = match.group("run_id")
        epoch = match.group("epoch")
        print(f'Continuing run_{run_id} using checkpoint file: {checkpoint} from epoch {epoch}')
        return run_id, int(epoch)
    else:
        raise ValueError('Checkpoint file has invalid format')

def init_wandb(args: dict, run_name: str, wandb_dir: str, run_id: Optional[str], group: Optional[str]) -> object:
    """
    Initialize the Weights & Biases run for tracking and logging.

    Args:
        args (dict): Dictionary of parameters and their values for this run.
        run_name (str): Name of the run.
        wandb_dir (str): Directory to store Weights & Biases data.
        run_id (Optional[str]): Existing run identifier for resuming, if any.

    Returns:
        object: An initialized wandb run object.
    """
    wandb.login()
    run = wandb.init(
        entity="cse3000",
        dir=wandb_dir,
        mode="disabled" if args['testing'] else "online",
        project="iclr",
        name=run_name,
        config=args,
        id=run_id if run_id is not None else None,    
        resume="must" if run_id is not None else None,
        group=group if group is not None else None,
    )
    wandb.log({"device": str(device)}, step=0)
    return run

def parse_pretrain_args(pretrain) -> Set[PretrainType]:
    """
    Parse pretraining arguments into a set of pretraining types.

    Args:
        pretrain (list): List of pretrain argument strings.

    Returns:
        Set[PretrainType]: Set of pretraining types deduced from the arguments.
    """
    pretrain_dict = {
        "mask": PretrainType.MASK,
        "mv": PretrainType.MASK_VECTOR,
        "lp": PretrainType.LINK_PRED,
    }

    pretrain_set = set()
    for pretrain_type in pretrain:
        pretrain_set.add(pretrain_dict[pretrain_type])

    return pretrain_set

def get_dataset(dataset_path: str, pretrain: Set[PretrainType], split_type, data_split, khop_neighbors):
    if "ibm" in dataset_path:
        dataset = IBMTransactionsAML(
            root=dataset_path, 
            pretrain=pretrain,
            split_type=split_type,
            splits=data_split, 
            khop_neighbors=khop_neighbors,
            ports=args["ports"]
        )
    elif "eth" in dataset_path:
       dataset = EthereumPhishingTransactions(
            root=dataset_path, 
            pretrain=pretrain,
            split_type=split_type,
            splits=data_split, 
            khop_neighbors=khop_neighbors,
            ports=args["ports"]
        ) 
    logger.info(f"Materializing dataset...")
    s = time.time()
    dataset.materialize()
    logger.info(f"Materialized in {time.time() - s:.2f} seconds")
    dataset.df.head(5)
    global num_numerical, num_categorical, num_columns
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in dataset.tensor_frame.col_names_dict else 0
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in dataset.tensor_frame.col_names_dict else 0 
    num_t = len(dataset.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in dataset.tensor_frame.col_names_dict else 0
    num_columns = num_numerical + num_categorical + num_t
    return dataset

def get_data_loaders(dataset: IBMTransactionsAML, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Setup data loaders for the training, validation, and test sets.

    Args:
        dataset (IBMTransactionsAML): The dataset to split into loaders.
        batch_size (int): The batch size for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for the training, validation, and test datasets.
    """
    train_dataset, val_dataset, test_dataset = dataset.split()
    train_loader = DataLoader(train_dataset.tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset.tensor_frame, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset.tensor_frame, batch_size=batch_size, shuffle=False)
    logger.info(f"train_loader size: {len(train_loader)}")
    logger.info(f"val_loader size: {len(val_loader)}")
    logger.info(f"test_loader size: {len(test_loader)}")
    wandb.log({
        "train_loader size": len(train_loader),
        "val_loader size": len(val_loader),
        "test_loader size": len(test_loader)
    },step=0)
    return train_loader, val_loader, test_loader

def get_model(dataset: IBMTransactionsAML, encoder, channels: int, num_layers: int, compile: bool, checkpoint: Optional[str], dropout: float) -> torch.nn.Module:
    """
    Initialize the machine learning model with the specified architecture and settings,
    optionally loading weights from a checkpoint.

    Args:
        dataset (IBMTransactionsAML): The dataset from which model configurations are derived.
        device (torch.device): The device (CPU or GPU) to run the model on.
        channels (int): The number of channels (or features) in the model's layers.
        num_layers (int): The number of layers in the model.
        pretrain_set (Set[PretrainType]): Set of pretraining types to apply.
        is_compile (bool): Whether to compile the model using TorchScript.
        checkpoint (Optional[str]): Path to the checkpoint file from which to load model weights.

    Returns:
        torch.nn.Module: The initialized (and possibly compiled) PyTorch model.
    """
    edge_index = dataset.train_graph.edge_index
    num_nodes = dataset.train_graph.num_nodes
    in_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    max_in_degree = int(in_degrees.max())
    in_degree_histogram = torch.zeros(max_in_degree + 1, dtype=torch.long)
    in_degree_histogram += torch.bincount(in_degrees, minlength=in_degree_histogram.numel())
    model = PNA(
        node_dim=1+int(args["ego"]),
        num_features=1, 
        num_gnn_layers=num_layers, 
        edge_dim=dataset.tensor_frame.num_cols*channels, 
        n_classes=1, 
        deg=in_degree_histogram,
        edge_updates=True,
        encoder=encoder,
        reverse_mp=args["reverse_mp"],
    )
    model.to(device)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model

def get_optimizer(encoder: torch.nn.Module, model: torch.nn.Module, decoders: list[torch.nn.Module], lr: float, eps: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Set up the optimizer for the model training, using AdamW with specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is to be set up.
        lr (float): Learning rate for the optimizer.
        eps (float): Epsilon parameter for the optimizer to improve numerical stability.
        weight_decay (float): Weight decay (L2 penalty) to apply.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [ #encoder is included in model
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    learnable_params = model_params + encoder_params
    for name, decoder in decoders:
        decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        optimizer_grouped_parameters.append({'params': [p for n, p in decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay})
        optimizer_grouped_parameters.append({'params': [p for n, p in decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}) 

        logger.info(f"{name}_params: {decoder_params}")
        learnable_params += decoder_params
    logger.info(f"encoder_params: {encoder_params}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"learnable_params: {learnable_params}")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    wandb.log({"learnable_params": learnable_params}, step=0)
    return optimizer

def main(checkpoint="", dataset="/path/to/your/file", run_name="/your/run/name", save_dir="/path/to/save/",
         seed=42, batch_size=200, channels=128, num_layers=3, lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=10,
         data_split=[0.6, 0.2, 0.2], dropout=0.5, split_type="temporal_daily", pretrain=["mask", "lp"], khop_neighbors=[100, 100], num_neg_samples=64,
         compile=False, testing=True, wandb_dir="/path/to/wandb", group="", mode="lp", moo="sum",
         ego=False, ports=False, reverse_mp=False):
    if mode != "lp":
        ValueError("Only link prediction is supported for now")
    global args
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
        'channels': channels,
        'split_type': split_type,
        'num_neg_samples': num_neg_samples,
        'pretrain': pretrain,
        'khop_neighbors': khop_neighbors,
        'num_layers': num_layers,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'mode': mode,
        'moo': moo,
        'ego': ego,
        'ports': ports,
        'reverse_mp': reverse_mp
    }
    logger.info(f"args: {args}")

    if checkpoint == "":
        checkpoint = None
        run_id, checkpoint_epoch = None, None
    else:
        run_id, checkpoint_epoch = parse_checkpoint(checkpoint)

    init_wandb(args, run_name, wandb_dir, run_id, group)
    pretrain_set = parse_pretrain_args(pretrain)
    dataset = get_dataset(dataset, pretrain_set, split_type, data_split, khop_neighbors)
    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size)

    encoder = dataset.get_encoder(channels)
    model = get_model(dataset, encoder, channels, num_layers, compile, checkpoint, dropout)
    num_categorical = [len(dataset.col_stats[col][StatType.COUNT][0]) for col in dataset.tensor_frame.col_names_dict[stype.categorical]] if stype.categorical in dataset.tensor_frame.col_names_dict else []
    mcm_decoder = MCMHead(channels, num_numerical, num_categorical, w=3).to(device)
    lp_decoder = LinkPredHead(n_classes=1, n_hidden=channels, dropout=dropout).to(device)

    if mode == "mcm-lp":
        decoders = [("mcm", mcm_decoder), ("lp", lp_decoder)]
    elif mode == "mcm":
        decoders = [("mcm", mcm_decoder)]
    elif mode == "lp":
        decoders = [("lp", lp_decoder)]
    
    # load decoders from checkpoint
    if checkpoint is not None:
        for name, decoder in decoders:
            decoder_checkpoint = checkpoint.replace("run", name)
            decoder.load_state_dict(torch.load(decoder_checkpoint, map_location=device))
            logger.info(f"{name} loaded from {decoder_checkpoint}")

    optimizer = get_optimizer(encoder, model, decoders, lr, eps, weight_decay)
    scheduler = None
    global ssloss, ssmetric
    ssloss = SSLoss(device, num_numerical)
    ssmetric = SSMetric(device)

    run_id = wandb.run.id
    logger.info(f"run_id: {run_id}")
    logger.info(f"run name {wandb.run.name}")
    logger.info(f"run url {wandb.run.url}")
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_epoch is not None:
        start_epoch = checkpoint_epoch + 1
        end_epoch = checkpoint_epoch + epochs + 1
    else:
        start_epoch = 1
        end_epoch = epochs + 1

    best_lp = 0
    best_acc = 0
    best_rmse = 2

    for epoch in range(start_epoch, end_epoch):
        logger.info(f"Epoch {epoch}:")
        if mode == "mcm-lp":
            loss = train(dataset, train_loader, epoch, encoder, model, lp_decoder, mcm_decoder, optimizer, scheduler, moo)
            logger.info(f"loss: {loss}")
            val_lp, val_mcm = eval(epoch, dataset, val_loader, encoder, model, lp_decoder, mcm_decoder, "val", moo)
            logger.info(f"val_mcm: {val_mcm}")
            logger.info(f"val_lp: {val_lp}")
            test_lp, test_mcm = eval(epoch, dataset, test_loader, encoder, model, lp_decoder, mcm_decoder, "test", moo)
            logger.info(f"test_mcm: {test_mcm}")
            logger.info(f"test_lp: {test_lp}")
        elif mode == "mcm":
            mcm_loss = train_mcm(dataset, train_loader, epoch, encoder, model, mcm_decoder, optimizer, scheduler)
            logger.info(f"loss_mcm: {mcm_loss}")
            val_mcm = eval_mcm(epoch, dataset, val_loader, encoder, model, mcm_decoder, "val")
            logger.info(f"val_mcm: {val_mcm}")
            test_mcm = eval_mcm(epoch, dataset, test_loader, encoder, model, mcm_decoder, "test")
            logger.info(f"test_mcm: {test_mcm}")
        elif mode == "lp":
            lp_loss = train_lp(dataset, train_loader, epoch, encoder, model, lp_decoder, optimizer, scheduler)
            logger.info(f"loss_lp: {lp_loss}")
            val_lp = eval_lp(epoch, dataset, val_loader, encoder, model, lp_decoder, "val")
            logger.info(f"val_lp: {val_lp}")
            test_lp = eval_lp(epoch, dataset, test_loader, encoder, model, lp_decoder, "test")
            logger.info(f"test_lp: {test_lp}")

        if mode == "mcm-lp" or mode == "mcm":            
            if best_acc < val_mcm['accuracy'] and not testing:
                model_save_path = os.path.join(save_dir, f'{run_id}_acc.pth')
                best_acc = val_mcm['accuracy']
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Best ACC model saved to {model_save_path}')
            if best_rmse > val_mcm['rmse'] and not testing:
                model_save_path = os.path.join(save_dir, f'{run_id}_rmse.pth')
                best_rmse = val_mcm['rmse']
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Best RMSE model saved to {model_save_path}')

        if mode == "mcm-lp" or mode == "lp":    
            if val_lp['mrr'] > best_lp and not testing:
                model_save_path = os.path.join(save_dir, f'{run_id}_mrr.pth')
                best_lp = val_lp['mrr']
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Best MRR model saved to {model_save_path}')
        
        if not testing:
            model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Checkpoint saved to {model_save_path}')
            # save decoders
            for name, decoder in decoders:
                decoder_save_path = os.path.join(save_dir, f'{name}_{run_id}_epoch_{epoch}.pth')
                torch.save(decoder.state_dict(), decoder_save_path)
                logger.info(f'{name} saved to {decoder_save_path}')
            if epoch > 1:
                rpath = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch-1}.pth')
                os.remove(rpath)
                logger.info(f'Previous checkpoint removed: {rpath}')
                for name, decoder in decoders:
                    rpath = os.path.join(save_dir, f'{name}_{run_id}_epoch_{epoch-1}.pth')
                    os.remove(rpath)
                    logger.info(f'Previous {name} removed: {rpath}')

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)