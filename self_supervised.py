"""
This module contains implementations for a self_supervised learning framework using
PyTorch, WandB, and other utilities to train, evaluate, and manage experiments of machine
learning models.
"""
import os
import re
from typing import Optional, Tuple, Set

import fire
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from torch_frame import TensorFrame
from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from src.datasets import IBMTransactionsAML
from src.nn.models.ft_transformer import FTTransformer
from src.datasets.util.mask import PretrainType

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

torch.set_float32_matmul_precision('high')
torch.set_num_threads(8)
torch.autograd.set_detect_anomaly(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_numerical, num_categorical, num_columns = 0, 0, 0

def parse_checkpoint(checkpoint: str) -> [str, int]:
    """
    Parse the checkpoint file to extract the run identifier and the epoch number.

    Args:
        checkpoint (str): Path to the checkpoint file.

    Returns:
        Tuple[str, int]: A tuple containing the run identifier and the epoch number.

    Raises:
        ValueError: If the checkpoint file does not exist or has an invalid format.
    """
    full_path = os.path.join(os.getcwd(), checkpoint)

    if not os.path.isfile(full_path):
        raise ValueError('Checkpoint file does not exist')

    pattern = r"^run_(?P<run_id>[a-zA-Z0-9]+)_epoch_(?P<epoch>\d+)\.pth$"
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
        project="rel-mm",
        name=run_name,
        config=args,
        id=run_id if run_id is not None else None,    
        resume="must" if run_id is not None else None,
        group=group if group is not None else None,
    )
    wandb.log({"device": str(device)})
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

def prepare_dataset(dataset_path: str, pretrain_set: Set[PretrainType], split_type, data_split, khop_neighbors) -> IBMTransactionsAML:
    """
    Prepare the dataset for training by loading it and initializing necessary configurations.

    Args:
        dataset_path (str): Path to the dataset.
        pretrain_set (Set[PretrainType]): Set of pretraining types to apply.

    Returns:
        IBMTransactionsAML: The prepared dataset.
    """
    dataset_path = dataset_path if "scratch" in dataset_path else os.getcwd() + dataset_path
    dataset = IBMTransactionsAML(
        root=dataset_path, 
        pretrain=pretrain_set,
        split_type=split_type,
        data_split=data_split,
        khop_neighbors=khop_neighbors,
    )
    dataset.materialize()
    global num_numerical, num_categorical, num_columns
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
    num_t = len(dataset.tensor_frame.col_names_dict[stype.timestamp])
    num_columns = num_numerical + num_categorical + num_t
    return dataset

def setup_data_loaders(dataset: IBMTransactionsAML, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    })
    return train_loader, val_loader, test_loader

def initialize_model(dataset: IBMTransactionsAML, device: torch.device, channels: int, num_layers: int,
                     pretrain_set: Set[PretrainType], is_compile: bool, checkpoint: Optional[str]) -> torch.nn.Module:
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
    model = FTTransformer(
        channels=channels,
        out_channels=None,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=dataset.tensor_frame.col_names_dict,
        stype_encoder_dict={
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearEncoder(),
            stype.timestamp: TimestampEncoder(),
        },
        pretrain=pretrain_set
    ).to(device)

    if checkpoint:
        # get epoch from checkpoint file name
        checkpoint.split(".pth")
        pattern = r"^saved_models/self-supervised/run_(?P<identifier>[a-zA-Z0-9]+)_epoch_(?P<epoch>\d+)\.pth$"
        match = re.match(pattern, checkpoint)

        if match:
            identifier = match.group("identifier")
            epoch = match.group("epoch")
            print(f'Continuing run_{identifier} using checkpoint file: {checkpoint} from epoch {epoch}')
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            raise ValueError('Checkpoint file has invalid format')
    else:
        model = torch.compile(model, dynamic=True) if is_compile else model

    return model

def setup_optimizer(encoder: torch.nn.Module, model: torch.nn.Module, decoders: list(torch.nn.Module), lr: float, eps: float, weight_decay: float) -> torch.optim.Optimizer:
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
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    # scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0, timescale=1000)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model_params: {model_params}")
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"encoder_params: {encoder_params}")
    learnable_params = model_params + encoder_params
    for name, decoder in decoders:
        decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        logger.info(f"{name}_params: {decoder_params}")
        learnable_params += decoder_params
    logger.info(f"learnable_params: {learnable_params}")
    wandb.log({"learnable_params": learnable_params})
    return optimizer

def lp_inputs(tf: TensorFrame, dataset, num_neg_samples):
    
    edges = tf.y[:,-3:]
    batch_size = len(edges)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, 'train')

    edge_attr = dataset.tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    drop_edge_ind = torch.tensor([x for x in range(int(batch_size))])
    mask = torch.zeros((edge_index.shape[1],)).long() #[E, ]
    mask = mask.index_fill_(dim=0, index=drop_edge_ind, value=1).bool() #[E, ]
    input_edge_index = edge_index[:, ~mask]
    input_edge_attr  = edge_attr[~mask]

    pos_edge_index = edge_index[:, mask]
    pos_edge_attr  = edge_attr[mask]

    # generate/sample negative edges
    neg_edges = []
    target_dict = pos_edge_attr.feat_dict
    for key, value in pos_edge_attr.feat_dict.items():
        attr = []
        # duplicate each row of the tensor by num_neg_samples times repeated values must be contiguous
        for r in value:
            if key == stype.timestamp:
                attr.append(r.repeat(num_neg_samples, 1, 1))
            else:
                attr.append(r.repeat(num_neg_samples, 1))
        target_dict[key] = torch.cat([target_dict[key], torch.cat(attr, dim=0)], dim=0)
    target_edge_attr = TensorFrame(target_dict, pos_edge_attr.col_names_dict)

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
    
    input_edge_index = input_edge_index.to(device)
    input_edge_attr = input_edge_attr.to(device)
    #pos_edge_index = pos_edge_index.to(device)
    #pos_edge_attr = pos_edge_attr.to(device)
    node_feats = node_feats.to(device)
    if len(neg_edges) > 0:
        #neg_edge_index = torch.cat(neg_edges, dim=1).to(device)
        neg_edge_index = torch.cat(neg_edges, dim=1)
    target_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).to(device)
    target_edge_attr = target_edge_attr.to(device)
    return node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr

def train_lp(dataset, loader, epoc: int, encoder, model, lp_decoder, optimizer, scheduler) -> float:
    encoder.train()
    model.train()
    lp_decoder.train()
    total_count = 0
    loss_lp_accum = 0

    with tqdm(loader, desc=f'Epoch {epoc}') as t:
        for tf in t:
            batch_size = len(tf.y)
            node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr = lp_inputs(tf, dataset)
            tf = tf.to(device)
            input_edge_attr, _ = encoder(input_edge_attr)
            # input_edge_attr = input_edge_attr.view(-1, num_columns * channels) 
            target_edge_attr, _ = encoder(target_edge_attr)
            # target_edge_attr = target_edge_attr.view(-1, num_columns * channels) 
            x_tab, x_gnn = model(node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr) #, True)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            # pos_edge_attr = x_tab[:batch_size,0,:]
            # neg_edge_attr = x_tab[batch_size:,0,:]
            pos_edge_attr = x_tab[:batch_size,:]
            neg_edge_attr = x_tab[batch_size:,:]

            # pos_edge_index = target_edge_index[:, :batch_size]
            # neg_edge_index = target_edge_index[:, batch_size:]
            # pos_edge_attr = target_edge_attr[:batch_size,:]
            # pos_edge_attr, _ = encoder(pos_edge_attr)
            # pos_edge_attr = pos_edge_attr.view(-1, num_columns * channels) 
            # neg_edge_attr = target_edge_attr[batch_size:,:]
            # neg_edge_attr, _ = encoder(neg_edge_attr)
            # neg_edge_attr = neg_edge_attr.view(-1, num_columns * channels) 
            # input_edge_attr, _ = encoder(input_edge_attr)
            # input_edge_attr = input_edge_attr.view(-1, num_columns * channels) 

            # pos_pred, neg_pred = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            #x_gnn, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) #, True)
            pos_pred, neg_pred = lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

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
def eval_lp(dataset, loader: DataLoader, encoder, model, decoder, dataset_name, num_neg_samples) -> float:
    encoder.eval()
    model.eval()
    decoder.eval()
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
            # input_edge_attr = input_edge_attr.view(-1, num_columns * channels) 
            target_edge_attr, _ = encoder(target_edge_attr)
            # target_edge_attr = target_edge_attr.view(-1, num_columns * channels) 
            x_tab, x_gnn = model(node_feats, input_edge_index, input_edge_attr, target_edge_index, target_edge_attr) #, True)
            pos_edge_index = target_edge_index[:, :batch_size]
            neg_edge_index = target_edge_index[:, batch_size:]
            # pos_edge_attr = x_tab[:batch_size,0,:]
            # neg_edge_attr = x_tab[batch_size:,0,:]
            pos_edge_attr = x_tab[:batch_size,:]
            neg_edge_attr = x_tab[batch_size:,:]

            # pos_edge_index = target_edge_index[:, :batch_size]
            # neg_edge_index = target_edge_index[:, batch_size:]
            # pos_edge_attr = target_edge_attr[:batch_size,:]
            # pos_edge_attr, _ = encoder(pos_edge_attr)
            # pos_edge_attr = pos_edge_attr.view(-1, num_columns * channels) 
            # neg_edge_attr = target_edge_attr[batch_size:,:]
            # neg_edge_attr, _ = encoder(neg_edge_attr)
            # neg_edge_attr = neg_edge_attr.view(-1, num_columns * channels) 
            # input_edge_attr, _ = encoder(input_edge_attr)
            # input_edge_attr = input_edge_attr.view(-1, num_columns * channels)

            # pos_pred, neg_pred = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)

            # x_gnn, pos_edge_attr, neg_edge_attr = model(node_feats, input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) #, True)
            pos_pred, neg_pred = decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
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

def main(checkpoint="", dataset="/path/to/your/file", run_name="/your/run/name", save_dir="/path/to/save/",
         seed=42, batch_size=200, channels=128, num_layers=3, lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=10,
         data_split=[0.6, 0.2, 0.2], split_type="temporal", pretrain=["mask"], khop_neighbors=[100, 100], num_neg_samples=64,
         is_compile=False, testing=True, wand_dir="/path/to/wandb", group=""):
    args = {
        "testing": testing,
        "seed": seed,
        "batch_size": batch_size,
        "channels": channels,
        "num_layers": num_layers,
        "pretrain": pretrain,
        "compile": is_compile,
        "lr": lr,
        "eps": eps,
        "epochs": epochs,
        "data_split": data_split,
        "split_type": split_type,
        "weight_decay": weight_decay,
        "device": device
    }
    torch.manual_seed(args['seed'])

    if checkpoint != "":
        run_id, checkpoint_epoch = parse_checkpoint(checkpoint)
    else:
        run_id, checkpoint_epoch = None, None

    init_wandb(args, run_name, wand_dir, run_id, group)
    pretrain_set = parse_pretrain_args(pretrain)
    dataset = prepare_dataset(dataset, pretrain_set, split_type, data_split, khop_neighbors)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset, batch_size)

    encoder = dataset.get_encoder(channels)
    model = initialize_model(dataset, device, channels, num_layers, pretrain_set, is_compile, checkpoint)
    optimizer = setup_optimizer(model, lr, eps, weight_decay)

    run_id = wandb.run.id
    os.makedirs(save_dir, exist_ok=True)
    best_lp = 0

    if checkpoint_epoch is not None:
        start_epoch = checkpoint_epoch + 1
        end_epoch = checkpoint_epoch + epochs + 1
    else:
        start_epoch = 1
        end_epoch = epochs + 1

    for epoch in range(start_epoch, end_epoch):
        train_loss = train_lp(dataset, train_loader, epoch, encoder, model, lp_decoder, optimizer, epoch)
        logger.info(f"Epoch {epoch} train loss: {train_loss['loss']}")
        #train_metric = eval_lp(model, train_loader, "tr", epoch)
        val_metric = eval_lp(val_loader, encoder, model, lp_decoder, "val", epoch)
        logger.info(f"Epoch {epoch} val: {val_metric}")
        test_metric = eval_lp(test_loader, encoder, model, lp_decoder, "test", epoch)
        logger.info(f"Epoch {epoch} test: {test_metric}")
        if not testing:
            model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Model saved to {model_save_path}')
            if test_metric['mrr'] > best_lp and not testing:
                model_save_path = os.path.join(save_dir, f'{run_id}_mrr.pth')
                best_lp = test_metric['mrr']
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Best MRR model saved to {model_save_path}')

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
