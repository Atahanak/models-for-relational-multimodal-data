"""
This module contains implementations for a self_supervised learning framework using
PyTorch, WandB, and other utilities to train, evaluate, and manage experiments of machine
learning models.
"""
import os
import re
from typing import Optional, Tuple, Set

import fire
import torch
import torch.nn.functional as F
from tqdm import tqdm
from icecream import ic
import wandb

from transformers import get_inverse_sqrt_schedule
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

torch.set_float32_matmul_precision('high')
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


def prepare_dataset(dataset_path: str, pretrain_set: Set[PretrainType], masked_dir: str) -> IBMTransactionsAML:
    """
    Prepare the dataset for training by loading it and initializing necessary configurations.

    Args:
        dataset_path (str): Path to the dataset.
        pretrain_set (Set[PretrainType]): Set of pretraining types to apply.

    Returns:
        IBMTransactionsAML: The prepared dataset.
    """
    dataset_path = dataset_path if "scratch" in dataset_path else os.getcwd() + dataset_path
    dataset = IBMTransactionsAML(root=dataset_path, pretrain=pretrain_set, masked_dir=masked_dir)
    ic(dataset)
    dataset.materialize()
    global num_numerical, num_categorical, num_columns
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
    num_columns = num_numerical + num_categorical
    dataset.df.head(5)

    ic(num_numerical, num_categorical, num_columns)
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
    ic(len(train_loader), len(val_loader), len(test_loader))
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


def setup_optimizer(model: torch.nn.Module, lr: float, eps: float, weight_decay: float) -> torch.optim.Optimizer:
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
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(learnable_params)
    wandb.log({"learnable_params": learnable_params})
    return optimizer


def train(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
    """
    Train the model for one epoch using the provided data loader and optimizer.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        epoch (int): The current epoch number.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    loss_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0

    with tqdm(train_loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            loss, loss_c, loss_n = calc_loss(pred, tf.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() * len(tf.y)
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            t.set_postfix(loss=f'{loss_accum / total_count:.4f}',
                          loss_c=f'{loss_c_accum / t_c:.4f}',
                          loss_n=f'{loss_n_accum / t_n:.4f}')
            wandb.log({"train_loss_mcm": loss_accum / total_count,
                       "train_loss_c": loss_c_accum / t_c,
                       "train_loss_n": loss_n_accum / t_n,
                       "epoch": epoch})
    return ((loss_c_accum / t_c) * (num_categorical / num_columns)) + (
            (loss_n_accum / t_n) * (num_numerical / num_columns))


@torch.no_grad()
def test(model: torch.nn.Module, test_loader: DataLoader, dataset_name: str, epoch: int) -> Tuple[float, float]:
    """
    Evaluate the model using the provided data loader and log performance metrics to wandb.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): The DataLoader providing test data.
        dataset_name (str): The name of the dataset to log as part of the metrics.
        epoch (int): The current epoch number for logging purposes.

    Returns:
        Tuple[float, float]: Tuple containing RMSE (root mean squared error) and accuracy.
    """
    model.eval()
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = 0
    t_n = t_c = 0
    with tqdm(test_loader, desc='Evaluating') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            _, loss_c, loss_n = calc_loss(pred, tf.y)
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            t_c += loss_c[1]
            t_n += loss_n[1]
            for i, ans in enumerate(tf.y):
                # ans --> [val, idx]
                # pred --> feature_type_num X type_num X batch_size
                if ans[1] > (num_numerical - 1):
                    accum_acc += (pred[1][int(ans[1]) - num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - pred[0][i][int(ans[1])])  #rmse

            # loss numerical
            loss_c_mcm = (((loss_c_accum / t_c) * (num_categorical / num_columns)) +
                          ((loss_n_accum / t_n) * (num_numerical / num_columns)))
            loss_c = loss_c_accum / t_c
            loss_n = loss_n_accum / t_n
            wandb.log({f"{dataset_name}_loss_mcm": loss_c_mcm,
                       f"{dataset_name}_loss_c": loss_c,
                       f"{dataset_name}_loss_n": loss_n,
                       "epoch": epoch})

            acc = accum_acc / t_c
            rmse = torch.sqrt(accum_l2 / t_n)
            loss = (loss_c_accum / t_c) + (loss_n_accum / t_n)
            t.set_postfix(accuracy=f'{acc:.4f}',
                          rmse=f'{rmse:.4f}',
                          loss=f'{loss:.4f}',
                          loss_c_mcm=f'{loss_c_mcm:.4f}',
                          loss_c=f'{loss_c:.4f}',
                          loss_n=f'{loss_n:.4f}')

        wandb.log({f"{dataset_name}_accuracy": accum_acc / t_c,
                   f"{dataset_name}_rmse": rmse,
                   f"{dataset_name}_loss": loss,
                   f"{dataset_name}_loss_c_mcm": loss_c_mcm,
                   f"{dataset_name}_loss_c": loss_c_accum / t_c,
                   f"{dataset_name}_loss_n": loss_n_accum / t_n,
                   "epoch": epoch})
        del tf
        del pred
        return [rmse, acc]


def calc_loss(pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, int], Tuple[float, int]]:
    """
    Calculate loss for a batch of predictions and corresponding true values.

    Args:
        pred (torch.Tensor): The predictions tensor.
        y (torch.Tensor): The true values tensor.

    Returns:
        Tuple[torch.Tensor, Tuple[float, int], Tuple[float, int]]: The computed loss, a tuple containing accumulated
        categorical loss and count, and a tuple containing accumulated numerical loss and count.
    """
    accum_n = accum_c = t_n = t_c = 0
    for i, ans in enumerate(y):
        # ans --> [val, idx]
        # pred --> feature_type_num X type_num X batch_size
        if ans[1] > (num_numerical - 1):
            t_c += 1
            a = torch.tensor(int(ans[0])).to(device)
            accum_c += F.cross_entropy(pred[1][int(ans[1]) - num_numerical][i], a)
            del a
        else:
            t_n += 1
            accum_n += torch.square(pred[0][i][int(ans[1])] - ans[0])  #mse
    return (accum_n / t_n) + torch.sqrt(accum_c / t_c), (accum_c, t_c), (accum_n, t_n)


def main(checkpoint="", dataset="/data/Over-Sampled_Tiny_Trans-c.csv", run_name="self-supervised",
         seed=42, batch_size=200, channels=128, num_layers=3, lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=10,
         data_split=[0.6, 0.2, 0.2], split_type="temporal", pretrain=["mask"],
         is_compile=False, testing=False, wand_dir="/mnt/data/", group="testing", masked_dir="/tmp/.cache/masked_columns", save_dir="saved_models/self-supervised"):
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
    dataset = prepare_dataset(dataset, pretrain_set, masked_dir)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset, batch_size)

    model = initialize_model(dataset, device, channels, num_layers, pretrain_set, is_compile, checkpoint)
    optimizer = setup_optimizer(model, lr, eps, weight_decay)

    run_id = wandb.run.id
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_epoch is not None:
        start_epoch = checkpoint_epoch + 1
        end_epoch = checkpoint_epoch + epochs + 1
    else:
        start_epoch = 1
        end_epoch = epochs + 1

    for epoch in range(start_epoch, end_epoch):
        train_loss = train(model, train_loader, optimizer, epoch)
        train_metric = test(model, train_loader, "tr", epoch)
        val_metric = test(model, val_loader, "val", epoch)
        test_metric = test(model, test_loader, "test", epoch)
        ic(
            train_loss,
            train_metric,
            val_metric,
            test_metric
        )
        model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
