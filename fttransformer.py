"""
This module contains implementations for a self_supervised learning framework using
PyTorch, WandB, and other utilities to train, evaluate, and manage experiments of machine
learning models.
"""
import os
import re
import time
from typing import Optional, Tuple, Set

import fire
import torch
import torch.nn.functional as F
from tqdm import tqdm
from icecream import ic
import wandb

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.data.stats import StatType

from src.datasets import IBMTransactionsAML, EthereumPhishingTransactions
from src.nn.models.ft_transformer import FTTransformer
from src.datasets.util.mask import PretrainType
from src.nn.decoder import SelfSupervisedHead
from src.utils.loss import SSLoss

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

torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_numerical, num_categorical, num_cat, num_columns = 0, 0, 0, 0


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


def prepare_dataset(dataset_path: str, pretrain_set: Set[PretrainType], masked_dir: str) -> IBMTransactionsAML:
    """
    Prepare the dataset for training by loading it and initializing necessary configurations.

    Args:
        dataset_path (str): Path to the dataset.
        pretrain_set (Set[PretrainType]): Set of pretraining types to apply.

    Returns:
        IBMTransactionsAML: The prepared dataset.
    """
    if "ibm" in dataset_path:
        dataset = IBMTransactionsAML(
            root=dataset_path, 
            pretrain=pretrain_set,
        )
    elif "eth" in dataset_path:
       dataset = EthereumPhishingTransactions(
            root=dataset_path,
            pretrain=pretrain_set,
        )
    s = time.time()
    dataset.materialize()
    logger.info(f"Materialized in {time.time() - s:.2f} seconds")
    global num_numerical, num_categorical, num_columns, num_cat
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical]) if stype.numerical in dataset.tensor_frame.col_names_dict else 0
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical]) if stype.categorical in dataset.tensor_frame.col_names_dict else 0 
    num_t = len(dataset.tensor_frame.col_names_dict[stype.timestamp]) if stype.timestamp in dataset.tensor_frame.col_names_dict else 0
    num_categorical = [len(dataset.col_stats[col][StatType.COUNT][0]) for col in dataset.tensor_frame.col_names_dict[stype.categorical]] if stype.categorical in dataset.tensor_frame.col_names_dict else []
    num_columns = num_numerical + num_cat
    dataset.df.head(5)
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
    logger.info(f"Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    wandb.log({
        "train_loader size": len(train_loader),
        "val_loader size": len(val_loader),
        "test_loader size": len(test_loader)
    }, step=0)
    return train_loader, val_loader, test_loader


def initialize_model(dataset: IBMTransactionsAML, device: torch.device, channels: int, num_layers: int, decoder_set: Set[PretrainType],
                    is_compile: bool, checkpoint: Optional[str]) -> torch.nn.Module:
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
    encoder = dataset.get_encoder(channels=channels)
    decoder = SelfSupervisedHead(channels, num_numerical, num_categorical)
    model = FTTransformer(
        channels=channels,
        num_layers=num_layers,
        encoder=encoder,
        decoder=decoder
    ).to(device)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    return encoder, model, decoder


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
    logger.info(f"Learnable parameters: {learnable_params}")
    wandb.log({"learnable_params": learnable_params}, step=0)
    return optimizer


def train(encoder: torch.nn.Module, model: torch.nn.Module, decoder: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
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
    loss_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 1e-12
    ssloss = SSLoss(device, num_numerical)
    with tqdm(train_loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            tf = tf.to(device)
            x, x_cls = model(tf)
            num_pred, cat_pred = decoder(x_cls)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]
            tf.y = tf.y.cpu()
            loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
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
                    "train_loss_n": loss_n_accum / t_n}, step=epoch)
    return ((loss_c_accum / t_c) * (num_cat / num_columns)) + ((loss_n_accum / t_n) * (num_numerical / num_columns))


@torch.no_grad()
def test(encoder: torch.nn.Module, model: torch.nn.Module, decoder: torch.nn.Module, test_loader: DataLoader, dataset_name: str, epoch: int) -> Tuple[float, float]:
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
    ssloss = SSLoss(device, num_numerical)
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = 0
    t_n = t_c = 1e-12
    with tqdm(test_loader, desc='Evaluating') as t:
        for tf in t:
            tf = tf.to(device)
            x, x_cls = model(tf)
            num_pred, cat_pred = decoder(x_cls)
            num_pred = num_pred.cpu()
            cat_pred = [x.cpu() for x in cat_pred]
            tf.y = tf.y.cpu()
            loss, loss_c, loss_n = ssloss.mcm_loss(cat_pred, num_pred, tf.y)
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            t_c += loss_c[1]
            t_n += loss_n[1]
            for i, ans in enumerate(tf.y):
                # ans --> [val, idx]
                # pred --> feature_type_num X type_num X batch_size
                if ans[1] > (num_numerical - 1):
                    accum_acc += (cat_pred[int(ans[1]) - num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - num_pred[i][int(ans[1])])  #rmse

            # loss numerical
            loss_c_mcm = (((loss_c_accum / t_c) * (num_cat / num_columns)) +
                          ((loss_n_accum / t_n) * (num_numerical / num_columns)))
            loss_c = loss_c_accum / t_c
            loss_n = loss_n_accum / t_n

            acc = accum_acc / t_c
            rmse = torch.sqrt(accum_l2 / t_n)
            loss = (loss_c_accum / t_c) + (loss_n_accum / t_n)
            t.set_postfix(accuracy=f'{acc:.4f}',
                          rmse=f'{rmse:.4f}',
                          loss=f'{loss:.4f}',
                          loss_c_mcm=f'{loss_c_mcm:.4f}',
                          loss_c=f'{loss_c:.4f}',
                          loss_n=f'{loss_n:.4f}')

        wandb.log({"epoch": epoch, f"{dataset_name}_accuracy": accum_acc / t_c,
                   f"{dataset_name}_rmse": rmse,
                   f"{dataset_name}_loss": loss,
                   f"{dataset_name}_loss_c_mcm": loss_c_mcm,
                   f"{dataset_name}_loss_c": loss_c_accum / t_c,
                   f"{dataset_name}_loss_n": loss_n_accum / t_n,
                   "epoch": epoch}, step=epoch)
        return [rmse, acc]


def main(checkpoint="None", dataset="/path/to/your/dataset/", run_name="fttransformer",
         seed=42, batch_size=200, channels=128, num_layers=3, lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=20,
         data_split=[0.6, 0.2, 0.2], split_type="temporal_daily", pretrain=["mask"],
         is_compile=False, testing=False, wandb_dir="/wandb/dir/", group="", masked_dir="", save_dir="/save/dir/"):
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

    if checkpoint == "None":
        checkpoint = None
        run_id, checkpoint_epoch = None, None
    else:
        run_id, checkpoint_epoch = parse_checkpoint(checkpoint)

    init_wandb(args, run_name, wandb_dir, run_id, group)

    pretrain_set = parse_pretrain_args(pretrain)
    dataset = prepare_dataset(dataset, pretrain_set, masked_dir)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset, batch_size)

    encoder, model, decoder = initialize_model(dataset, device, channels, num_layers, pretrain_set, is_compile, checkpoint)
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
        train_loss = train(encoder, model, decoder, train_loader, optimizer, epoch)
        logger.info(f"Epoch {epoch} Train Loss: {train_loss}")
        # train_metric = test(model, train_loader, "tr", epoch)
        val_metric = test(encoder, model, decoder, val_loader, "val", epoch)
        logger.info(f"Epoch {epoch} Val RMSE: {val_metric[0]}, Val Accuracy: {val_metric[1]}")
        test_metric = test(encoder, model, decoder, test_loader, "test", epoch)
        logger.info(f"Epoch {epoch} Test RMSE: {test_metric[0]}, Test Accuracy: {test_metric[1]}")
        if not testing:
            model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_save_path)
            remove_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch - 1}.pth')
            if os.path.exists(remove_path):
                os.remove(remove_path)
                logger.info(f"Removed {remove_path}")
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
