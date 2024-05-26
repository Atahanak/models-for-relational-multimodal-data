import torch
import torch.nn.functional as F
import os
import argparse
import wandb
import math
import time
from typing import Any
from torch.nn import Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import MeanSquaredError
from tqdm import tqdm
import torch_frame
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    MultiCategoricalEmbeddingEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stype_encoder import TimestampEncoder
from torch_frame.typing import TaskType
from src import AmazonFashionDataset, TextToEmbedding, TextToEmbeddingFinetune
from icecream import ic
from peft import LoraConfig, TaskType as peftTaskType



def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    ########### Wandb Setup ############
    wandb.login()
    run = wandb.init(
        mode="disabled" if args.testing else "online",
        project=f"rel-mm", 
        name=args.name,
        config=args
    )
    wandb.save(args.script_path)
    ####################################

    ########### Define Text Encoder ############
    # if model == "distilbert-base-uncased":
    #         target_modules = ["ffn.lin1"]
    # elif model == "sentence-transformers/all-distilroberta-v1":
    #     target_modules = ["intermediate.dense"]
    # else:
    #     target_modules = "all-linear"
        
    if args.finetune:
        peft_config = LoraConfig(
            task_type=peftTaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            inference_mode=False,
            lora_dropout=args.lora_dropout,
            bias="none",
            # target_modules=target_modules,
        )
        text_encoder = TextToEmbeddingFinetune(model=args.text_model, peft_config=peft_config)
        text_stype = torch_frame.text_tokenized
        col_to_text_tokenizer_cfg = TextTokenizerConfig(text_tokenizer=text_encoder.tokenize,
                                batch_size=args.batch_size_tokenizer)
        kwargs = {
            "col_to_text_tokenizer_cfg": col_to_text_tokenizer_cfg
        }
    else:
        text_encoder = TextToEmbedding(model=args.text_model, device=device)
        text_stype = torch_frame.text_embedded
        col_to_text_embedder_cfg = TextEmbedderConfig(text_embedder=text_encoder, batch_size=args.batch_size_embedder)
        kwargs = {
            "col_to_text_embedder_cfg": col_to_text_embedder_cfg
        }
    ############################################

    ########### Load Dataset ############
    dataset = AmazonFashionDataset(
        root=args.root, 
        nrows=args.nrows,
        text_stype=text_stype,
        **kwargs)
    
    # batch_size = 512
    # if args.finetune:
    #     batch_size = model_batch_size[args.text_model]
    #     col_stypes = list(dataset.col_to_stype.values())
    #     n_tokenized = len([
    #         col_stype for col_stype in col_stypes
    #         if col_stype == torch_frame.stype.text_tokenized
    #     ])
    #     batch_size //= n_tokenized
    
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                            shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
    test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)
    wandb.log({
        "train_loader size": len(train_loader), 
        "val_loader size": len(val_loader), 
        "test_loader size": len(test_loader)
    })
    ####################################

    ########### Define Model, Loss, Metric, Optimizer ############
    # dataset.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fun = MSELoss()
    metric_computer = MeanSquaredError(squared=False).to(device)
    higher_is_better = False

    model = FTTransformer(
        channels=args.channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=get_stype_encoder_dict(
                    text_stype, text_encoder, train_tensor_frame, args),
    ).to(device)
    
    model.reset_parameters()  
    model = torch.compile(model, dynamic=True) if args.compile else model
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.numel() for p in model.parameters())
    ic(learnable_params, model_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma_rate)
    ##############################################################

    main_torch(higher_is_better, args, model, train_loader,
                   val_loader, test_loader, lr_scheduler, optimizer, dataset.task_type, loss_fun, metric_computer, device)



def main_torch(
    higher_is_better: bool,
    args: argparse.Namespace,
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    lr_scheduler: Any,
    optimizer: Any,
    task_type: TaskType,
    loss_fun: Any,
    metric_computer: Any,
    device: torch.device,
):
    start_time = time.time()
    if higher_is_better:
        best_val_metric = 0
    else:
        best_val_metric = math.inf

    for epoch in range(1, args.epochs + 1):
        metrics = train(model, train_loader, optimizer, epoch, loss_fun, device, task_type)
        val_metric = test(model, val_loader, metric_computer, device, task_type)
        metrics.update({"val_metric": val_metric})
        metrics.update({"epoch": epoch})
        wandb.log(metrics)
       
        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer, device, task_type)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer, device, task_type)
        lr_scheduler.step()
        print(f'Train Loss: {metrics["train_loss"]:.4f}, Val Metric: {val_metric:.4f}')

    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'total_time': end_time - start_time,
    }
    print(result_dict)


def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_fun: Any,
    device: torch.device,
    task_type: TaskType,
) -> float:
    model.train()
    loss_accum = total_count = 0

    # Lists to store time measurements
    data_load_times = []
    device_transfer_times = []
    loss_computation_times = []
    backward_times = []
    forward_times = []
    batch_times = []

    # Start initial time before the loop begins
    end_time = time.time()

    for tf in tqdm(loader, desc=f"Epoch: {epoch}"):
        # Calculate data load time from the end of the previous iteration
        start_time = time.time()
        data_load_time = start_time - end_time

        # Time to transfer data to device
        start_transfer_time = time.time()
        tf = tf.to(device)
        y = tf.y
        device_transfer_time = time.time() - start_transfer_time

        # Forward pass and loss computation\
        start_forward_time = time.time()
        pred = model(tf)
        forward_time = time.time() - start_forward_time

        if pred.size(1) == 1:
            pred = pred.view(-1, )

        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
        elif task_type == TaskType.REGRESSION:
            y = y.to(torch.float)
        
        start_loss_time = time.time()
        loss = loss_fun(pred, y)
        loss_computation_time = time.time() - start_loss_time

        # Backward pass
        start_backward_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - start_backward_time

        # Record times
        data_load_times.append(data_load_time)
        device_transfer_times.append(device_transfer_time)
        loss_computation_times.append(loss_computation_time)
        backward_times.append(backward_time)
        forward_times.append(forward_time)

        # Accumulate loss for average calculation
        loss_accum += float(loss.item()) * len(tf.y)
        total_count += len(tf.y)

        # Update end_time for the next iteration's load time measurement
        end_time = time.time()
        batch_times.append(end_time - start_time)

    metrics = {
        "data_load_time": sum(data_load_times),
        "device_transfer_time": sum(device_transfer_times),
        "loss_computation_time": sum(loss_computation_times),
        "backward_time": sum(backward_times),
        "forward_time": sum(forward_times),
        "batch_time": sum(batch_times),
        "train_loss": loss_accum / total_count
    }
    return metrics

@torch.no_grad()
def test(
    model: Module,
    loader: DataLoader,
    metric_computer: Any,
    device: torch.device,
    task_type: TaskType,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()

def get_stype_encoder_dict(
    text_stype: torch_frame.stype,
    text_encoder: Any,
    train_tensor_frame: torch_frame.TensorFrame,
    args: argparse.Namespace,
) -> dict[torch_frame.stype, StypeEncoder]:
    if not args.finetune:
        text_stype_encoder = LinearEmbeddingEncoder()
    else:
        model_cfg = ModelConfig(
            model=text_encoder,
            out_channels=768)#model_out_channels[args.text_model])
        col_to_model_cfg = {
            col_name: model_cfg
            for col_name in train_tensor_frame.col_names_dict[
                torch_frame.text_tokenized]
        }
        text_stype_encoder = LinearModelEncoder(
            col_to_model_cfg=col_to_model_cfg)

    stype_encoder_dict = {
        torch_frame.categorical: EmbeddingEncoder(),
        torch_frame.numerical: LinearEncoder(),
        # If text_stype is text_embedded,
        # it becomes embedding after materialization
        text_stype.parent: text_stype_encoder,
        torch_frame.multicategorical: MultiCategoricalEmbeddingEncoder(),
        torch_frame.timestamp: TimestampEncoder()
    }
    return stype_encoder_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_tokenizer", type=int, default=10000)
    parser.add_argument("--batch_size_embedder", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--nrows", type=int, default=100)
    parser.add_argument("--name", type=str, default="fashion")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--task_type", type=str,
        choices=[
            "binary_classification", "multiclass_classification", "regression"
        ],
        default="multiclass_classification",)    
    parser.add_argument("--gamma_rate", type=float, default=0.9)
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--root", type=str)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--script_path", type=str, default="")

    return parser.parse_args()

if __name__ == "__main__":
    main()