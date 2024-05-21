import argparse
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from icecream import ic
from peft import LoraConfig, get_peft_model
from peft import TaskType as peftTaskType
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)
import evaluate
import torch_frame
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.data import DataLoader, MultiNestedTensor
from torch_frame.nn import (EmbeddingEncoder, FTTransformer,
                            LinearEmbeddingEncoder, LinearEncoder,
                            LinearModelEncoder,
                            MultiCategoricalEmbeddingEncoder, StypeEncoder)
from torch_frame.nn.encoder.stype_encoder import TimestampEncoder
from torch_frame.typing import TextTokenizationOutputs, TaskType
from src import Custom_Dataset
import wandb
from icecream import ic



def use_llm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########### Wandb Setup ############
    wandb.login()
    run = wandb.init(
        mode="disabled" if args.testing else "online",
        project=f"rel-mm", 
        name=args.name,
        config=args
    )
    script_path = "/home/cgriu/cse3000/slurm/separate/scripts/"+args.name+".sh"
    wandb.save(script_path)

    text_encoder = TextToEmbedding(model=args.text_model, device=device)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype": text_stype,
        "col_to_text_embedder_cfg": TextEmbedderConfig(text_embedder=text_encoder, batch_size=args.st2_batch_size_embedder),
    }
    ic(args.task_type)
    tt = TaskType(args.task_type)
    ic(tt)
    dataset = Custom_Dataset(
        root=args.root, 
        task_type=TaskType(args.task_type),
        nrows=args.nrows,
        **kwargs)
    
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_dataset_tensor = train_dataset.tensor_frame
    val_dataset_tensor = val_dataset.tensor_frame
    test_dataset_tensor = test_dataset.tensor_frame
    train_loader = DataLoader(train_dataset_tensor, batch_size=args.st2_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_tensor, batch_size=args.st2_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_tensor, batch_size=args.st2_batch_size, shuffle=False)

    out_channels = 1
    loss_fun = MSELoss()
    metric_computer = MeanSquaredError(squared=False).to(device)
    higher_is_better = False

    model = FTTransformer(
        channels=args.channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
        stype_encoder_dict=get_stype_encoder_dict(
                    text_stype, text_encoder, train_dataset.tensor_frame, args),
    ).to(device)

    model.reset_parameters()
    model = torch.compile(model, dynamic=True) if args.compile else model
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(learnable_params)
    model_size = sum(p.numel() for p in model.parameters())
    ic(model_size)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma_rate)

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

    for epoch in range(1, args.st2_epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, loss_fun, device, task_type)
        val_metric = test(model, val_loader, metric_computer, device, task_type)
        wandb.log({f"train_loss": train_loss, f"val_metric": val_metric})
        if higher_is_better:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer, device, task_type)
        else:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer, device, task_type)
        lr_scheduler.step()
        print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

    end_time = time.time()
    result_dict = {
        'args': args.__dict__,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'total_time': end_time - start_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dict, args.result_path)


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
        ic(loss, loss.dtype)
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

    # Log average of times
    wandb.log({
        "epoch": epoch,
        "data_load_time": sum(data_load_times) / len(data_load_times),
        "device_transfer_time": sum(device_transfer_times) / len(device_transfer_times),
        "loss_computation_time": sum(loss_computation_times) / len(loss_computation_times),
        "backward_time": sum(backward_times) / len(backward_times),
        "forward_time": sum(forward_times) / len(forward_times),
        "batch_time": sum(batch_times) / len(batch_times),
    })

    return loss_accum / total_count

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

class TextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        embedding_model_learnable_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad)
        ic(embedding_model_learnable_params)
    def __call__(self, sentences: list[str]) -> Tensor:
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        for key in inputs:
            if isinstance(inputs[key], Tensor):
                inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # [batch_size, max_length or batch_max_length]
        # Value is either one or zero, where zero means that
        # the token is not attended to other tokens.
        mask = inputs["attention_mask"]
        return (mean_pooling(out.last_hidden_state.detach(),
                             mask).squeeze(1).cpu())


def main():
    args = parse_args()
    use_llm(args)
    return

        

class TextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device)
        embedding_model_learnable_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad)
        ic(embedding_model_learnable_params)
    def __call__(self, sentences: list[str]) -> Tensor:
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        for key in inputs:
            if isinstance(inputs[key], Tensor):
                inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # [batch_size, max_length or batch_max_length]
        # Value is either one or zero, where zero means that
        # the token is not attended to other tokens.
        mask = inputs["attention_mask"]
        return (mean_pooling(out.last_hidden_state.detach(),
                             mask).squeeze(1).cpu())


class TextToEmbeddingFinetune(torch.nn.Module):
    r"""Include :obj:`tokenize` that converts text data to tokens, and
    :obj:`forward` function that converts tokens to embeddings with a
    text model, whose parameters will also be finetuned along with the
    tabular learning. The pooling strategy used here to derive sentence
    embedding is the mean pooling which takes mean value of all tokens'
    embeddings.

    Args:
        model (str): Model name to load by using :obj:`transformers`,
            such as :obj:`distilbert-base-uncased` and
            :obj:`sentence-transformers/all-distilroberta-v1`.
    """
    def __init__(self, model: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

        if model == "distilbert-base-uncased":
            target_modules = ["ffn.lin1"]
        elif model == "sentence-transformers/all-distilroberta-v1":
            target_modules = ["intermediate.dense"]
        else:
            target_modules = "all-linear"

        peft_config = LoraConfig(
            task_type=peftTaskType.FEATURE_EXTRACTION,
            r=32,
            lora_alpha=32,
            inference_mode=False,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        embedding_model_fineteune_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad)
        ic(embedding_model_fineteune_params)

    def forward(self, feat: dict[str, MultiNestedTensor]) -> Tensor:
        # Pad [batch_size, 1, *] into [batch_size, 1, batch_max_seq_len], then,
        # squeeze to [batch_size, batch_max_seq_len].
        input_ids = feat["input_ids"].to_dense(fill_value=0).squeeze(dim=1)
        # Set attention_mask of padding idx to be False
        mask = feat["attention_mask"].to_dense(fill_value=0).squeeze(dim=1)

        # Get text embeddings for each text tokenized column
        # `out.last_hidden_state` has the shape:
        # [batch_size, batch_max_seq_len, text_model_out_channels]
        out = self.model(input_ids=input_ids, attention_mask=mask)

        # Return value has the shape [batch_size, 1, text_model_out_channels]
        return mean_pooling(out.last_hidden_state, mask)

    def tokenize(self, sentences: list[str]) -> TextTokenizationOutputs:
        # Tokenize batches of sentences
        return self.tokenizer(sentences, truncation=True, padding=True,
                              return_tensors="pt")
    
def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)

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
            out_channels=model_out_channels[args.text_model])
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
    parser.add_argument("--lr", type=float, default=0.0001)
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
        default="regression",)    
    parser.add_argument("--pos_weight", type=bool, default=False)
    parser.add_argument("--gamma_rate", type=float, default=0.9)
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--result_path", type=str, default="/home/cgriu/cse3000/slurm/fashion/results/result.pth")
    parser.add_argument("--root", type=str, default="/scratch/cgriu/AML_dataset/AMAZON_FASHION.csv")
    parser.add_argument("--st2_epochs", type=int, default=10)
    parser.add_argument("--st2_batch_size_embedder", type=int, default=5)
    parser.add_argument("--st2_batch_size", type=int, default=256)

    return parser.parse_args()

model_out_channels = {
    "distilbert-base-uncased": 768,
    "roberta-large": 1024,
    "microsoft/deberta-v3-large": 1024,
    "google/electra-large-discriminator": 1024,
    "sentence-transformers/all-distilroberta-v1": 768,
}

# Set for a 16 GB GPU
model_batch_size = {
    "distilbert-base-uncased": 128,
    "roberta-large": 16,
    "microsoft/deberta-v3-large": 8,
    "google/electra-large-discriminator": 16,
    "sentence-transformers/all-distilroberta-v1": 128*4,
}
    

if __name__ == "__main__":
    main()