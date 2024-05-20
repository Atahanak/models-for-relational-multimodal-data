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
from torch_frame.typing import TextTokenizationOutputs, TaskType as TTT
from src import Custom_Dataset
import wandb




def use_llm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = TextToEmbedding(model=args.text_model, device=device)
    text_stype = torch_frame.text_embedded
    kwargs = {
        "text_stype": text_stype,
        "col_to_text_embedder_cfg": TextEmbedderConfig(text_embedder=text_encoder, batch_size=args.batch_size_embedder),
    }
    dataset = Custom_Dataset(
        root=args.root, 
        task_type=TTT(args.task_type),
        nrows=args.nrows,
        **kwargs)
    
    dataset.materialize()
    train_dataset, val_dataset, test_dataset = dataset.split()

    out_channels = 1
    loss_fun = MSELoss()
    metric_computer = MeanSquaredError(squared=False).to(device)

    # Define model
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


    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        eval_steps=5,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=3,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.st2_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        fp16=True,
    )

    # Define compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.tensor(predictions).to(device)
        labels = torch.tensor(labels).to(device)
        metric_computer.update(predictions, labels)
        result = metric_computer.compute()
        metric_computer.reset()
        return {"mse": result.item()}

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=None,  # Adjust if necessary
        data_collator=None,  # Adjust if necessary
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics,
    )

    # Train and evaluate the model
    trainer.train()
    trainer.evaluate(eval_dataset=val_dataset)

def main():
    args = parse_args()
    # finetune_llm(args)
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
    parser.add_argument("--pos_weight", type=bool, default=False)
    parser.add_argument("--gamma_rate", type=float, default=0.9)
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--result_path", type=str, default="/home/cgriu/cse3000/slurm/fashion/results/result.pth")
    parser.add_argument("--root", type=str, default="/scratch/cgriu/AML_dataset/AMAZON_FASHION.csv")
    parser.add_argument("--st1_epochs", type=int, default=10)
    parser.add_argument("--st1_lora_alpha", type=int, default=1)
    parser.add_argument("--st1_lora_dropout", type=float, default=0.1)
    parser.add_argument("--st1_r", type=int, default=8)
    parser.add_argument("--st1_per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--st1_per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--st1_learning_rate", type=float, default=2e-5)
    parser.add_argument("--st1_weight_decay", type=float, default=0.01)
    parser.add_argument("--st2_epochs", type=int, default=10)

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