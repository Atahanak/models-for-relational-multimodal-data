import torch
from transformers import (AutoModel, AutoTokenizer)
from torch import Tensor
from icecream import ic
from src.utils import mean_pooling, last_pooling

class TextToEmbedding:
    def __init__(self, model: str, device: torch.device):
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if model == "intfloat/e5-mistral-7b-instruct":
            # Use last pooling here because this model is
            # a decoder (causal) language model that only
            # the last token attends to all previous tokens:
            self.pooling = "last"
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(model).to(device)
            self.pooling = "mean"
        embedding_model_learnable_params = sum( p.numel() for p in self.model.parameters() if p.requires_grad)
        ic(embedding_model_learnable_params)
    def __call__(self, sentences: list[str]) -> Tensor:
        if self.model_name == "intfloat/e5-mistral-7b-instruct":
            sentences = [(f"Instruct: Retrieve relevant knowledge and "
                          f"embeddings.\nQuery: {sentence}")
                         for sentence in sentences]
            max_length = 4096
            inputs = self.tokenizer(
                sentences,
                max_length=max_length - 1,
                truncation=True,
                return_attention_mask=False,
                padding=False,
            )
            inputs["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in inputs["input_ids"]
            ]
            inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
        else:
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
        if self.pooling == "mean":
            return (mean_pooling(out.last_hidden_state.detach(),
                                 mask).squeeze(1).cpu())
        elif self.pooling == "last":
            return last_pooling(out.last_hidden_state,
                                mask).detach().cpu().to(torch.float32)
        else:
            raise ValueError(f"{self.pooling} is not supported.")
    