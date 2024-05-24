import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from icecream import ic
from peft import LoraConfig, get_peft_model
from torch import Tensor
from src.utils import mean_pooling
from torch_frame.nn.encoder.stype_encoder import MultiNestedTensor
from torch_frame.typing import TextTokenizationOutputs


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
    def __init__(self, model: str, peft_config: LoraConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

        if peft_config is not None:
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