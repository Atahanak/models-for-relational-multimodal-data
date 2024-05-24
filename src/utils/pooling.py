import torch
from torch import Tensor
from icecream import ic

def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()).float())
    embedding = torch.sum(
        last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return embedding.unsqueeze(1)

def last_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    # Find the last token that attends to previous tokens.
    sequence_lengths = attention_mask.sum(dim=1) - 1
    # ic(last_hidden_state.shape)
    # ic(sequence_lengths.shape)
    # ic(sequence_lengths)
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths]