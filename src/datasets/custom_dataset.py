import torch
import torch_frame
import pandas as pd
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.typing import TaskType
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from icecream import ic

class Custom_Dataset(torch_frame.data.Dataset):
        _dataset_stype_to_col = {
            'amazon_fashion': {
                torch_frame.categorical: ['overall'],
                torch_frame.text_embedded: ['reviewText']
                # overall             int64
                # verified             bool
                # reviewTime         object
                # reviewerID         object
                # asin               object
                # reviewerName       object
                # reviewText         object
                # summary            object
                # unixReviewTime      int64
                # vote              float64
                # style              object
                # image              object
            }
        }

        _dataset_target_col = {
            'amazon_fashion': 'overall'
        }

        r"""
        Args:
            root (str): Root directory of the dataset.
        """
        def __init__(
            self, 
            root: str, 
            task_type: TaskType,
            name: str = 'amazon_fashion',
            text_stype: torch_frame.stype = torch_frame.text_embedded,
            col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
            | TextEmbedderConfig | None = None,
            col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
            | TextTokenizerConfig | None = None,
            pretrain='', 
            split_type='temporal', 
            splits=[0.6, 0.2, 0.2], 
            khop_neighbors=[100, 100], 
            nrows=None
        ):
            self.root = root
            self._task_type = task_type
            self.name = name
            if not text_stype.is_text_stype:
                raise ValueError(f"`text_stype` should be a text stype, "
                             f"got {text_stype}.")
            self.text_stype = text_stype
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors
            
            target_col = self._dataset_target_col[self.name]
            stype_to_col = self._dataset_stype_to_col[name]

            col_to_stype = {}
            for stype in stype_to_col:
                cols = stype_to_col[stype]
                for col in cols:
                    if stype == torch_frame.text_embedded:
                        col_to_stype[col] = self.text_stype
                    else:
                        col_to_stype[col] = stype

            self.df = pd.read_csv(root, header=0, nrows=nrows, usecols=col_to_stype.keys())
            self.df = self.df.dropna()
            self.df = self.df.reset_index(drop=True)
            # ic(self.df.columns)
            # ic(self.df.shape)

            self.random_split()
            ic(self.df.head())
            # estimate size of the dataset in GB
            self.size = self.df.memory_usage(deep=True).sum() / 1e9
            ic(self.size)

            super().__init__(self.df, col_to_stype, target_col=target_col, 
                             split_col='split', 
                             col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                             col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg)
        
        def random_split(self, seed=42):
            self.df['split'] = torch_frame.utils.generate_random_split(length=len(self.df), seed=0, train_ratio=0.8, val_ratio=0.1)
            ic(len(self.df))
            ic(self.df['split'].value_counts())