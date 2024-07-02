import os

import numpy as np
import pandas as pd
from enum import Enum

import torch_geometric
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

import torch_frame
import torch
from icecream import ic
from collections import Counter


class PretrainType(Enum):
    MASK = 1
    MASK_VECTOR = 2
    LINK_PRED = 3


def create_graph(self, col_to_stype, src_column, dst_column):
    self.sampler = None
    # TODO: update Mapper to handle non-integer ids, e.g. strings | Assumes ids are integers and starts from 0!
    self.df['link'] = self.df[[src_column, dst_column]].apply(list, axis=1)
    col_to_stype['link'] = torch_frame.relation

    def append_index_to_link(row):
        row['link'].append(float(row.name))
        return row

    self.df = self.df.apply(append_index_to_link, axis=1)

    # get number of uique ids in the dataset
    num_nodes = len(set(self.df[src_column].to_list() + self.df[dst_column].to_list()))

    # init train and val graph
    self.edges = self.df['link'].to_numpy()
    self.train_edges = self.df[self.df['split'] == 0]['link'].to_numpy()
    # self.train_edges = self.edges
    # val_edges = self.df[self.df['split'] == 1]['link'].to_numpy()

    source = torch.tensor([int(edge[0]) for edge in self.train_edges], dtype=torch.long)
    destination = torch.tensor([int(edge[1]) for edge in self.train_edges], dtype=torch.long)
    ids = torch.tensor([int(edge[2]) for edge in self.train_edges], dtype=torch.long)
    train_edge_index = torch.stack([source, destination], dim=0)
    x = torch.arange(num_nodes)
    self.train_graph = torch_geometric.data.Data(x=x, edge_index=train_edge_index, edge_attr=ids)
    self.sampler = NeighborSampler(self.train_graph, num_neighbors=self.khop_neighbors)
    return col_to_stype


def create_mask(root, df, maskable_columns: list[str], masked_dir: str):
    # Generate which columns to mask and store in file for reproducibility across different runs
    os.makedirs(masked_dir, exist_ok=True)
    dir_masked_columns = masked_dir + root.split("/")[-1].removesuffix(".csv") + ".npy" # get name of file in root, and remove .csv
    print(dir_masked_columns)
    if os.path.exists(dir_masked_columns):
        mask = np.load(dir_masked_columns)
    else:
        # maskable_columns = num_columns + cat_columns
        mask = np.random.choice(maskable_columns, size=df.shape[0], replace=True)
        np.save(dir_masked_columns, mask)
    return mask

def set_target_col(self: torch_frame.data.Dataset, pretrain: set[PretrainType],
                   col_to_stype: dict[str, torch_frame.stype], supervised_col: str) -> dict[str, torch_frame.stype]:
    # Handle supervised column
    if not pretrain:
        col_to_stype[supervised_col] = torch_frame.categorical
        self.target_col = supervised_col
        return col_to_stype

    # Handle pretrain columns
    if {PretrainType.MASK, PretrainType.LINK_PRED}.issubset(pretrain):
        # Handles combinations of {MCM+MV+LP} and {MCM+LP}
        # merge link and mask columns into a column called target
        self.df['target'] = self.df['mask'] + self.df['link']
        col_to_stype['target'] = torch_frame.mask
        self.target_col = 'target'
        ic(self.df['link'][0:5])
        ic(self.df['mask'][0:5])
        ic(self.df['target'][0:5])
        self.df = self.df.drop(columns=['link', 'mask'])
        del col_to_stype['link']
        del col_to_stype['mask']
    elif PretrainType.MASK in pretrain:
        # Handles combinations of {MCM+MV} and {MCM}
        self.target_col = 'mask'
    elif PretrainType.LINK_PRED in pretrain:
        self.target_col = 'link'
    else:
        self.target_col = ''
    return col_to_stype


def apply_mask(self: torch_frame.data.Dataset, cat_columns: list[str], num_columns: list[str],
               col_to_stype: dict[str, torch_frame.stype], mask_type: str) -> dict[str, torch_frame.stype]:
    maskable_columns = cat_columns + num_columns

    def _impute_mask_vector(row: pd.Series):
        # 1. Get which column we have chosen to mask
        masked_column = row["maskable_column"]
        original_value = row[masked_column]
        # 2. Choose a replacement from prob distribution
        if masked_column in cat_columns:
            # Don't select the original value
            cat_values = list(distributions_cat[masked_column].keys())
            p_original = distributions_cat[masked_column][original_value]
            replacement = np.random.choice(cat_values,
                                           p=[p + (p_original/(len(cat_values)-1)) if cat_values[i] != original_value else 0
                                              for i, p in enumerate(distributions_cat[masked_column].values())])
        else:
            replacement = np.random.normal(distributions_num[masked_column][0], distributions_num[masked_column][1])
        row['mask'] = [original_value, masked_column]
        row[masked_column] = replacement
        return row

    # Prepare values to impute for faster computation
    if mask_type != "remove":
        counter_cat = {col: Counter(self.df[col]) for col in cat_columns}
        distributions_cat = dict()
        for cat_column in cat_columns:
            s = sum(counter_cat[cat_column].values())
            distributions_cat[cat_column] = {k: v / s for k, v in counter_cat[cat_column].items()}
        distributions_num = {col: (self.df[col].mean(), self.df[col].std()) for col in num_columns}

    # Prepare values to remove for faster computation
    if mask_type != "replace":
        avg_per_num_col = {col: self.df[col].mean() for col in num_columns}

    # Apply mask to the dataset
    self.df['mask'] = None
    if mask_type == "remove":
        self.df = self.df.apply(_mask_column, args=(avg_per_num_col,), axis=1)
    elif mask_type == "replace":
        self.df = self.df.apply(_impute_mask_vector, axis=1)
    elif mask_type == "bert":
        def _choose_mask_type(row: pd.Series):
            p = np.random.rand()
            if p < 0.8:
                return _mask_column(row, avg_per_num_col)
            elif p < 0.9:
                return _impute_mask_vector(row)
            else:
                mask_column = row["maskable_column"]
                original_value = row[mask_column]
                row['mask'] = [original_value, mask_column]
                return row

        self.df = self.df.apply(_choose_mask_type, axis=1)

    col_to_stype['mask'] = torch_frame.mask
    return col_to_stype


# Randomly mask a column of each row and store original value and max index
def _mask_column(row: pd.Series, avg_per_num_col):
    col_to_mask = row["maskable_column"]  # Choose a column randomly
    original_value = row[col_to_mask]
    row['mask'] = [original_value, col_to_mask]  # Store original value and max index in 'mask' column

    # row[col_to_mask] = np.nan
    # hack to escape nan error in torch_frame
    if col_to_mask in avg_per_num_col.keys():
        row[col_to_mask] = avg_per_num_col[col_to_mask]
    else:
        row[col_to_mask] = '[MASK]'
    return row


def _generate_mask_vector(maskable_vector: list[str], p_m: float = None, n: int = None):
    # Generates a mask vector for a given row, either with n masked elements or binomially with probability p_m
    if n is not None:
        return np.random.choice(maskable_vector, n, replace=False)
    else:
        return [col for col in maskable_vector if np.random.rand() < p_m]
