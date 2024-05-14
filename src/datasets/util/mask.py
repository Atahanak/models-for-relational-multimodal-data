import numpy as np
import pandas as pd
from enum import Enum

import torch_geometric
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

import torch_frame
import torch
from icecream import ic


class PretrainType(Enum):
    MASK = 1
    MASK_VECTOR = 2
    LINK_PRED = 3


def apply_transformation(self: torch_frame.data.Dataset,
                         src_column: str, dst_column: str,
                         maskable_columns: list[str],
                         col_to_stype: dict[str, torch_frame.stype],
                         transformation: PretrainType) -> dict[str, torch_frame.stype]:
    if transformation == PretrainType.MASK:
        return apply_mask(self, maskable_columns, col_to_stype)
    elif transformation == PretrainType.MASK_VECTOR:
        return apply_mask_vector(self, maskable_columns, col_to_stype)
    else:
        return apply_link_pred(self, src_column, dst_column, col_to_stype)


def set_target_col(self: torch_frame.data.Dataset, pretrain: set[PretrainType],
                   col_to_stype: dict[str, torch_frame.stype]) -> dict[str, torch_frame.stype]:
    if {PretrainType.MASK, PretrainType.LINK_PRED}.issubset(pretrain):
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
        self.target_col = 'mask'
    elif PretrainType.LINK_PRED in pretrain:
        self.target_col = 'link'
    elif PretrainType.MASK_VECTOR in pretrain:
        self.target_col = 'mask_vector'
    else:
        col_to_stype['Is Laundering'] = torch_frame.categorical
        self.target_col = 'Is Laundering'
    return col_to_stype


def apply_mask(self: torch_frame.data.Dataset, maskable_columns: list[str], col_to_stype: dict[str, torch_frame.stype]) \
        -> dict[str, torch_frame.stype]:
    self.df['mask'] = None
    self.df = self.df.apply(_mask_column, args=(maskable_columns,), axis=1)
    col_to_stype['mask'] = torch_frame.mask
    return col_to_stype


def apply_mask_vector(self: torch_frame.data.Dataset, maskable_columns: list[str],
                      col_to_stype: dict[str, torch_frame.stype], p_m: float = 0.4):
    # TODO: implement mask vector transformation, by (1) generating a random vector of 0s and 1s and
    #  (2) imputing the "masked" values from the distribution of the selected column
    self.df['mask_vector'] = None

    # Helper function to impute masked values from the column's distribution
    def _impute_mask_vector(row: pd.Series):
        # TODO: potential problem being that we sample the same value as we are replacing
        mask_vector = _generate_mask_vector(maskable_columns, p_m)
        original_value_col_pairs = []
        for mask_column in mask_vector:
            original_value_col_pairs.append([row[mask_column], mask_column])
            col_ser = self.df[mask_column]
            replacement = col_ser[col_ser != row[mask_column]].sample().values[0]
            # print("Changed value from ", row[mask_column], " to ", replacement, " in column ", mask_column)
            row[mask_column] = replacement
        # Store two vectors to predict: the previous values (reconstruction) and mask vector (mask prediction)
        row['mask_vector'] = original_value_col_pairs
        return row
    col_to_stype['mask_vector'] = torch_frame.mask_vector
    self.df = self.df.apply(_impute_mask_vector, axis=1)
    return col_to_stype


def apply_link_pred(self: torch_frame.data.Dataset,
                    src_column: str, dst_column: str,
                    col_to_stype: dict[str, torch_frame.stype]) \
        -> dict[str, torch_frame.stype]:
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


# Randomly mask a column of each row and store original value and max index
def _mask_column(row: pd.Series, maskable_cols):
    col_to_mask = np.random.choice(maskable_cols)  # Choose a column randomly
    original_value = row[col_to_mask]
    row['mask'] = [original_value, col_to_mask]  # Store original value and max index in 'mask' column

    # row[col_to_mask] = np.nan
    # hack to escape nan error in torch_frame
    if col_to_mask in ['Amount Received', 'Amount Paid']:
        row[col_to_mask] = -1
    else:
        row[col_to_mask] = '[MASK]'
    return row


def _generate_mask_vector(maskable_vector: list[str], p_m: float):
    # Takes a vector where 0s are not maskable and 1s are maskable, and returns a vector with 1s masked with probability p_m
    return [col for col in maskable_vector if np.random.rand() < p_m]

