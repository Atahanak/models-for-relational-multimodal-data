import os

import numpy as np
import pandas as pd
from enum import Enum

import torch_geometric
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

import torch_frame
import torch
from collections import Counter

import time
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

class PretrainType(Enum):
    MASK = 1
    MASK_VECTOR = 2
    LINK_PRED = 3

# def create_graph(self, col_to_stype, src_column, dst_column):
#     self.sampler = None
#     # TODO: update Mapper to handle non-integer ids, e.g. strings | Assumes ids are integers and starts from 0!
#     self.df['link'] = self.df[[src_column, dst_column]].apply(list, axis=1)
#     col_to_stype['link'] = torch_frame.relation

#     def append_index_to_link(row):
#         row['link'].append(float(row.name))
#         return row

#     self.df = self.df.apply(append_index_to_link, axis=1)

#     # get number of uique ids in the dataset
#     num_nodes = len(set(self.df[src_column].to_list() + self.df[dst_column].to_list()))

#     # init train and val graph
#     # convert slef.df['link'] to torch tensor
#     self.edges = torch.tensor(self.df['link'].to_list(), dtype=torch.long)
#     #self.edges = self.df['link'].to_numpy()
#     #self.train_edges = self.df[self.df['split'] == 0]['link'].to_numpy()
#     self.train_edges = torch.tensor(self.df[self.df['split'] == 0]['link'].to_list(), dtype=torch.long)
#     # self.train_edges = self.edges
#     # val_edges = self.df[self.df['split'] == 1]['link'].to_numpy()

#     #source = torch.tensor([int(edge[0]) for edge in self.train_edges], dtype=torch.long)
#     source = self.train_edges[:, 0]
#     #destination = torch.tensor([int(edge[1]) for edge in self.train_edges], dtype=torch.long)
#     destination = self.train_edges[:, 1]
#     #ids = torch.tensor([int(edge[2]) for edge in self.train_edges], dtype=torch.long)
#     ids = self.train_edges[:, 2]
#     train_edge_index = torch.stack([source, destination], dim=0)
#     x = torch.arange(num_nodes)
#     start = time.time()
#     self.train_graph = torch_geometric.data.Data(x=x, edge_index=train_edge_index, edge_attr=ids)
#     self.sampler = NeighborSampler(self.train_graph, num_neighbors=self.khop_neighbors)
#     print(f"Time to create graph: {time.time() - start}")
#     return col_to_stype

def create_graph(self, col_to_stype, src_column, dst_column):

    # Convert src and dst columns to tensors directly
    src = torch.tensor(self.df[src_column].values, dtype=torch.long)
    dst = torch.tensor(self.df[dst_column].values, dtype=torch.long)
    
    # Create edge index tensor
    edge_index = torch.stack([src, dst], dim=0)
    
    # Create edge attributes (ids)
    ids = torch.arange(len(src), dtype=torch.float)
    
    # Compute number of unique nodes
    num_nodes = len(torch.unique(edge_index))

    # Create node features
    x = torch.arange(num_nodes)

    # Create the full graph
    self.edges = torch.cat([edge_index, ids.unsqueeze(0)], dim=0).t()
    # Create the 'link' column in the DataFrame
    self.df['link'] = self.edges.tolist()

    # Create train graph
    train_mask = self.df['split'] == 0
    train_mask = torch.tensor(train_mask.to_numpy(), dtype=torch.bool)
    train_edge_index = edge_index[:, train_mask]
    train_ids = ids[train_mask]
    self.train_graph = torch_geometric.data.Data(x=x, edge_index=train_edge_index, edge_attr=train_ids)
    self.train_sampler = NeighborSampler(self.train_graph, num_neighbors=self.khop_neighbors)

    # Create val graph
    val_mask = val_mask = self.df['split'].isin([0, 1])
    val_mask = torch.tensor(val_mask.to_numpy(), dtype=torch.bool)
    val_edge_index = edge_index[:, val_mask]
    val_ids = ids[val_mask]
    self.val_graph = torch_geometric.data.Data(x=x, edge_index=val_edge_index, edge_attr=val_ids)
    self.val_sampler = NeighborSampler(self.val_graph, num_neighbors=self.khop_neighbors)

    # Create test graph
    test_edge_index = edge_index
    test_ids = ids
    self.test_graph = torch_geometric.data.Data(x=x, edge_index=test_edge_index, edge_attr=test_ids)
    self.test_sampler = NeighborSampler(self.test_graph, num_neighbors=self.khop_neighbors)

    # Update col_to_stype
    col_to_stype['link'] = torch_frame.relation
    
    return col_to_stype

def create_mask(self, maskable_columns: list[str]):
    # Generate which columns to mask and store in file for reproducibility across different runs
    dir_masked_columns = self.root + ".npy"
    if os.path.exists(dir_masked_columns):
        logger.info(f"Loading masked columns from {dir_masked_columns}")
        mask = np.load(dir_masked_columns)
    else:
        logger.info(f"Creating masked columns and saving to {dir_masked_columns}")
        mask = np.random.choice(maskable_columns, size=self.df.shape[0], replace=True)
        # # Check for NaN values and reselect columns if needed
        # for i in range(self.df.shape[0]):
        #     while pd.isna(self.df.loc[i, mask[i]]):
        #         mask[i] = np.random.choice(maskable_columns)
        np.save(dir_masked_columns, mask)
    return mask
    
def set_target_col(self: torch_frame.data.Dataset, pretrain: set[PretrainType],
                   col_to_stype: dict[str, torch_frame.stype], supervised_col: str) -> dict[str, torch_frame.stype]:
    # Handle supervised column
    if not pretrain:
        # !!! self.df['Is Laundering'] column stores strings "0" and "1".
        #self.df['target'] = self.df[supervised_col].apply(lambda x: [float(x)]) + self.df['link'] 
        self.df['target'] = self.df.apply(lambda row: [float(row[supervised_col])] + row['link'], axis=1)
        self.target_col = 'target'
        col_to_stype['target'] = torch_frame.relation
        self.df = self.df.drop(columns=['link'])
        del col_to_stype['link']
        return col_to_stype

    # Handle pretrain columns
    if {PretrainType.MASK, PretrainType.LINK_PRED}.issubset(pretrain):
        # Handles combinations of {MCM+MV+LP} and {MCM+LP}
        # merge link and mask columns into a column called target
        self.df['target'] = self.df['mask'] + self.df['link']
        col_to_stype['target'] = torch_frame.mask
        self.target_col = 'target'
        self.df = self.df.drop(columns=['link', 'mask'])
        del col_to_stype['link']
        del col_to_stype['mask']
    elif PretrainType.MASK in pretrain:
        # Handles combinations of {MCM+MV} and {MCM}
        self.target_col = 'mask'
        if 'link' in col_to_stype:
            del col_to_stype['link']
    elif PretrainType.LINK_PRED in pretrain:
        self.target_col = 'link'
        if 'mask' in col_to_stype:
            del col_to_stype['mask']
    else:
        self.target_col = ''
    return col_to_stype

# def apply_mask(self: torch_frame.data.Dataset, cat_columns: list[str], num_columns: list[str],
#                col_to_stype: dict[str, torch_frame.stype], mask_type: str) -> dict[str, torch_frame.stype]:
#     maskable_columns = cat_columns + num_columns

#     def _impute_mask_vector(row: pd.Series):
#         # 1. Get which column we have chosen to mask
#         masked_column = row["maskable_column"]
#         original_value = row[masked_column]
#         # 2. Choose a replacement from prob distribution
#         if masked_column in cat_columns:
#             # Don't select the original value
#             cat_values = list(distributions_cat[masked_column].keys())
#             p_original = distributions_cat[masked_column][original_value]
#             replacement = np.random.choice(cat_values,
#                                            p=[p + (p_original/(len(cat_values)-1)) if cat_values[i] != original_value else 0
#                                               for i, p in enumerate(distributions_cat[masked_column].values())])
#         elif masked_column in num_columns:
#             replacement = np.random.normal(distributions_num[masked_column][0], distributions_num[masked_column][1])
#         row['mask'] = [original_value, masked_column]
#         row[masked_column] = replacement
#         return row

#     # Prepare values to impute for faster computation
#     if mask_type != "remove":
#         counter_cat = {col: Counter(self.df[col]) for col in cat_columns}
#         distributions_cat = dict()
#         for cat_column in cat_columns:
#             s = sum(counter_cat[cat_column].values())
#             distributions_cat[cat_column] = {k: v / s for k, v in counter_cat[cat_column].items()}
#         distributions_num = {col: (self.df[col].mean(), self.df[col].std()) for col in num_columns}

#     # Prepare values to remove for faster computation
#     if mask_type != "replace":
#         avg_per_num_col = {col: self.df[col].mean() for col in num_columns}

#     # Apply mask to the dataset
#     self.df['mask'] = None
#     if mask_type == "remove":
#         self.df = self.df.apply(_mask_column, args=(avg_per_num_col,), axis=1)
#     elif mask_type == "replace":
#         self.df = self.df.apply(_impute_mask_vector, axis=1)
#     elif mask_type == "bert":
#         def _choose_mask_type(row: pd.Series):
#             p = np.random.rand()
#             if p < 0.8:
#                 return _mask_column(row, avg_per_num_col)
#             elif p < 0.9:
#                 return _impute_mask_vector(row)
#             else:
#                 mask_column = row["maskable_column"]
#                 original_value = row[mask_column]
#                 row['mask'] = [original_value, mask_column]
#                 return row

#         self.df = self.df.apply(_choose_mask_type, axis=1)

#     col_to_stype['mask'] = torch_frame.mask
#     return col_to_stype

# # Randomly mask a column of each row and store original value and max index
# def _mask_column(row: pd.Series, avg_per_num_col):
#     col_to_mask = row["maskable_column"]  # Choose a column randomly
#     original_value = row[col_to_mask]
#     row['mask'] = [original_value, col_to_mask]  # Store original value and max index in 'mask' column

#     # row[col_to_mask] = np.nan
#     # hack to escape nan error in torch_frame
#     if col_to_mask in avg_per_num_col.keys():
#         row[col_to_mask] = avg_per_num_col[col_to_mask]
#     else:
#         row[col_to_mask] = '[MASK]'
#     return row

def apply_mask(self: torch_frame.data.Dataset, cat_columns: list[str], num_columns: list[str],
               col_to_stype: dict[str, torch_frame.stype], mask_type: str) -> dict[str, torch_frame.stype]:
    
    # Prepare values for imputation and removal upfront
    distributions_cat = {}
    distributions_num = {}
    avg_per_num_col = {}
    
    if mask_type != "remove":
        for col in cat_columns:
            counter = Counter(self.df[col])
            total = sum(counter.values())
            distributions_cat[col] = {k: v / total for k, v in counter.items()}
        
        distributions_num = {col: (self.df[col].mean(), self.df[col].std()) for col in num_columns}
    
    if mask_type != "replace":
        avg_per_num_col = {col: self.df[col].mean() for col in num_columns}
    
    # mask_col = np.random.choice(maskable_columns, size=len(self.df)) #, replace=False)
    mask_col = create_mask(self, num_columns + cat_columns)
    self.df['maskable_column'] = mask_col
    original_values = self.df.apply(lambda row: row[row['maskable_column']], axis=1)
    self.df['mask'] = [list(x) for x in zip(original_values, mask_col)]
    
    if mask_type == "remove":
        self.df = mask_remove(self.df, avg_per_num_col)
    elif mask_type == "replace":
        self.df = mask_replace(self.df, distributions_cat, distributions_num, cat_columns, num_columns)
    elif mask_type == "bert":
        self.df = mask_bert(self.df, avg_per_num_col, distributions_cat, distributions_num, cat_columns, num_columns)

    col_to_stype['mask'] = torch_frame.mask
    self.df = self.df.drop('maskable_column', axis=1)
    return col_to_stype

def mask_remove(df, avg_per_num_col):
    for col in avg_per_num_col:
        mask = df['maskable_column'] == col
        df.loc[mask, col] = avg_per_num_col[col]
    
    cat_mask = ~df['maskable_column'].isin(avg_per_num_col)
    df.loc[cat_mask, df.loc[cat_mask, 'maskable_column']] = '[MASK]'
    
    return df

# def mask_replace(df, distributions_cat, distributions_num, cat_columns, num_columns):
#     for col in cat_columns:
#         mask = df['maskable_column'] == col
#         if mask.any():
#             values = list(distributions_cat[col].keys())
#             probs = list(distributions_cat[col].values())
#             df.loc[mask, col] = np.random.choice(values, size=mask.sum(), p=probs)

#             cat_values = list(distributions_cat[masked_column].keys())
#             p_original = distributions_cat[masked_column][original_value]
#             replacement = np.random.choice(cat_values,
#                                            p=[p + (p_original/(len(cat_values)-1)) if cat_values[i] != original_value else 0
#                                               for i, p in enumerate(distributions_cat[masked_column].values())])
    
#     for col in num_columns:
#         mask = df['maskable_column'] == col
#         if mask.any():
#             mean, std = distributions_num[col]
#             df.loc[mask, col] = np.random.normal(mean, std, size=mask.sum())
    
#     return df
def safe_isnan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False

def mask_replace(df, distributions_cat, distributions_num, cat_columns, num_columns):
    for col in cat_columns:
        mask = df['maskable_column'] == col
        if mask.any():
            values = list(distributions_cat[col].keys())
            probs = list(distributions_cat[col].values())
            # print(values)
            # print(probs)
            
            # Get original values for the masked column
            original_values = df.loc[mask, col].values
            
            # Create a function to adjust probabilities for each row
            def adjust_probs(orig_value):
                p_original = distributions_cat[col][orig_value]
                # and not (np.isnan(values[i]) and np.isnan(orig_value))
                #adj = [p + (p_original/(len(values)-1)) if (values[i] != orig_value) and not (safe_isnan(values[i]) and safe_isnan(orig_value)) else 0
                #adj = [p + (p_original/(len(values)-1)) if (values[i] != orig_value) and not (safe_isnan(values[i])) else 0
                adj = [p + (p_original/(len(values)-1)) if (values[i] != orig_value) else 0
                        for i, p in enumerate(probs)]
                # if sum(adj) != 1:
                #     for i, p in enumerate(probs):
                #         print(values[i], orig_value, values[i] == orig_value, safe_isnan(values[i]), safe_isnan(orig_value))
                #     print(col)
                #     print(len(values))
                #     print(orig_value)
                #     print(p_original)
                #     print(adj)
                #     assert sum(adj) == 1   
                return adj
            
            # # Apply the probability adjustment for each row
            adjusted_probs = [adjust_probs(ov) for ov in original_values]

            # Use list comprehension for random choice
            replacements = [np.random.choice(values, p=p) for p in adjusted_probs]
            
            df.loc[mask, col] = replacements

    for col in num_columns:
        mask = df['maskable_column'] == col
        if mask.any():
            mean, std = distributions_num[col]
            df.loc[mask, col] = np.random.normal(mean, std, size=mask.sum())
    
    return df

def mask_bert(df, avg_per_num_col, distributions_cat, distributions_num, cat_columns, num_columns):
    probs = np.random.rand(len(df))
    remove_mask = probs < 0.8
    replace_mask = (probs >= 0.8) & (probs < 0.9)
    
    df.loc[remove_mask] = mask_remove(df.loc[remove_mask], avg_per_num_col)
    df.loc[replace_mask] = mask_replace(df.loc[replace_mask], distributions_cat, distributions_num, cat_columns, num_columns)
    
    return df
