import os

import numpy as np
from enum import Enum

import torch_frame
from collections import Counter

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
