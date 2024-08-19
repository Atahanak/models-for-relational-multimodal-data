import os

import numpy as np
from enum import Enum

import torch_frame

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
        np.save(dir_masked_columns, mask)
    return mask
    
def set_target_col(self: torch_frame.data.Dataset, pretrain: set[PretrainType],
                   col_to_stype: dict[str, torch_frame.stype], supervised_col: str) -> dict[str, torch_frame.stype]:
    # Handle supervised column
    if not pretrain:
        # !!! self.df['Is Laundering'] column stores strings "0" and "1".
        #self.df['target'] = self.df[supervised_col].apply(lambda x: [float(x)]) + self.df['link'] 
        if 'link' in self.df.columns:
            self.df['target'] = self.df.apply(lambda row: [float(row[supervised_col])] + row['link'], axis=1)
        else:
            self.df['target'] = self.df[supervised_col].apply(lambda x: [float(x)])
        self.target_col = 'target'
        col_to_stype['target'] = torch_frame.relation
        if 'link' in col_to_stype:
            del col_to_stype['link']
        if 'link' in self.df.columns:
            self.df = self.df.drop(columns=['link'])
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
    
    if supervised_col is not None:
        self.df['target'] = self.df['target'] + self.df[supervised_col]
    return col_to_stype
