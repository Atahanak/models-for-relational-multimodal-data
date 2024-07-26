import os.path

import torch
import torch_frame
from torch_geometric.sampler import EdgeSamplerInput
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

import pandas as pd
import numpy as np
from .util.mask import PretrainType, set_target_col, apply_mask, create_graph, create_mask
from .util.split import apply_split

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

class EthereumPhishingTransactions(torch_frame.data.Dataset):
        r"""`"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" https://arxiv.org/pdf/2306.16424.pdf`_.
        
        IBM Transactions for Anti-Money Laundering (AML) dataset.
        The dataset contains 10 columns:
        - Timestamp: The timestamp of the transaction.
        - From Bank: The bank from which the transaction is made.
        - From ID: The ID of the sender.
        - To Bank: The bank to which the transaction is made.
        - To ID: The ID of the receiver.
        - Amount Received: The amount received by the receiver.
        - Receiving Currency: The currency in which the amount is received.
        - Amount Paid: The amount paid by the sender.
        - Payment Currency: The currency in which the amount is paid.
        - Payment Format: The format of the payment.
        - Is Laundering: The label indicating whether the transaction is a money laundering transaction.

        Args:
            root (str): Root directory of the dataset.
            preetrain (bool): Whether to use the pretrain split or not (default: False).
        """
        def __init__(self, root, mask_type="replace", pretrain: set[PretrainType] = set(), split_type='temporal', splits=[0.6, 0.2, 0.2], khop_neighbors=[100, 100], masked_dir="/tmp/.cache/masked_columns"):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors

            names = [
                'nonce',
                'from_address',
                'to_address',
                #'transaction_index',
                'value',
                'gas',
                'gas_price',
                # 'receipt_status',
                'block_timestamp',
                # 'phishing',
            ]
            dtypes = {
                'nonce': 'float64',
                'from_address': 'float64',
                'to_address': 'float64',
                #'transaction_index': 'category',
                'value': 'float64',
                'gas': 'float64',
                'gas_price': 'float64',
                # 'receipt_status': 'category',
                'block_timestamp': 'float64',
                # 'phishing': 'category',
            }

            self.df = pd.read_csv(root, names=names, dtype=dtypes, header=0)         
            self.df.reset_index(inplace=True)
            
            col_to_stype = {
                'nonce': torch_frame.numerical,
                #'transaction_index': torch_frame.categorical,
                'value': torch_frame.numerical,
                'gas': torch_frame.numerical,
                'gas_price': torch_frame.numerical,
                # 'receipt_status': torch_frame.categorical,
                'block_timestamp': torch_frame.timestamp,
            }
            num_columns = ['nonce', 'value', 'gas', 'gas_price']
            #cat_columns = ['receipt_status']
            #cat_columns = ['transaction_index']
            cat_columns = []

            self.df = apply_split(self.df, self.split_type, self.splits, "block_timestamp")
            
            logger.info(f'Creating graph...')
            start = time.time()
            col_to_stype = create_graph(self, col_to_stype, "from_address", "to_address")
            logger.info(f'Graph created in {time.time()-start} seconds.')

            if PretrainType.MASK in pretrain:
                logger.info(f'Applying mask...')
                start = time.time()
                col_to_stype = apply_mask(self, cat_columns, num_columns, col_to_stype, mask_type)
                logger.info(f'Mask applied in {time.time()-start} seconds.')

            col_to_stype = set_target_col(self, pretrain, col_to_stype, "phishing")
            super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col)

        def sample_neighbors(self, edges, train=True) -> (torch.Tensor, torch.Tensor):
            """k-hop sampling.
            
            If k-hop sampling, this method **guarantees** that the first 
            ``n_seed_edges`` edges in the resulting table are the seed edges in the
            same order as given by ``idx``.

            Does not support multi-graphs

            Parameters
            ----------
            idx : int | list[int] | array
                Edge indices to use as seed for k-hop sampling.
            num_neighbors: int | list[int] | array
                Number of neighbors to sample for each seed edge.
            Returns
            -------
            pd.DataFrame
                Sampled edge data
            """
            edges = torch.tensor(edges, dtype=torch.int)
            row = edges[:, 0]
            col = edges[:, 1]
            idx = edges[:, 2] 
            input = EdgeSamplerInput(None, torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long))
            out = self.sampler.sample_from_edges(input)
            
            perm = self.sampler.edge_permutation 
            e_id = perm[out.edge] if perm is not None else out.edge

            is_new_edge = ~torch.isin(e_id, idx)
            new_edges = e_id[is_new_edge]

            if len(new_edges) > 0:
                row = torch.cat([row, self.edges[new_edges, 0]])
                col = torch.cat([col, self.edges[new_edges, 1]])
                idx = torch.cat([idx, new_edges])

            return row, col, idx
        
        def get_encoder(self, channels):
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
                stype.timestamp: TimestampEncoder(),
            }
            encoder = StypeWiseFeatureEncoder(
                        out_channels=channels,
                        col_stats=self.col_stats,
                        col_names_dict=self.tensor_frame.col_names_dict,
                        stype_encoder_dict=stype_encoder_dict,
            )
            return encoder
