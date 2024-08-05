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
from .util.mask import PretrainType, set_target_col, create_mask
from .util.graph import create_graph, add_ports
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

class IBMTransactionsAML(torch_frame.data.Dataset):
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
        def __init__(self, root, mask_type="replace", pretrain: set[PretrainType] = set(), split_type='temporal_daily', splits=[0.6, 0.2, 0.2], khop_neighbors=[100, 100], ports=False):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors
            self.timestamp_col = 'Timestamp'
            self.pretrain = pretrain
            self.mask_type = mask_type

            names = [
                'Timestamp',
                'From Bank',
                'From ID',
                'To Bank',
                'To ID',
                'Amount Received',
                'Receiving Currency',
                'Amount Paid',
                'Payment Currency',
                'Payment Format',
                'Is Laundering',
            ]
            dtypes = {
                'From Bank': 'category',
                'From ID': 'float64',
                'To Bank': 'category',
                'To ID': 'float64',
                'Amount Received': 'float64',
                'Receiving Currency': 'category',
                'Amount Paid': 'float64',
                'Payment Currency': 'category',
                'Payment Format': 'category',
                'Is Laundering': 'category',
            }

            self.df = pd.read_csv(root, names=names, dtype=dtypes, header=0)         
            col_to_stype = {
                'From Bank': torch_frame.categorical,
                'To Bank': torch_frame.categorical,
                'Payment Currency': torch_frame.categorical,
                'Receiving Currency': torch_frame.categorical,
                'Payment Format': torch_frame.categorical,
                'Timestamp': torch_frame.timestamp,
                'Amount Paid': torch_frame.numerical,
                #'Amount Received': torch_frame.numerical
            }
            # #num_columns = ['Amount Received', 'Amount Paid']
            # #self.num_columns = ['Amount Received']
            # self.num_columns = ['Amount Paid']
            # #self.cat_columns = ['Receiving Currency', 'Payment Format']
            # self.cat_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
            # self.maskable_columns = self.num_columns + self.cat_columns

            # Split into train, validation, test sets
            self.df = apply_split(self.df, self.split_type, self.splits, self.timestamp_col)
            
            logger.info(f'Creating graph...')
            start = time.time()
            col_to_stype = create_graph(self, col_to_stype, "From ID", "To ID")
            logger.info(f'Graph created in {time.time()-start:.2f} seconds.')

            if ports:
                logger.info(f'Adding ports...')
                start = time.time()
                add_ports(self)
                col_to_stype['in_port'] = stype.numerical
                col_to_stype['out_port'] = stype.numerical
                logger.info(f'Ports added in {time.time()-start:.2f} seconds.')

            if PretrainType.MASK in pretrain:
                self.num_columns = ['Amount Paid']
                self.cat_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
                self.maskable_columns = self.num_columns + self.cat_columns
                mask_col = create_mask(self, self.maskable_columns)
                self.df['maskable_column'] = mask_col
                original_values = self.df.apply(lambda row: row[row['maskable_column']], axis=1)
                self.df['mask'] = [list(x) for x in zip(original_values, mask_col)]
                col_to_stype['mask'] = torch_frame.mask
            else:
                self.maskable_columns = None

            col_to_stype = set_target_col(self, pretrain, col_to_stype, "Is Laundering")
            super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col, maskable_columns= self.maskable_columns)

        def sample_neighbors(self, edges, mode="train") -> (torch.Tensor, torch.Tensor): # type: ignore
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

                data/src/datasets/ibm_transactions_for_aml.py:123: UserWarning: To copy construct from a tensor, it is recommended to use 
                sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
            """
            edges = torch.tensor(edges, dtype=torch.int)
            row = edges[:, 0]
            col = edges[:, 1]
            idx = edges[:, 2] 
            input = EdgeSamplerInput(None, torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long))
            
            if mode == 'train':
                out = self.train_sampler.sample_from_edges(input)
                perm = self.train_sampler.edge_permutation 
            elif mode == 'val':
                out = self.val_sampler.sample_from_edges(input)
                perm = self.val_sampler.edge_permutation 
            elif mode =='test':
                out = self.test_sampler.sample_from_edges(input)
                perm = self.test_sampler.edge_permutation 
            else:
                raise ValueError("Invalid sampling mode! Valid values: ['train', 'val', 'test']")

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
