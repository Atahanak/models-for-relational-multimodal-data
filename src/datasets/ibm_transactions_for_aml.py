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
        def __init__(self, root, mask_type="replace", pretrain: set[PretrainType] = set(), split_type='temporal', splits=[0.6, 0.2, 0.2], khop_neighbors=[100, 100], masked_dir="/tmp/.cache/masked_columns"):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors

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
            #num_columns = ['Amount Received', 'Amount Paid']
            num_columns = ['Amount Paid']
            cat_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']

            # Split into train, validation, test sets
            self.df = apply_split(self.df, self.split_type, self.splits, "Timestamp")

            col_to_stype = create_graph(self, col_to_stype, "From ID", "To ID")

            # Apply input corruption
            if PretrainType.MASK in pretrain:
                # Create mask vector
                mask = create_mask(self, num_columns + cat_columns, masked_dir)
                self.df["maskable_column"] = mask
                col_to_stype = apply_mask(self, cat_columns, num_columns, col_to_stype, mask_type)
                # for transformation in pretrain:
                #     col_to_stype = apply_transformation(self, "From ID", "To ID", cat_columns, num_columns, col_to_stype, transformation, mask_type)
                # Remove columns that are not needed
                self.df = self.df.drop('maskable_column', axis=1)

            # Define target column to predict
            col_to_stype = set_target_col(self, pretrain, col_to_stype, "Is Laundering")
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
            
            row = [int(edge[0]) for edge in edges]
            col = [int(edge[1]) for edge in edges]
            idx = [int(edge[2]) for edge in edges]

            input = EdgeSamplerInput(None, torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long))
            out = self.sampler.sample_from_edges(input)
            
            perm = self.sampler.edge_permutation 
            e_id = perm[out.edge] if perm is not None else out.edge

            edge_set = set()
            for id in idx:
                edge_set.add(id)
            for _, v in enumerate(e_id.numpy()):
                assert self.edges[v][2] == v #sanity check
                if v not in edge_set:
                    row.append(self.edges[v][0])
                    col.append(self.edges[v][1])
                    idx.append(v)
            khop_row = torch.tensor(row, dtype=torch.long)
            khop_col = torch.tensor(col, dtype=torch.long)
            return khop_row, khop_col, idx
        
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
