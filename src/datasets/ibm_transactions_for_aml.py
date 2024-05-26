import os.path

import torch
import torch_frame
import torch_geometric
from torch_geometric.sampler.neighbor_sampler import NeighborSampler
from torch_geometric.sampler import EdgeSamplerInput
import pandas as pd

from icecream import ic
import numpy as np
from src.datasets.util.mask import PretrainType, apply_transformation, set_target_col
from src.datasets.util.split import apply_split


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
        def __init__(self, root, mask_type, pretrain: set[PretrainType] = None, split_type='temporal', splits=[0.6, 0.2, 0.2], khop_neighbors=[100, 100]):
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
                'Amount Received': torch_frame.numerical
            }
            num_columns = ['Amount Received', 'Amount Paid']
            cat_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']

            # Generate which columns to mask and store in file for reproducibility across different runs
            if os.path.exists("/scratch/imcauliffe/masked_columns.npy"):
                mask = np.load("/scratch/imcauliffe/masked_columns.npy")
            else:
                maskable_columns = num_columns + cat_columns
                mask = np.random.choice(maskable_columns, size=self.df.shape[0], replace=True)
                np.save("/scratch/imcauliffe/masked_columns.npy", mask)

            # Split into train, validation, test sets
            self.df = apply_split(self.df, self.split_type, self.splits)

            # Apply input corruption
            self.df["maskable_column"] = mask
            if pretrain:
                for transformation in pretrain:
                    col_to_stype = apply_transformation(self, "From ID", "To ID", cat_columns, num_columns, col_to_stype, transformation, mask_type)

            # Define target column to predict
            col_to_stype = set_target_col(self, pretrain, col_to_stype)

            # Remove columns that are not needed
            self.df = self.df.drop('maskable_column', axis=1)

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
                # else:
                #     #ic(v)
                #     pass
            khop_row = torch.tensor(row, dtype=torch.long)
            khop_col = torch.tensor(col, dtype=torch.long)
            #else:
            #    idx = torch.hstack((torch.tensor(idx, dtype=torch.long), out.edge)) 
            #    khop_row = torch.hstack((torch.tensor(row, dtype=torch.long), out.row))
            #    khop_col = torch.hstack((torch.tensor(col, dtype=torch.long), out.col))

            #ic(len(khop_row), len(khop_col), len(idx))
            #ic(self.df.loc[out.edge].iloc[0])
            return khop_row, khop_col, idx