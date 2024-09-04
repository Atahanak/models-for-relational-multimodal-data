import os.path

import torch
import torch_frame
from torch_geometric.sampler import EdgeSamplerInput, NodeSamplerInput
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
from .util.graph import create_graph, add_ports, add_EgoIDs
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

class IBMTransactionsAML():
        def __init__(self, root, mask_type="replace", pretrain: set[PretrainType] = set(), split_type='random', splits=[0.65, 0.15, 0.2], khop_neighbors=[100, 100], ports=False, ego=False, channels=64):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors
            self.pretrain = pretrain
            self.mask_type = mask_type
            self.ego = ego

            logger.info(f'Creating edges...')
            self.edges = IBMTransactionsAMLTransactions(root, mask_type=mask_type, pretrain=pretrain, split_type=split_type, splits=splits, khop_neighbors=khop_neighbors, ports=ports)
            self.edges.materialize()
            self.edges.init_encoder(channels)
            logger.info(f'Edges created.')

            max_id = self.edges.df[['From ID', 'To ID']].max().max()
            logger.info(f'Creating nodes...')
            self.nodes = IBMTransactionsAMLNodes(max_id, ego=ego)
            self.nodes.materialize()
            self.nodes.init_encoder(channels)
            logger.info(f'Nodes created.')
            
            self.num_columns = [col for col in self.nodes.df.columns]
            self.cat_columns = []

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
            edges = edges.type(torch.long)
            row = edges[:, 0]
            col = edges[:, 1]
            idx = edges[:, 2] 
            input = EdgeSamplerInput(None, row.type(torch.long), col.type(dtype=torch.long))
            
            if mode == 'train':
                out = self.edges.train_sampler.sample_from_edges(input)
                perm = self.edges.train_sampler.edge_permutation 
            elif mode == 'val':
                out = self.edges.val_sampler.sample_from_edges(input)
                perm = self.edges.val_sampler.edge_permutation 
            elif mode =='test':
                out = self.edges.test_sampler.sample_from_edges(input)
                perm = self.edges.test_sampler.edge_permutation 
            else:
                raise ValueError("Invalid sampling mode! Valid values: ['train', 'val', 'test']")

            e_id = perm[out.edge] if perm is not None else out.edge

            is_new_edge = ~torch.isin(e_id, idx)
            new_edges = e_id[is_new_edge]

            if len(new_edges) > 0:
                row = torch.cat([row, self.edges.edges[new_edges, 0]])
                col = torch.cat([col, self.edges.edges[new_edges, 1]])
                idx = torch.cat([idx, new_edges])

            return row, col, idx

        def sample_neighbors_from_nodes(self, nodes, mode="train") -> (torch.Tensor, torch.Tensor): # type: ignore
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
            nodes = nodes.type(torch.long)
            input = NodeSamplerInput(None, nodes)
            
            if mode == 'train':
                out = self.edges.train_sampler.sample_from_nodes(input)
                perm = self.edges.train_sampler.edge_permutation 
            elif mode == 'val':
                out = self.edges.val_sampler.sample_from_nodes(input)
                perm = self.edges.val_sampler.edge_permutation 
            elif mode =='test':
                out = self.edges.test_sampler.sample_from_nodes(input)
                perm = self.edges.test_sampler.edge_permutation 
            else:
                raise ValueError("Invalid sampling mode! Valid values: ['train', 'val', 'test']")

            e_id = perm[out.edge] if perm is not None else out.edge
            row = self.edges.edges[e_id, 0]
            col = self.edges.edges[e_id, 1]
            idx = e_id

            return row, col, idx
    
        def get_graph_inputs(self, batch: torch_frame.TensorFrame, mode='train', args=None):

            y, edges = batch.y[:, :-3], batch.y[:,-3:]
            khop_source, khop_destination, idx = self.sample_neighbors(edges, mode)
            edge_attr = self.edges.tensor_frame.__getitem__(idx)
            edge_attr, _ = self.edges.encoder(edge_attr)

            nodes = torch.unique(torch.cat([khop_source, khop_destination])).type(torch.long)

            node_attr = self.nodes.tensor_frame.__getitem__(nodes)

            n_id_map = {value.item(): index for index, value in enumerate(nodes)}
            local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
            local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
            edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

            if args.ego:
                batch_size = len(batch.y)
                node_attr = add_EgoIDs(node_attr, edge_index[:, :batch_size])
            node_attr, _ = self.nodes.encoder(node_attr)

            return node_attr, edge_index, edge_attr, y, None

class IBMTransactionsAMLTransactions(torch_frame.data.Dataset):
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
                #'From Bank': torch_frame.categorical,
                #'To Bank': torch_frame.categorical,
                'Payment Currency': torch_frame.categorical,
                'Receiving Currency': torch_frame.categorical,
                'Payment Format': torch_frame.categorical,
                'Timestamp': torch_frame.timestamp,
                'Amount Paid': torch_frame.numerical,
                #'Amount Received': torch_frame.numerical
            }
            self.num_columns = ['Amount Paid']
            self.cat_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
            self.masked_numerical_columns = ['Amount Paid']
            self.masked_categorical_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']

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
        
        def init_encoder(self, channels):
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
                stype.timestamp: TimestampEncoder(),
            }
            self.encoder = StypeWiseFeatureEncoder(
                        out_channels=channels,
                        col_stats=self.col_stats,
                        col_names_dict=self.tensor_frame.col_names_dict,
                        stype_encoder_dict=stype_encoder_dict,
            )

class IBMTransactionsAMLNodes(torch_frame.data.Dataset):
    def __init__(self, num_nodes, ego=False):
        self.num_nodes = num_nodes
        self.df = pd.DataFrame(np.arange(num_nodes+1), columns=['node_id'])
        self.df['node_attr'] = 1

        self.df.reset_index(inplace=True)
        col_to_stype = {
            'node_attr': stype.numerical,
        }
        self.masked_numerical_columns = []
        self.masked_categorical_columns = []
        if ego:
            self.df['ego'] = 1
            col_to_stype['ego'] = stype.numerical
        super().__init__(self.df, col_to_stype)
    
    def init_encoder(self, channels):
        self.encoder = StypeWiseFeatureEncoder(
            channels,
            self.col_stats,
            self.tensor_frame.col_names_dict,
            {stype.numerical: LinearEncoder()}
        )