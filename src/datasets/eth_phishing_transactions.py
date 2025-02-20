import os.path

import torch
import torch_frame
from torch_geometric.sampler import EdgeSamplerInput, NodeSamplerInput
from torch_frame import stype
from torch_frame.nn import (
    ProjectionEncoder,
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

class EthereumPhishing():
        def __init__(self, root, mask_type="replace", supervised=False, pretrain: set[PretrainType] = set(), split_type='random', splits=[0.65, 0.15, 0.2], khop_neighbors=[100, 100], ports=False, ego=False, channels=64, use_cutoffs=False):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors
            self.pretrain = pretrain
            self.mask_type = mask_type

            logger.info(f'Creating nodes...')
            self.nodes = EthereumPhishingNodes(os.path.join(root, 'nodes.csv'), split_type=split_type, splits=splits, ego=ego)
            self.nodes.materialize()
            self.nodes.init_encoder(channels)
            logger.info(f'Nodes created.')
            logger.info(f'Creating edges...')
            cutoffs = None
            # if pretrain set not empty
            if use_cutoffs:
                cutoffs = self.nodes.cutoffs
            self.edges = EthereumPhishingTransactions(os.path.join(root, 'edges.csv'), mask_type=mask_type, pretrain=pretrain, split_type=split_type, splits=splits, khop_neighbors=khop_neighbors, ports=ports, cutoffs=cutoffs)
            self.edges.materialize()
            self.edges.init_encoder(channels)
            logger.info(f'Edges created.')
            
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
            input = EdgeSamplerInput(None, torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long))
            
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
            # print(nodes)
            # neighbors = self.edges.train_graph.edge_index[1][self.edges.train_graph.edge_index[0] == nodes[0]]
            # print(f"Neighbors of node {nodes[0]}: {neighbors.tolist()}")

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
            # print(out)
            # print(out.edge)

            e_id = perm[out.edge] if perm is not None else out.edge
            row = self.edges.edges[e_id, 0]
            col = self.edges.edges[e_id, 1]
            idx = e_id

            return row, col, idx
    
        def get_graph_inputs(self, batch: torch_frame.TensorFrame, mode='train', args=None):

            y, ids = batch.y[:, 0], batch.y[:, 1]
            khop_source, khop_destination, idx = self.sample_neighbors_from_nodes(ids, mode)

            nodes = torch.unique(torch.cat([khop_source, khop_destination]))
            if len(ids.shape) > 1:
                ids = ids.squeeze()
            nodes = torch.cat([ids, nodes[~torch.isin(nodes, ids)]]).type(torch.long)

            n_id_map = {value.item(): index for index, value in enumerate(nodes)}
            local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
            local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
            edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

            edge_attr = self.edges.tensor_frame.__getitem__(idx)
            #edge_attr, _ = self.edges.encoder(edge_attr)

            node_attr = self.nodes.tensor_frame.__getitem__(nodes)
            if args.ego:
                batch_size = len(batch.y)
                node_attr = add_EgoIDs(node_attr, edge_index[:, :batch_size])
            #node_attr, _ = self.nodes.encoder(node_attr)

            return node_attr, edge_index, edge_attr, y, None

        def get_mcm_inputs(self, batch: torch_frame.TensorFrame, mode='train', args=None):

            y, edges = batch.y[:, :-3], batch.y[:,-3:]
            khop_source, khop_destination, idx = self.sample_neighbors(edges, mode)
            edge_attr = self.edges.tensor_frame.__getitem__(idx)
            #edge_attr, _ = self.edges.encoder(edge_attr)

            nodes = torch.unique(torch.cat([khop_source, khop_destination])).type(torch.long)

            node_attr = self.nodes.tensor_frame.__getitem__(nodes)

            n_id_map = {value.item(): index for index, value in enumerate(nodes)}
            local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
            local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
            edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

            if args.ego:
                batch_size = len(batch.y)
                node_attr = add_EgoIDs(node_attr, edge_index[:, :batch_size])
            #node_attr, _ = self.nodes.encoder(node_attr)

            return node_attr, edge_index, edge_attr, y, None
        
        # def get_mcm_inputs(self, batch: torch_frame.TensorFrame, mode='train', args=None):
        #     y, ids = batch.y[:, 0], batch.y[:, 1]
        #     khop_source, khop_destination, idx = self.sample_neighbors_from_nodes(ids, mode)

        #     nodes = torch.unique(torch.cat([khop_source, khop_destination]))
        #     nodes = torch.cat([ids, nodes[~torch.isin(nodes, ids)]]).type(torch.long)

        #     n_id_map = {value.item(): index for index, value in enumerate(nodes)}
        #     local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
        #     local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
        #     edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

        #     edge_attr = self.edges.tensor_frame.__getitem__(idx)
        #     y_mcm = edge_attr.y
        #     edge_attr, _ = self.edges.encoder(edge_attr)

        #     node_attr = self.nodes.tensor_frame.__getitem__(nodes)
        #     if args.ego:
        #         batch_size = len(batch.y)
        #         node_attr = add_EgoIDs(node_attr, edge_index[:, :batch_size])
        #     node_attr, _ = self.nodes.encoder(node_attr)

        #     return node_attr, edge_index, edge_attr, y, y_mcm, None

class EthereumPhishingTransactions(torch_frame.data.Dataset):
        def __init__(self, root, mask_type="replace", pretrain: set[PretrainType] = set(), split_type='temporal', splits=[0.6, 0.2, 0.2], khop_neighbors=[100, 100], ports=False, cutoffs=None):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            self.khop_neighbors = khop_neighbors
            self.timestamp_col = 'block_timestamp'
            self.pretrain = pretrain
            self.mask_type = mask_type
            
            self.df = pd.read_csv(root, header=0)
            print(self.df.describe())
            self.df.reset_index(inplace=True)
            
            col_to_stype = {
                'nonce': torch_frame.numerical,
                'value': torch_frame.numerical,
                'gas': torch_frame.numerical,
                'gas_price': torch_frame.numerical,
                'block_timestamp': torch_frame.timestamp,
            }
            self.masked_numerical_columns = ['nonce', 'value', 'gas', 'gas_price']
            self.masked_categorical_columns = []
            if cutoffs is not None:
                self.df = apply_split(self.df, 'cutoff', cutoffs, self.timestamp_col)
            else:
                self.df = apply_split(self.df, self.split_type, self.splits, self.timestamp_col)
            
            logger.info(f'Creating graph...')
            start = time.time()
            col_to_stype = create_graph(self, col_to_stype, "from_address", "to_address")
            logger.info(f'Graph created in {time.time()-start} seconds.')

            if ports:
                logger.info(f'Adding ports...')
                start = time.time()
                add_ports(self)
                col_to_stype['in_port'] = stype.numerical
                col_to_stype['out_port'] = stype.numerical
                logger.info(f'Ports added in {time.time()-start:.2f} seconds.')

            if PretrainType.MASK in pretrain:
                self.maskable_columns = self.masked_numerical_columns + self.masked_categorical_columns
                mask_col = create_mask(self, self.maskable_columns)
                self.df['maskable_column'] = mask_col
                original_values = self.df.apply(lambda row: row[row['maskable_column']], axis=1)
                self.df['mask'] = [list(x) for x in zip(original_values, mask_col)]
                col_to_stype['mask'] = torch_frame.mask
            else:
                self.maskable_columns = None
            
            if PretrainType.MASK in pretrain or PretrainType.LINK_PRED in pretrain:
                col_to_stype = set_target_col(self, pretrain, col_to_stype, None)
            else:
                self.target_col = None
                del col_to_stype['link']
            #super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col, maskable_columns= self.maskable_columns)
            super().__init__(self.df, col_to_stype, target_col=self.target_col, split_col='split', maskable_columns= self.maskable_columns)

        def init_encoder(self, channels):
            self.encoder = StypeWiseFeatureEncoder(
                channels,
                self.col_stats,
                self.tensor_frame.col_names_dict,
                {stype.numerical: LinearEncoder(), stype.timestamp: TimestampEncoder()}
            )

class EthereumPhishingNodes(torch_frame.data.Dataset):
    def __init__(self, root, split_type='temporal', splits=[0.65, 0.15, 0.2], ego=False):
        self.root = root
        self.split_type = split_type
        self.splits = splits
        self.df = pd.read_csv(root)
        self.df['target'] = self.df.apply(lambda row: [row['label'], row['node']], axis=1)
        self.target_col = 'target'

        print(self.df.head())
        self.timestamp_col = 'first_transaction'
        self.cutoffs = self.get_split_timestamps()
        self.df = apply_split(self.df, 'cutoff', self.cutoffs, self.timestamp_col)

        self.df.reset_index(inplace=True)
        col_to_stype = {
            'target': stype.relation,
        }
        if ego:
            self.df['EgoID'] = 0
            col_to_stype['EgoID'] = stype.relation
        else:
            self.df['node_attr'] = 1
            col_to_stype['node_attr'] = stype.relation
        self.masked_numerical_columns = []
        self.masked_categorical_columns = []
        super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col)
    
    def init_encoder(self, channels):
        self.encoder = StypeWiseFeatureEncoder(
            channels,
            self.col_stats,
            self.tensor_frame.col_names_dict,
            {stype.numerical: LinearEncoder(), stype.relation: ProjectionEncoder()}
        )
    
    def get_split_timestamps(self):
        # Sort the DataFrame by the timestamp column
        sorted_df = self.df.sort_values(by=self.timestamp_col)
        print(sorted_df.head())
        
        # Calculate the number of rows for each split
        train_size = int(len(sorted_df) * self.splits[0])
        validation_size = train_size + int(len(sorted_df) * self.splits[1])
        
        # Get the train and validation cutoff timestamps
        train_cutoff = sorted_df[self.timestamp_col].iloc[train_size - 1]
        validation_cutoff = sorted_df[self.timestamp_col].iloc[validation_size - 1]
        
        print(f"Train cutoff: {train_cutoff}, Validation cutoff: {validation_cutoff}")
      
        return train_cutoff, validation_cutoff