import gzip
import itertools
import os
import tokenize

import pandas as pd
import numpy as np
import requests
import torch
from transformers import BertModel, BertTokenizer

import torch_frame
import torch_geometric
from icecream import ic
from torch_geometric.loader import NeighborSampler
from torch_geometric.sampler import EdgeSamplerInput

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig


class AmazonFashionDataset(torch_frame.data.Dataset):
    """
    A class to handle the Amazon Fashion dataset with specific functionalities for data preprocessing,
    dataset splitting, and pretraining configurations for graph-based machine learning tasks.

    The dataset has:
    -reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    -verified - Verified status of the user
    -asin - ID of the product, e.g. 0000013714
    -vote - helpful votes of the review
    -style - a disctionary of the product metadata, e.g., "Format" is "Hardcover"
    -reviewText - text of the review
    -overall - rating of the product
    -summary - summary of the review
    -unixReviewTime - time of the review (unix time)

    Attributes:
        root (str): Root directory where the dataset is located.
        pretrain (str): Type of pretraining configuration ('mask' for masking, 'link' for link prediction).
        split_type (str): Type of data splitting method ('random', 'temporal', 'temporal_balanced').
        splits (list): Proportions for train, validation, and test split.
    """

    def __init__(
            self, 
            root=None, 
            pretrain='', 
            split_type='random', 
            splits=[0.8, 0.1, 0.1],
            khop_neighbors=[100, 100],
            text_stype: torch_frame.stype = torch_frame.text_embedded,
            col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
            | TextEmbedderConfig | None = None,
            col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
            | TextTokenizerConfig | None = None,
            nrows=None
            ):
        # Set the root to None to install/pre-process the dataset
        if root is None:
            self.retrieve_dataset()
            root = "datasets/AMAZON_FASHION.csv"

        self.root = root
        self.split_type = split_type
        self.splits = splits
        self.pretrain = pretrain
        self.khop_neighbors = khop_neighbors

        if not text_stype.is_text_stype:
                raise ValueError(f"`text_stype` should be a text stype, "
                             f"got {text_stype}.")

        names = [
            'overall', 'verified', 'reviewerID', 'asin', 'reviewText', 'summary',
            'unixReviewTime', 'vote',
        ]
        dtypes = {
            'overall': 'float32',
            'verified': 'bool',
            'reviewerID': 'category',
            'asin': 'category',
            # 'reviewerName': 'object',
            'reviewText': 'object',
            'summary': 'object',
            'unixReviewTime': 'int64',
            'vote': 'float64',
            # 'style': 'object'
        }

        # self.df = pd.read_csv(root, dtype=dtypes, names=names, header=0)
        self.df = pd.read_csv(root, header=0, nrows=nrows)
        # self.df = self.df.dropna()
        # self.df = self.df.reset_index(drop=True)
        # self.df['unixReviewTime'] = pd.to_datetime(self.df['unixReviewTime'], format='%m %d, %Y')

        col_to_stype = {
            'overall': torch_frame.numerical,  # Numerical rating
            'verified': torch_frame.categorical,  # Binary data treated as categorical
            'reviewerID': torch_frame.categorical,  # Categorical, used for IDs
            'asin': torch_frame.categorical,  # Categorical unique identifiers for products
            # 'reviewerName': torch_frame.text_embedded,  # For names, using embedded text if names are used in model
            'reviewText': text_stype,
            'summary': text_stype,
            'unixReviewTime': torch_frame.timestamp,  # Unix timestamps
            'vote': torch_frame.numerical,  # Numerical votes count
            # 'style': torch_frame.multicategorical  # If 'style' contains multiple categorical data or key-value pairs
        }

        if self.split_type == 'temporal':
            self.temporal_balanced_split()
        elif self.split_type == 'temporal_balanced':
            self.temporal_balanced_split()
        else:
            self.random_split()

        if 'mask' in pretrain:
            maskable_columns = ['overall', 'verified', 'vote',
                                'summary']
            self.df['mask'] = None
            self.df = self.df.apply(self.mask_column, args=(maskable_columns,), axis=1)
            col_to_stype['mask'] = torch_frame.mask

        self.sampler = None
        if 'lp' in pretrain:  # initialize training graph and neighbor sampler
            # Convert strings to ints
            unique_ids = pd.concat([self.df['reviewerID'], self.df['asin']]).unique()
            self.id_map = {original_id: i for i, original_id in enumerate(unique_ids)}
            # Now use this map to update 'From ID' and 'To ID' in the DataFrame to use these new IDs
            self.df['reviewerID'] = self.df['reviewerID'].map(self.id_map)
            self.df['asin'] = self.df['asin'].map(self.id_map)

            self.df['link'] = self.df[['reviewerID', 'asin']].apply(list, axis=1)
            col_to_stype['link'] = torch_frame.relation

            def append_index_to_link(row):
                row['link'].append(float(row.name))
                return row

            self.df = self.df.apply(append_index_to_link, axis=1)

            # get number of uique ids in the dataset
            num_nodes = len(set(self.df['reviewerID'].to_list() + self.df['asin'].to_list()))

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

        if pretrain == 'lp':
            self.target_col = 'link'
        elif pretrain == 'mask':
            self.target_col = 'mask'
        elif pretrain == 'lp+mask' or pretrain == 'mask+lp':
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
        else:
            col_to_stype['overall'] = torch_frame.numerical
            self.target_col = 'overall'

        super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col,
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                         col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg)

    def random_split(self):
        self.df['split'] = torch_frame.utils.generate_random_split(length=len(self.df), seed=0, train_ratio=self.splits[0], val_ratio=self.splits[1])


    def temporal_split(self):
        assert 'unixReviewTime' in self.df.columns, \
            'split is only available for datasets with a "unixReviewTime" column'
        self.df = self.df.sort_values(by='unixReviewTime')
        train_size = int(self.df.shape[0] * 0.3)
        validation_size = int(self.df.shape[0] * 0.1)
        test_size = self.df.shape[0] - train_size - validation_size

        # add split column, use 0 for train, 1 for validation, 2 for test
        self.df['split'] = [0] * train_size + [1] * validation_size + [2] * test_size

    def temporal_balanced_split(self):
        assert 'unixReviewTime' in self.df.columns, \
            'split is only available for datasets with a "reviewTime" column'
        self.df['unixReviewTime'] = self.df['unixReviewTime'] - self.df['unixReviewTime'].min()

        timestamps = torch.Tensor(self.df['unixReviewTime'].to_numpy())
        n_days = int(timestamps.max() / (3600 * 24) + 1)

        daily_inds, daily_reviews = [], []  # irs = illicit ratios, inds = indices, reviews = reviews
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_inds.append(day_inds)
            daily_reviews.append(day_inds.shape[0])

        split_per = self.splits
        daily_totals = np.array(daily_reviews)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i, j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v / split_totals_sum for v in split_totals]
                split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
                score = max(split_error)
                split_scores[(i, j)] = score

        i, j = min(split_scores, key=split_scores.get)
        # split contains a list for each split (train, validation and test) and each list contains the days that are
        # part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]

        # Now, we separate the reviews based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                if daily_inds[day].numel() > 0:
                    # print(daily_inds[day])
                    temp_tensor = daily_inds[day].unsqueeze(0)
                    split_inds[i].extend(temp_tensor)  # split_inds contains a list for each split (tr, val, te) which
                    # contains the indices of each day separately

        # print(f"Train indices: {len(split_inds[0])}")  # Debug print
        # print(f"Validation indices: {len(split_inds[1])}")  # Debug print
        # print(f"Test indices: {len(split_inds[2])}")  # Debug print
        # for i, tensor in enumerate(split_inds[1][:30]):
        #     print(f"Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}, numel={tensor.numel()}")

        # add a new split column to df
        self.df['split'] = 0

        # Set values for val_inds and te_inds
        self.df.loc[torch.cat(split_inds[1]), 'split'] = 1
        self.df.loc[torch.cat(split_inds[2]), 'split'] = 2

    # Randomly mask a column of each row and store original value and max index
    def mask_column(self, row, maskable_cols):
        col_to_mask = np.random.choice(maskable_cols)  # Choose a column randomly
        original_value = row[col_to_mask]
        row['mask'] = [original_value, col_to_mask]  # Store original value and max index in 'mask' column

        # row[col_to_mask] = np.nan
        # hack to escape nan error in torch_frame
        if col_to_mask in ['overall', 'vote']:
            row[col_to_mask] = -1
        else:
            row[col_to_mask] = '[MASK]'
        return row

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
            assert self.edges[v][2] == v  # sanity check
            if v not in edge_set:
                row.append(self.edges[v][0])
                col.append(self.edges[v][1])
                idx.append(v)
            # else:
            #     #ic(v)
            #     pass
        khop_row = torch.tensor(row, dtype=torch.long)
        khop_col = torch.tensor(col, dtype=torch.long)
        # else:
        #    idx = torch.hstack((torch.tensor(idx, dtype=torch.long), out.edge))
        #    khop_row = torch.hstack((torch.tensor(row, dtype=torch.long), out.row))
        #    khop_col = torch.hstack((torch.tensor(col, dtype=torch.long), out.col))

        # ic(len(khop_row), len(khop_col), len(idx))
        # ic(self.df.loc[out.edge].iloc[0])
        return khop_row, khop_col, idx

    def embed_text(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def retrieve_dataset(self):
        url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz"
        output_directory = "datasets/"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        file_name = url.split('/')[-1]
        download_path = os.path.join(output_directory, file_name)

        response = requests.get(url, stream=True, verify=False)
        if response.status_code == 200:
            with open(download_path, 'wb') as f:
                f.write(response.raw.read())
            print("Download successful.")

            if file_name.endswith('.gz'):
                output_file_name = file_name[:-3]
                output_path = os.path.join(output_directory, output_file_name)

                with gzip.open(download_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                print("Decompression successful.")

                os.remove(download_path)  # Remove the compressed file
            else:
                raise Exception("The file downloaded is not gzip.")
        else:
            raise Exception("Failed!!")

        csv_output_path = output_path.replace('.json', '.csv')
        print(csv_output_path)

        print("Converting to CSV...")
        df = pd.read_json(output_path, lines=True)

        # only unixreviewtime will be used
        df = df.drop(['reviewTime', 'style', 'image', 'reviewerName'], axis=1)

        df.to_csv(csv_output_path, index=False)
        print("Successful")