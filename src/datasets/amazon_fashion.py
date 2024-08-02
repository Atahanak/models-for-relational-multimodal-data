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

from .util.mask import PretrainType, set_target_col, apply_mask
from .util.graph import create_graph, add_ports
from src.datasets.util.split import apply_split

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
            split_type='random', 
            splits=[0.8, 0.1, 0.1],
            khop_neighbors=[100, 100],
            text_stype: torch_frame.stype = torch_frame.text_embedded,
            col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
            | TextEmbedderConfig | None = None,
            col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
            | TextTokenizerConfig | None = None,
            nrows=None,
            mask_type="replace",
            pretrain=True,
            ports=False
            ):
        if pretrain is None:
            pretrain = {PretrainType.MASK, PretrainType.LINK_PRED}
        # Set the root to None to install/pre-process the dataset
        if root is None:
            self.retrieve_dataset()
            root = ".cache/datasets/AMAZON_FASHION.csv"

        self.root = root
        self.split_type = split_type
        self.splits = splits
        self.khop_neighbors = khop_neighbors

        if not text_stype.is_text_stype:
                raise ValueError(f"`text_stype` should be a text stype, "
                             f"got {text_stype}.")

        names = [
            'overall', 
            'verified', 
            'reviewerID', 
            'asin', 
            'reviewText', 
            'summary',
            'unixReviewTime', 
            'vote',
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
        num_columns = ["vote"]
        cat_columns = ["verified"]

        self.df = apply_split(
            df=self.df, 
            split_type=self.split_type, 
            splits=self.splits,
            timestamp_col='unixReviewTime')
        
        if pretrain:
            # Create mask vector
            mask = create_mask(self.root, self.df, num_columns + cat_columns, masked_dir)
            self.df["maskable_column"] = mask
            # Prepare for graph generation
            # Convert strings to ints
            unique_ids = pd.concat([self.df['reviewerID'], self.df['asin']]).unique()
            self.id_map = {original_id: i for i, original_id in enumerate(unique_ids)}
            # Now use this map to update 'From ID' and 'To ID' in the DataFrame to use these new IDs
            self.df['reviewerID'] = self.df['reviewerID'].map(self.id_map)
            self.df['asin'] = self.df['asin'].map(self.id_map)
            col_to_stype = create_graph(self, col_to_stype, "reviewerID", "asin")

            col_to_stype = apply_mask(self, cat_columns, num_columns, col_to_stype, mask_type)
            # Remove columns that are not needed
            self.df = self.df.drop('maskable_column', axis=1)

        # Define target column to predict
        print(pretrain)
        col_to_stype = set_target_col(self, pretrain, col_to_stype, "overall")

        super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col,
                         col_to_text_embedder_cfg=col_to_text_embedder_cfg,
                         col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg)

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
        output_directory = ".cache/datasets/"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

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
