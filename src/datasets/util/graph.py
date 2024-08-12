import numpy as np
import torch
import torch_geometric
from torch_geometric.sampler import NeighborSampler
import torch_frame
from torch_frame import stype

def create_graph(self, col_to_stype, src_column, dst_column):

    # Convert src and dst columns to tensors directly
    src = torch.tensor(self.df[src_column].values, dtype=torch.long)
    dst = torch.tensor(self.df[dst_column].values, dtype=torch.long)
    
    # Create edge index tensor
    edge_index = torch.stack([src, dst], dim=0)
    
    # Create edge attributes (ids)
    ids = torch.arange(len(src), dtype=torch.float)
    
    # Compute number of unique nodes
    num_nodes = len(torch.unique(edge_index))

    # Create node features
    x = torch.arange(num_nodes)

    # Create the full graph
    self.edges = torch.cat([edge_index, ids.unsqueeze(0)], dim=0).t()
    # Create the 'link' column in the DataFrame
    self.df['link'] = self.edges.tolist()

    if 'split' in self.df.columns:
        # Create train graph
        train_mask = self.df['split'] == 0
        train_mask = torch.tensor(train_mask.to_numpy(), dtype=torch.bool)
        train_edge_index = edge_index[:, train_mask]
        train_ids = ids[train_mask]
        self.train_graph = torch_geometric.data.Data(x=x, edge_index=train_edge_index, edge_attr=train_ids)
        self.train_sampler = NeighborSampler(self.train_graph, num_neighbors=self.khop_neighbors)

        # Create val graph
        val_mask = val_mask = self.df['split'].isin([0, 1])
        val_mask = torch.tensor(val_mask.to_numpy(), dtype=torch.bool)
        val_edge_index = edge_index[:, val_mask]
        val_ids = ids[val_mask]
        self.val_graph = torch_geometric.data.Data(x=x, edge_index=val_edge_index, edge_attr=val_ids)
        self.val_sampler = NeighborSampler(self.val_graph, num_neighbors=self.khop_neighbors)

        # Create test graph
        test_edge_index = edge_index
        test_ids = ids
        timestamps = torch.tensor(self.df[self.timestamp_col].values, dtype=torch.long)
        self.test_graph = torch_geometric.data.Data(x=x, edge_index=test_edge_index, edge_attr=test_ids, timestamps=timestamps)
        self.test_sampler = NeighborSampler(self.test_graph, num_neighbors=self.khop_neighbors)
    else:
        print('No split column found. Using the same graph for train, val and test.')
        self.test_graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=ids)
        self.test_sampler = NeighborSampler(self.test_graph, num_neighbors=self.khop_neighbors)
        self.train_graph = self.test_graph
        self.train_sampler = self.test_sampler
        self.val_graph = self.test_graph
        self.val_sampler = self.test_sampler

    # Update col_to_stype
    col_to_stype['link'] = torch_frame.relation
    
    return col_to_stype

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if not hasattr(data, 'timestamps') else data.timestamps.reshape((-1,1))
    #timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def add_ports(self):
        '''Adds port numberings to the edge features'''
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self.test_graph)
        in_ports = ports(self.test_graph.edge_index, adj_list_in)
        out_ports = ports(self.test_graph.edge_index.flipud(), adj_list_out)
        self.df['in_port'] = in_ports
        self.df['out_port'] = out_ports

def add_EgoIDs(x, seed_edge_index):
    device = x.device
    ids = torch.zeros((x.shape[0], 1), device=device)
    nodes = torch.unique(seed_edge_index.contiguous().view(-1)).to(device)
    ids[nodes] = 1 
    x = torch.cat([x, ids], dim=1)
    return x

# def add_EgoIDs(x, seed_edge_index):
#     device = x.device
#     # x is tensor_frame
#     col_names_dict = x.col_names_dict
#     col_names_dict[stype.numerical].append('ego_id')
#     feat_dict = x.feat_dict

#     ids = torch.zeros((len(feat_dict[stype.numerical]), 1), device=device)
#     nodes = torch.unique(seed_edge_index.contiguous().view(-1)).to(device)
#     ids[nodes] = 1
#     feat_dict[stype.numerical] = torch.cat([feat_dict[stype.numerical], ids.unsqueeze(1)], dim=1)
#     x = torch_frame.TensorFrame(feat_dict, col_names_dict, x.y)
#     return x
