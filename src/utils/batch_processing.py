import torch
import numpy as np

from torch_frame import TensorFrame
from torch_frame import stype

from src.primitives import negative_sampling

from torch_geometric.sampler import EdgeSamplerInput


def node_inputs(dataset, batch: TensorFrame, tensor_frame: TensorFrame, mode='train', args=None):

    ids = batch.get_col_feat("node")
    y = batch.y
    khop_source, khop_destination, idx = dataset.sample_neighbors_from_nodes(ids, 'test')
    edge_attr = tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    # make sure ids are at the start of nodes
    #nodes = torch.cat([ids, nodes[~torch.isin(nodes, ids)]])
    #RuntimeError: Tensors must have same number of dimensions: got 2 and 1
    nodes = torch.cat([ids, nodes[~torch.isin(nodes, ids)].unsqueeze(1)])

    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    if args.ego:
        batch_size = len(batch.y)
        node_feats = addEgoIDs(node_feats, edge_index[:, :batch_size])

    return node_feats, edge_index, edge_attr, y

def graph_inputs(dataset, batch: TensorFrame, tensor_frame: TensorFrame, mode='train', args=None):

    edges = batch.y[:,-3:]
    y = batch.y[:, 0].to(torch.long)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, mode)
    edge_attr = tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    local_khop_source = torch.tensor([n_id_map[node.item()] for node in khop_source], dtype=torch.long)
    local_khop_destination = torch.tensor([n_id_map[node.item()] for node in khop_destination], dtype=torch.long)
    edge_index = torch.cat((local_khop_source.unsqueeze(0), local_khop_destination.unsqueeze(0)))

    if args.ego:
        batch_size = len(batch.y)
        node_feats = addEgoIDs(node_feats, edge_index[:, :batch_size])

    return node_feats, edge_index, edge_attr, y

def mcm_inputs(tf: TensorFrame, dataset, mode = 'train', ego=False):
    batch_size = len(tf.y)
    edges = tf.y[:,-3:]
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, mode)

    edge_attr = dataset.tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()


    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    vectorized_map = np.vectorize(lambda x: n_id_map[x])
    khop_combined = torch.cat((khop_source, khop_destination))
    local_khop_combined = torch.LongTensor(vectorized_map(khop_combined.numpy()))
    local_khop_source, local_khop_destination = local_khop_combined.split(khop_source.size(0))
    edge_index = torch.stack((local_khop_source, local_khop_destination))
    if ego:
        node_feats = addEgoIDs(node_feats, edge_index[:, :batch_size])

    target_edge_index = edge_index[:, :batch_size]
    target_edge_attr  = edge_attr[:batch_size]
    return node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr 

def sample_neighbors_b(dataset, edges, mode="train") -> (torch.Tensor, torch.Tensor): # type: ignore
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
    edges = edges.to(dtype=torch.int)
    row = edges[:, 0]
    col = edges[:, 1]
    idx = edges[:, 2] 
    input = EdgeSamplerInput(None, row.to(dtype=torch.long), col.to(dtype=torch.long))
    
    if mode == 'train':
        out = dataset.train_sampler.sample_from_edges(input)
        perm = dataset.train_sampler.edge_permutation 
    elif mode == 'val':
        out = dataset.val_sampler.sample_from_edges(input)
        perm = dataset.val_sampler.edge_permutation 
    elif mode =='test':
        out = dataset.test_sampler.sample_from_edges(input)
        perm = dataset.test_sampler.edge_permutation 
    else:
        raise ValueError("Invalid sampling mode! Valid values: ['train', 'val', 'test']")

    batch = out.batch
    e_batch = torch.tensor([batch[src] for src in out.row])
    e_id = perm[out.edge] if perm is not None else out.edge

    is_new_edge = ~torch.isin(e_id, idx)
    new_edges = e_id[is_new_edge]
    batch = e_batch[is_new_edge]

    if len(new_edges) > 0:
        row = torch.cat([row, dataset.edges[new_edges, 0]])
        col = torch.cat([col, dataset.edges[new_edges, 1]])
        idx = torch.cat([idx, new_edges])
        batch = torch.cat([torch.arange(len(edges)), batch])
    order = batch.argsort(stable=True)
    row = row[order]
    col = col[order]
    idx = idx[order]
    batch = batch[order]

    return row, col, idx, batch

def mcm_inputs_b(tf: TensorFrame, dataset, mode = 'train', ego=False):
    batch_size = len(tf.y)
    edges = tf.y[:,-3:]
    khop_source, khop_destination, idx, batch = sample_neighbors_b(dataset, edges, mode)

    edge_attr = dataset.tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    vectorized_map = np.vectorize(lambda x: n_id_map[x])
    khop_combined = torch.cat((khop_source, khop_destination))
    local_khop_combined = torch.LongTensor(vectorized_map(khop_combined.numpy()))
    local_khop_source, local_khop_destination = local_khop_combined.split(khop_source.size(0))
    edge_index = torch.stack((local_khop_source, local_khop_destination))
    if ego:
        node_feats = addEgoIDs(node_feats, edge_index[:, :batch_size])

    target_idx = torch.bincount(batch, minlength=batch_size)
    target_idx = torch.concat([torch.zeros(1), torch.cumsum(target_idx, dim=0)[:-1]]).to(torch.long)
    return node_feats, edge_index, edge_attr, target_idx, batch

# def convert_to_tensor(batch, input_tensor, cols, hidden):
#     print(input_tensor.shape)
#     # print(batch.shape)
#     # input_tensor = input_tensor.view(input_tensor.shape[0], -1)
#     # print(input_tensor.shape)
#     # order = batch.argsort()
#     # batch = batch[order]
#     # input_tensor = input_tensor[order]
#     output_tensor, mask = to_dense_batch(input_tensor, batch)

#     print(output_tensor.shape)
#     print(mask.shape)
#     # print(mask)
#     # import sys
#     # sys.exit()

#     # print(output_tensor.shape)
#     # print(mask.shape)
#     # print(mask)
#     return output_tensor, mask
#     import sys
#     sys.exit()
    
def lp_inputs(tf: TensorFrame, dataset, num_neg_samples=64, mode='train', ego=False):
    edges = tf.y[:,-3:]
    batch_size = len(edges)
    khop_source, khop_destination, idx = dataset.sample_neighbors(edges, mode)
    
    edge_attr = dataset.tensor_frame.__getitem__(idx)

    nodes = torch.unique(torch.cat([khop_source, khop_destination]))
    num_nodes = nodes.shape[0]
    node_feats = torch.ones(num_nodes).view(-1,num_nodes).t()

    n_id_map = {value.item(): index for index, value in enumerate(nodes)}
    vectorized_map = np.vectorize(lambda x: n_id_map[x])
    khop_combined = torch.cat((khop_source, khop_destination))
    local_khop_combined = torch.LongTensor(vectorized_map(khop_combined.numpy()))
    local_khop_source, local_khop_destination = local_khop_combined.split(khop_source.size(0))
    edge_index = torch.stack((local_khop_source, local_khop_destination))

    if ego:
        node_feats = addEgoIDs(node_feats, edge_index[:, :batch_size])

    neigh_edge_index = edge_index[:, batch_size:]
    neigh_edge_attr  = edge_attr[batch_size:]

    pos_edge_index = edge_index[:, :batch_size]
    pos_edge_attr  = edge_attr[:batch_size]

    # generate/sample negative edges
    target_dict = pos_edge_attr.feat_dict
    for key, value in pos_edge_attr.feat_dict.items():
        attr = []
        # duplicate each row of the tensor by num_neg_samples times repeated values must be contiguous
        for r in value:
            if key == stype.timestamp:
                attr.append(r.repeat(num_neg_samples, 1, 1))
            else:
                attr.append(r.repeat(num_neg_samples, 1))
        target_dict[key] = torch.cat([target_dict[key], torch.cat(attr, dim=0)], dim=0)
    target_edge_attr = TensorFrame(target_dict, pos_edge_attr.col_names_dict)
    
    neg_edge_index = negative_sampling.generate_negative_samples(edge_index.tolist(), pos_edge_index.tolist(), num_neg_samples)
    neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
    target_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    return node_feats, edge_index, edge_attr, neigh_edge_index, neigh_edge_attr, target_edge_index, target_edge_attr


def addEgoIDs(x, seed_edge_index):
    device = x.device
    ids = torch.zeros((x.shape[0], 1), device=device)
    nodes = torch.unique(seed_edge_index.contiguous().view(-1)).to(device)
    ids[nodes] = 1 
    x = torch.cat([x, ids], dim=1)
    return x
