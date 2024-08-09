import torch
import numpy as np

from torch_frame import TensorFrame
from torch_frame import stype

from src.primitives import negative_sampling

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
