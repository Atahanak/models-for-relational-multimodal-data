import torch
import numpy as np

from torch_frame import TensorFrame
from torch_frame import stype

from src.primitives import negative_sampling

def mcm_inputs(tf: TensorFrame, dataset, mode = 'train'):
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

    edge_index = edge_index
    edge_attr  = edge_attr
    target_edge_index = edge_index[:, :batch_size]
    target_edge_attr  = edge_attr[:batch_size]
    return node_feats, edge_index, edge_attr, target_edge_index, target_edge_attr  
    
def lp_inputs(tf: TensorFrame, dataset, num_neg_samples=64, mode='train'):
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