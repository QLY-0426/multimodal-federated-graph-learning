import torch
import torch.nn.functional as F

from openfgl.flcore.fedtad._utils import construct_graph, cal_topo_emb


def construct_modal_graph(features, modality_dims, topk):
    """Optional modal view of the original FedTAD graph, preserving raw scale."""
    normalized = F.normalize(features, p=2, dim=1)
    graph = construct_graph(features, normalized @ normalized.t(), k=topk)
    graph.x_image, graph.x_text = features.split((modality_dims[1], modality_dims[0]), dim=1)
    return graph


def class_reliability(data, train_mask, num_classes, max_walk_length):
    """Original FedTAD CKR calculation on the loader's multimodal x features."""
    ckr = torch.zeros(num_classes).to(data.x.device)
    graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=data.num_nodes,
                             max_walk_length=max_walk_length).to(data.x.device)
    ft_emb = torch.cat((data.x, graph_emb), dim=1)
    # flatten preserves iteration order and also supports one training node.
    for train_i in train_mask.nonzero().flatten():
        neighbor = data.edge_index[1, :][data.edge_index[0, :] == train_i]
        node_all = 0
        for neighbor_j in neighbor:
            node_all += torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
        node_all += 1
        node_all /= neighbor.shape[0] + 1
        ckr[data.y[train_i]] += node_all
    return ckr / ckr.sum(0)
