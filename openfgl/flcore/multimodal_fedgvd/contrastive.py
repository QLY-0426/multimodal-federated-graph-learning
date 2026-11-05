"""Construct aligned modality graphs without introducing coordinator nodes."""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def _validate(cross_k, similarity_threshold):
    if not isinstance(cross_k, int) or cross_k < 0:
        raise ValueError('cross_k must be a nonnegative integer')
    if not -1 <= similarity_threshold <= 1:
        raise ValueError('similarity_threshold must be in [-1, 1]')


@torch.no_grad()
def build_modal_edges(features, edge_index, client_ids, available, cross_k=2,
                      similarity_threshold=0.0, edge_weight=None):
    """Retain available local edges and add symmetric cross-client cosine edges.

    Similarity storage is at most 1024 by N; the returned graph has at most
    twice N * cross_k added edges. Duplicate edges retain their maximum weight.
    """
    _validate(cross_k, similarity_threshold)
    n = features.size(0)
    available = available.to(device=features.device, dtype=torch.bool)
    client_ids = client_ids.to(features.device)
    edge_index = edge_index.to(device=features.device, dtype=torch.long)
    weights = (features.new_ones(edge_index.size(1)) if edge_weight is None
               else edge_weight.to(device=features.device, dtype=features.dtype))
    keep = available[edge_index[0]] & available[edge_index[1]]
    edges, weights = edge_index[:, keep], weights[keep]
    additions, added_weights = [edges], [weights]
    if cross_k and n:
        normalized = F.normalize(features, dim=1)
        k = min(cross_k, n)
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            similarity = normalized[start:stop] @ normalized.t()
            eligible = (available[start:stop, None] & available[None, :]
                        & (client_ids[start:stop, None] != client_ids[None, :]))
            similarity.masked_fill_(~eligible, -torch.inf)
            values, neighbors = similarity.topk(k, dim=1)
            valid = torch.isfinite(values) & (values >= similarity_threshold)
            sources = torch.arange(start, stop, device=features.device)[:, None].expand_as(neighbors)
            additions.append(torch.stack((sources[valid], neighbors[valid])))
            added_weights.append(values[valid].clamp_min(1e-8))
    edges = torch.cat(additions, dim=1)
    weights = torch.cat(added_weights)
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    weights = torch.cat((weights, weights))
    return coalesce(edges, weights, num_nodes=n, reduce='max')


def get_coordinated_data(graphs, cross_k=2, similarity_threshold=0.0):
    """Concatenate sorted clients and construct two graphs on identical nodes.

    ``modality_mask`` uses [text, image] columns. Compatibility ``x`` uses
    concatenated, separately normalized [image, text] features.
    """
    _validate(cross_k, similarity_threshold)
    if not graphs:
        raise ValueError('graphs must contain at least one client')
    texts, images, labels, masks, edges, weights, ids = [], [], [], [], [], [], []
    client_slices, offset = {}, 0
    for client_id in sorted(graphs):
        graph = graphs[client_id]
        text, image = graph.x_text, graph.x_image
        n = text.size(0)
        if image.size(0) != n:
            raise ValueError('text and image features must have the same node count')
        mask = getattr(graph, 'modality_mask', None)
        if mask is None:
            mask = torch.ones((n, 2), dtype=torch.bool, device=text.device)
        if tuple(mask.shape) != (n, 2):
            raise ValueError('modality_mask must have shape [num_nodes, 2]')
        local_edges = graph.edge_index.to(device=text.device, dtype=torch.long)
        local_weights = getattr(graph, 'edge_weight', None)
        if local_weights is None:
            local_weights = text.new_ones(local_edges.size(1))
        texts.append(text)
        images.append(image)
        labels.append(graph.y)
        masks.append(mask.to(device=text.device, dtype=torch.bool))
        edges.append(local_edges + offset)
        weights.append(local_weights.to(device=text.device, dtype=text.dtype))
        ids.append(torch.full((n,), client_id, device=text.device, dtype=torch.long))
        client_slices[client_id] = slice(offset, offset + n)
        offset += n
    text, image = torch.cat(texts), torch.cat(images)
    mask, client_ids = torch.cat(masks), torch.cat(ids)
    local_edges, local_weights = torch.cat(edges, dim=1), torch.cat(weights)
    data = Data(x=torch.cat((F.normalize(image, dim=1), F.normalize(text, dim=1)), dim=1),
                x_text=text, x_image=image, y=torch.cat(labels), modality_mask=mask,
                edge_index=local_edges, edge_weight=local_weights)
    for column, (modality, features) in enumerate((('text', text), ('image', image))):
        modal_edges, modal_weights = build_modal_edges(
            features, local_edges, client_ids, mask[:, column], cross_k,
            similarity_threshold, local_weights)
        setattr(data, 'edge_index_' + modality, modal_edges)
        setattr(data, 'edge_weight_' + modality, modal_weights)
    return data, client_slices
