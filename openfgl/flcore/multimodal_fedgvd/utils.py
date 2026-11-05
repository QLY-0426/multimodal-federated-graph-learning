"""Aligned multimodal graph payloads and local augmentation."""

import hashlib
import json

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def modality_mask(data):
    mask = getattr(data, "modality_mask", None)
    if mask is None:
        mask = torch.ones((data.num_nodes, 2), dtype=torch.bool, device=data.x_text.device)
    return mask


def graph_to_payload(data):
    fields = ("x_text", "x_image", "y", "edge_index", "edge_weight", "modality_mask")
    return {name: getattr(data, name).detach().cpu().clone() for name in fields}


def graph_from_payload(payload, device):
    data = Data(**{key: value.to(device) for key, value in payload.items()})
    data.x = torch.cat([F.normalize(data.x_image, dim=1),
                        F.normalize(data.x_text, dim=1)], dim=1)
    return data


def append_synthetic_data(processed_data, synthetic):
    """Leave the real evaluation graph and masks untouched."""
    original = processed_data["data"]
    n = original.num_nodes
    data = original.clone()
    for name in ("x", "x_text", "x_image", "y"):
        setattr(data, name, torch.cat([getattr(original, name), getattr(synthetic, name)], dim=0))
    data.edge_index = torch.cat([original.edge_index, synthetic.edge_index + n], dim=1)
    real_weight = getattr(original, "edge_weight", None)
    if real_weight is None:
        real_weight = original.x.new_ones(original.num_edges)
    data.edge_weight = torch.cat([real_weight, synthetic.edge_weight])
    data.modality_mask = torch.cat([modality_mask(original), modality_mask(synthetic)])
    data.num_nodes = n + synthetic.num_nodes
    if getattr(data, "node_id", None) is not None:
        data.node_id = torch.cat([original.node_id, original.node_id.new_full((synthetic.num_nodes,), -1)])
    result = {"data": data}
    for name in ("train_mask", "val_mask", "test_mask"):
        extra = torch.full((synthetic.num_nodes,), name == "train_mask",
                           dtype=torch.bool, device=data.x.device)
        result[name] = torch.cat([processed_data[name], extra])
    return result


def cache_signature(processed_data, settings):
    """Include graph, split and compression settings; never test labels."""
    # Increment whenever condensation semantics change; do not mutate caller settings.
    settings = {**settings, "condense_version": 2}
    data = processed_data["data"]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    tensors = [data.x_text, data.x_image, data.edge_index, modality_mask(data)]
    weight = getattr(data, "edge_weight", None)
    if weight is not None:
        tensors.append(weight)
    for name in ("train_mask", "val_mask"):
        tensors.extend([processed_data[name], data.y[processed_data[name]]])
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
