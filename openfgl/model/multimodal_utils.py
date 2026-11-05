"""Independent modality branches and detached entropy-based reliability fusion."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def reliability_fusion(text_logits, image_logits, modality_mask=None, temperature=0.5):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    logits = torch.stack((text_logits, image_logits), dim=1)
    available = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device) if modality_mask is None else modality_mask.to(device=logits.device, dtype=torch.bool)
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1) / max(math.log(logits.size(-1)), 1e-12)
        scores = (-entropy / temperature).masked_fill(~available, -torch.inf)
        scores = torch.where(available.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = F.softmax(scores, dim=-1) * available
    fused = (logits.masked_fill(~available.unsqueeze(-1), 0) * weights.unsqueeze(-1)).sum(1)
    return fused, weights


def multimodal_supervised_loss(modal_logits, fused_logits, labels, mask, modality_mask=None, branch_weight=0.5):
    available = torch.ones(modal_logits.shape[:2], dtype=torch.bool, device=modal_logits.device) if modality_mask is None else modality_mask.to(device=modal_logits.device, dtype=torch.bool)
    selected = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    selected[mask] = True
    selected = selected & available.any(1)
    zero = (modal_logits.sum() + fused_logits.sum()) * 0
    if not selected.any():
        return zero
    fused_loss = F.cross_entropy(fused_logits[selected], labels[selected])
    branch_losses = [F.cross_entropy(modal_logits[selected & available[:, i], i], labels[selected & available[:, i]]) for i in range(2) if (selected & available[:, i]).any()]
    return fused_loss + branch_weight * torch.stack(branch_losses).mean()


class _Branch(nn.Module):
    def __init__(self, kind, input_dim, hid_dim, output_dim, num_layers, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATConv, SGConv, SAGEConv
        factories = {'gcn': GCNConv, 'gat': GATConv, 'sgc': SGConv, 'graphsage': SAGEConv, 'mlp': nn.Linear}
        dims = [input_dim] + [hid_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(factories[kind](a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.kind, self.dropout = kind, dropout

    def forward(self, x, edge_index, edge_weight, present):
        for layer in self.layers[:-1]:
            x = self._apply_layer(layer, x, edge_index, edge_weight)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
            x = x.masked_fill(~present[:, None], 0)
        logits = self._apply_layer(self.layers[-1], x, edge_index, edge_weight)
        return x.masked_fill(~present[:, None], 0), logits.masked_fill(~present[:, None], 0)

    def _apply_layer(self, layer, x, edge_index, edge_weight):
        if self.kind == 'mlp':
            return layer(x)
        if self.kind in ('gcn', 'sgc'):
            return layer(x, edge_index, edge_weight=edge_weight)
        return layer(x, edge_index)


class DualBranchModel(nn.Module):
    def __init__(self, kind, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5, fusion_temperature=0.5):
        super().__init__()
        if isinstance(input_dim, int):
            if input_dim <= 0 or input_dim % 2:
                raise ValueError('integer input_dim must be positive and even')
            input_dim = (input_dim // 2, input_dim // 2)
        if len(input_dim) != 2 or min(input_dim) <= 0 or num_layers < 1:
            raise ValueError('two positive modality dimensions and num_layers >= 1 required')
        if not math.isfinite(fusion_temperature) or fusion_temperature <= 0:
            raise ValueError('fusion_temperature must be finite and positive')
        self.fusion_temperature = fusion_temperature
        self.fusion_mode = 'reliability'
        self.input_mode = 'modalities'
        self.modality_dims = tuple(input_dim)
        self.text_branch = _Branch(kind, input_dim[0], hid_dim, output_dim, num_layers, dropout)
        self.image_branch = _Branch(kind, input_dim[1], hid_dim, output_dim, num_layers, dropout)

    def forward_modalities(self, data):
        if self.input_mode == 'concatenated':
            # Match the dataset's x layout and FedTAD's raw generated features.
            image, text = data.x.split((self.modality_dims[1], self.modality_dims[0]), dim=1)
            features = (text, image)
        else:
            features = (data.x_text, data.x_image)
        available = getattr(data, 'modality_mask', None)
        if available is None:
            available = torch.ones((features[0].size(0), 2), dtype=torch.bool, device=features[0].device)
        else:
            available = available.to(device=features[0].device, dtype=torch.bool)
        embeddings, logits = [], []
        for i, (name, branch) in enumerate((('text', self.text_branch), ('image', self.image_branch))):
            present = available[:, i]
            x = features[i].masked_fill(~present[:, None], 0)
            if self.input_mode == 'modalities':
                x = F.normalize(x, p=2, dim=-1)
            edges = getattr(data, 'edge_index_' + name, None)
            edges = data.edge_index if edges is None else edges
            weights = getattr(data, 'edge_weight_' + name, None)
            if weights is None:
                weights = getattr(data, 'edge_weight', None)
                # Shared weights apply only to the shared topology.
                if getattr(data, 'edge_index_' + name, None) is not None:
                    weights = None
            keep = present[edges[0]] & present[edges[1]]
            embedding, prediction = branch(x, edges[:, keep], weights[keep] if weights is not None else None, present)
            embeddings.append(embedding)
            logits.append(prediction)
        if self.fusion_mode == 'mean':
            weights = available.to(logits[0].dtype)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
            fused = (torch.stack(logits, dim=1) * weights.unsqueeze(-1)).sum(1)
        else:
            fused, weights = reliability_fusion(*logits, modality_mask=available, temperature=self.fusion_temperature)
        return torch.cat(embeddings, dim=-1), fused, torch.stack(logits, dim=1), weights

    def forward(self, data):
        embedding, logits, _, _ = self.forward_modalities(data)
        return embedding, logits
