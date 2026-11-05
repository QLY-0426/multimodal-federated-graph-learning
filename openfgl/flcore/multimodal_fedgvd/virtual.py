"""Differentiable dual-modal subgraph integrators for the server only.

One shared node index per client has independent text/image features and edges.
Eq. (9) is applied separately in each modality. Virtual labels are explicitly
defined as synthetic class distributions, since the paper leaves them unspecified.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import coalesce


class VirtualGraphBuilder(nn.Module):
    def __init__(self, base, client_slices, num_classes, hidden_dim=128, cross_k=2):
        super().__init__()
        if not isinstance(cross_k, int) or cross_k < 0:
            raise ValueError('virtual cross_k must be a nonnegative integer')
        if hidden_dim < 1 or num_classes < 1:
            raise ValueError('hidden_dim and num_classes must be positive')
        self.cross_k, self.num_classes = cross_k, num_classes
        dims = (base.x_text.size(1), base.x_image.size(1))
        self.queries = nn.ParameterList(nn.Parameter(torch.randn(d)) for d in dims)
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(),
                                               nn.Linear(hidden_dim, d)) for d in dims)
        self.update_graph(base, client_slices)

    def update_graph(self, base, client_slices):
        """Refresh graph membership after late arrivals without resetting parameters."""
        self.base, self.client_slices = base, dict(client_slices)
        n, count = base.num_nodes, len(client_slices)
        if not count or any(s.stop <= s.start for s in client_slices.values()):
            raise ValueError('virtual nodes require nonempty client subgraphs')
        membership = torch.zeros(count, n, dtype=torch.bool, device=base.x_text.device)
        for index, part in enumerate(client_slices.values()):
            membership[index, part] = True
        self.groups = [membership & base.modality_mask[:, m] for m in range(2)]
        self.present = torch.stack([group.any(1) for group in self.groups], dim=1)
        observed = membership & base.modality_mask.any(1)
        labels = F.one_hot(base.y, self.num_classes).to(base.x_text.dtype)
        self.targets = observed.to(base.x_text.dtype) @ labels / observed.sum(1, keepdim=True).clamp_min(1)
        # Modality supervision only uses labels of nodes observed in that modality.
        self.modal_targets = torch.stack([
            group.to(base.x_text.dtype) @ labels
            / group.sum(1, keepdim=True).clamp_min(1)
            for group in self.groups], dim=1)
        self.fixed_edges, self.fixed_weights = [], []
        for m, name in enumerate(('text', 'image')):
            clients, nodes = self.groups[m].nonzero(as_tuple=True)
            stars = torch.stack((nodes, n + clients))
            local = getattr(base, 'edge_index_' + name)
            weights = getattr(base, 'edge_weight_' + name)
            self.fixed_edges.append(torch.cat((local, stars, stars.flip(0)), dim=1))
            self.fixed_weights.append(torch.cat((weights, weights.new_ones(stars.size(1) * 2))))

    def _aggregate(self, column):
        features = (self.base.x_text, self.base.x_image)[column]
        score = F.normalize(features, dim=1) @ F.normalize(self.queries[column], dim=0)
        scores = score.expand(len(self.client_slices), -1).masked_fill(~self.groups[column], -torch.inf)
        scores = torch.where(self.present[:, column, None], scores, torch.zeros_like(scores))
        attention = F.softmax(scores, dim=1) * self.groups[column]
        return self.mlps[column](attention @ features).masked_fill(~self.present[:, column, None], 0)

    def forward(self):
        data = self.base.clone()
        n, count = data.num_nodes, len(self.client_slices)
        virtual = [self._aggregate(m) for m in range(2)]
        data.x_text = torch.cat((data.x_text, virtual[0]))
        data.x_image = torch.cat((data.x_image, virtual[1]))
        data.x = torch.cat((F.normalize(data.x_image, dim=1), F.normalize(data.x_text, dim=1)), dim=1)
        data.y = torch.cat((data.y, data.y.new_full((count,), -1)))
        data.modality_mask = torch.cat((data.modality_mask, self.present))
        data.synthetic_mask = torch.arange(n + count, device=data.y.device) < n
        data.virtual_start = n
        data.virtual_targets, data.virtual_modal_targets = self.targets, self.modal_targets
        data.num_nodes = n + count
        for m, name in enumerate(('text', 'image')):
            # Discrete top-k selection; gradients flow through virtual features.
            # Unit edge weights avoid negative GCN degrees for negative cosine.
            with torch.no_grad():
                normalized = F.normalize(virtual[m], dim=1)
                similarity = normalized @ normalized.t()
                eligible = self.present[:, m, None] & self.present[None, :, m]
                eligible.fill_diagonal_(False)
                similarity.masked_fill_(~eligible, -torch.inf)
                values, neighbors = similarity.topk(min(self.cross_k, max(0, count - 1)), dim=1)
                sources = torch.arange(count, device=data.y.device)[:, None].expand_as(neighbors)
                valid = torch.isfinite(values)
                cross = torch.stack((sources[valid] + n, neighbors[valid] + n))
                edges = torch.cat((self.fixed_edges[m], cross, cross.flip(0)), dim=1)
                weights = torch.cat((self.fixed_weights[m], data.x.new_ones(2 * cross.size(1))))
                edges, weights = coalesce(edges, weights, num_nodes=n + count, reduce='max')
            setattr(data, 'edge_index_' + name, edges)
            setattr(data, 'edge_weight_' + name, weights)
        return data


def integrator_contrastive_loss(data, temperature=.5, mode='paper'):
    """Eq. (13), or explicitly opt-in InfoNCE (includes positive in denominator).

The global anchor is mean-pooled from valid virtual features in each modality.
The literal paper loss can be negative; it is evaluated using logsumexp.
"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('contrastive temperature must be finite and positive')
    if mode not in ('paper', 'infonce'):
        raise ValueError('contrastive mode must be paper or infonce')
    losses = []
    for m, all_features in enumerate((data.x_text, data.x_image)):
        features = all_features[data.virtual_start:][data.modality_mask[data.virtual_start:, m]]
        if len(features) < 2:
            continue
        normalized = F.normalize(features, dim=1)
        global_feature = F.normalize(features.mean(0), dim=0)
        positive = (normalized @ global_feature) / temperature
        negative = normalized @ normalized.t() / temperature
        negative = negative.masked_fill(torch.eye(len(features), device=features.device, dtype=torch.bool), -torch.inf)
        denominator = torch.logsumexp(negative, dim=1)
        if mode == 'infonce':
            denominator = torch.logaddexp(positive, denominator)
        losses.append((denominator - positive).mean())
    return torch.stack(losses).mean() if losses else (data.x_text.sum() + data.x_image.sum()) * 0


def virtual_classification_loss(data, fused_logits, modal_logits, branch_weight=.5):
    start = data.virtual_start
    available = data.modality_mask[start:]
    losses = []
    for m in range(2):
        keep = available[:, m]
        if keep.any():
            losses.append(-(data.virtual_modal_targets[keep, m]
                            * F.log_softmax(modal_logits[start:][keep, m], dim=-1)).sum(-1).mean())
    keep = available.any(1)
    if not keep.any():
        return fused_logits.sum() * 0
    fused = -(data.virtual_targets[keep] * F.log_softmax(fused_logits[start:][keep], dim=-1)).sum(-1).mean()
    return fused + branch_weight * torch.stack(losses).mean()
