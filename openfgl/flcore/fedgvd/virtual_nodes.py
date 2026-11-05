"""FedGVD section 3.2, equations 9--14.

Explicit choices where the paper is underspecified: shared query/MLP, mean
global representation, condensed label distributions as virtual soft targets.
Top-k is directed: each integrator receives from its k most similar peers.
Discrete neighbors are rebuilt each forward; features retain gradients.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data


class SubgraphIntegrators(nn.Module):
    def __init__(self, datalist, hidden_dim, top_k=3, temperature=1.0):
        super().__init__()
        if not datalist or any(g.x.size(0) == 0 for g in datalist):
            raise ValueError('Each participating client must supply a nonempty condensed graph')
        if top_k < 0 or int(top_k) != top_k or temperature <= 0:
            raise ValueError('top_k must be a nonnegative integer and temperature positive')
        batch = Batch.from_data_list(datalist)
        self.num_graphs = len(datalist)
        self.num_original = batch.x.size(0)
        self.top_k = min(int(top_k), self.num_graphs - 1)
        self.temperature = temperature
        self.register_buffer('original_x', batch.x.detach().clone())
        self.register_buffer('labels', batch.y.long().view(-1).detach().clone())
        self.register_buffer('owners', batch.batch.clone())
        original_ids = torch.arange(self.num_original, device=batch.x.device)
        virtual_ids = self.num_original + batch.batch
        cross = torch.stack([original_ids, virtual_ids])
        self.register_buffer('base_edges', torch.cat([batch.edge_index, cross, cross.flip(0)], dim=1))
        dim = batch.x.size(1)
        self.query = nn.Parameter(torch.empty(dim, device=batch.x.device))
        nn.init.normal_(self.query, std=dim ** -0.5)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)).to(batch.x.device)

    def features(self):
        pooled = []
        for cid in range(self.num_graphs):
            x = self.original_x[self.owners == cid]
            attention = F.cosine_similarity(x, self.query.unsqueeze(0), dim=1).softmax(dim=0)
            pooled.append((attention.unsqueeze(1) * x).sum(dim=0))
        return self.mlp(torch.stack(pooled))

    def integrator_edges(self, features):
        if self.top_k == 0:
            return self.base_edges.new_empty((2, 0))
        with torch.no_grad():
            unit = F.normalize(features, dim=1)
            similarity = unit @ unit.t()
            similarity.fill_diagonal_(-torch.inf)
            neighbors = similarity.argsort(dim=1, descending=True, stable=True)[:, :self.top_k]
            receivers = torch.arange(self.num_graphs, device=features.device).repeat_interleave(self.top_k)
            return torch.stack([neighbors.reshape(-1), receivers]) + self.num_original

    def build_graph(self):
        virtual = self.features()
        return Data(x=torch.cat([self.original_x, virtual]), y=self.labels,
                    edge_index=torch.cat([self.base_edges, self.integrator_edges(virtual)], dim=1))

    def class_targets(self, num_classes):
        return torch.stack([F.one_hot(self.labels[self.owners == cid], num_classes).float().mean(0)
                            for cid in range(self.num_graphs)])

    def contrastive_loss(self, virtual):
        if self.num_graphs == 1:
            return virtual.sum() * 0.
        unit = F.normalize(virtual, dim=1)
        global_feature = F.normalize(virtual.mean(0), dim=0)
        positive = (unit @ global_feature) / self.temperature
        negative = (unit @ unit.t()) / self.temperature
        negative = negative.masked_fill(torch.eye(self.num_graphs, dtype=torch.bool, device=virtual.device), -torch.inf)
        # Equation 13 excludes the global positive from the denominator.
        return (torch.logsumexp(negative, dim=1) - positive).mean()
