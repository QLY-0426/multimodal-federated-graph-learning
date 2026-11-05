"""Two original FED-PUB masked GCN branches with available-modality mean fusion."""
import torch
from torch import nn
from torch.nn import functional as F

from openfgl.flcore.fedpub.maskedgcn import MaskedGCN


class _MaskedBranch(MaskedGCN):
    def __init__(self, input_dim, hid_dim, output_dim, dropout, **kwargs):
        super().__init__(input_dim, hid_dim, output_dim, **kwargs)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, present):
        # Mask after every layer so biases cannot make missing nodes send messages.
        x = x.masked_fill(~present[:, None], 0)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.masked_fill(~present[:, None], 0)
        embedding = self.conv2(x, edge_index, edge_weight)
        embedding = embedding.masked_fill(~present[:, None], 0)
        logits = self.clsif(F.dropout(F.relu(embedding), p=self.dropout, training=self.training))
        return embedding, logits.masked_fill(~present[:, None], 0)


class MultimodalMaskedGCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout=0.5,
                 l1=1e-3, laye_mask_one=True, clsf_mask_one=True):
        super().__init__()
        if isinstance(input_dim, int):
            if input_dim <= 0 or input_dim % 2:
                raise ValueError('integer input_dim must be positive and even')
            input_dim = (input_dim // 2, input_dim // 2)
        if len(input_dim) != 2 or min(input_dim) <= 0:
            raise ValueError('two positive modality dimensions are required')
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be between zero and one')
        self.modality_dims = tuple(input_dim)  # text, image
        options = dict(l1=l1, laye_mask_one=laye_mask_one, clsf_mask_one=clsf_mask_one)
        self.text_branch = _MaskedBranch(input_dim[0], hid_dim, output_dim, dropout, **options)
        self.image_branch = _MaskedBranch(input_dim[1], hid_dim, output_dim, dropout, **options)

    def forward_modalities(self, data):
        # Match MAGB's [image, text] x layout and preserve Gaussian proxy scale.
        image, text = data.x.split((self.modality_dims[1], self.modality_dims[0]), dim=1)
        available = getattr(data, 'modality_mask', None)
        if available is None:
            available = torch.ones((data.x.size(0), 2), dtype=torch.bool, device=data.x.device)
        else:
            available = available.to(device=data.x.device, dtype=torch.bool)
        embeddings, logits = [], []
        for i, (name, branch, features) in enumerate((
                ('text', self.text_branch, text), ('image', self.image_branch, image))):
            edges = getattr(data, 'edge_index_' + name, None)
            if edges is None:
                edges = data.edge_index
                edge_weight = getattr(data, 'edge_weight', None)
                if edge_weight is None:
                    edge_weight = getattr(data, 'edge_attr', None)
            else:
                edge_weight = getattr(data, 'edge_weight_' + name, None)
            present = available[:, i]
            keep = present[edges[0]] & present[edges[1]]
            embedding, prediction = branch(features, edges[:, keep],
                edge_weight[keep] if edge_weight is not None else None, present)
            embeddings.append(embedding)
            logits.append(prediction)
        modal_logits = torch.stack(logits, dim=1)
        weights = available.to(modal_logits.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
        fused = (modal_logits * weights.unsqueeze(-1)).sum(1)
        return torch.cat(embeddings, dim=-1), fused, modal_logits, weights

    def forward(self, data):
        embedding, logits, _, _ = self.forward_modalities(data)
        return embedding, logits
