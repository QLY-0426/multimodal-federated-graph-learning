import importlib

import unittest
import torch
from torch_geometric.data import Data

from openfgl.model.multimodal_utils import reliability_fusion, multimodal_supervised_loss


MODELS = [('gcn', 'GCN'), ('gat', 'GAT'), ('sgc', 'SGC'), ('mlp', 'MLP'), ('graphsage', 'GraphSAGE')]


def graph():
    torch.manual_seed(13)
    return Data(x_text=torch.randn(5, 4), x_image=torch.randn(5, 6),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]]))


def model(kind, name):
    cls = getattr(importlib.import_module('openfgl.model.multimodal_' + kind), 'Multimodal' + name)
    return cls((4, 6), 8, 3, dropout=0)


def test_independent_branches_and_backward(kind, name):
    net, data = model(kind, name), graph()
    embedding, fused, logits, weights = net.forward_modalities(data)
    assert embedding.shape[0] == 5
    assert fused.shape == (5, 3)
    assert logits.shape == (5, 2, 3)
    assert weights.shape == (5, 2)
    assert torch.allclose(weights.sum(1), torch.ones(5))
    changed = data.clone()
    changed.x_image = torch.randn_like(data.x_image)
    changed_logits = net.forward_modalities(changed)[2]
    assert torch.allclose(logits[:, 0], changed_logits[:, 0])
    loss = multimodal_supervised_loss(logits, fused, torch.tensor([0, 1, 2, 0, 1]), torch.ones(5, dtype=torch.bool))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.text_branch.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.image_branch.parameters())
    assert torch.allclose(net(data)[1], fused)


def test_masked_inputs_and_both_missing(kind, name):
    net, data = model(kind, name), graph()
    data.modality_mask = torch.tensor([[1, 0], [1, 1], [0, 0], [1, 0], [1, 1]], dtype=torch.bool)
    original = net.forward_modalities(data)
    data.x_image[~data.modality_mask[:, 1]] = 1e6
    changed = net.forward_modalities(data)
    assert torch.allclose(original[1], changed[1])
    assert torch.isfinite(changed[1]).all()
    assert torch.equal(changed[1][2], torch.zeros(3))
    assert torch.equal(changed[3][2], torch.zeros(2))
    assert torch.equal(changed[3][0], torch.tensor([1., 0.]))


def test_separate_topologies(kind, name):
    net, data = model(kind, name), graph()
    before = net.forward_modalities(data)[2]
    data.edge_index_image = torch.empty((2, 0), dtype=torch.long)
    after = net.forward_modalities(data)[2]
    assert torch.allclose(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_edge_weights_differentiable(kind, name):
    net, data = model(kind, name), graph()
    data.edge_weight_text = torch.rand(data.edge_index.size(1), requires_grad=True)
    data.edge_weight_image = torch.rand(data.edge_index.size(1), requires_grad=True)
    net(data)[1].square().sum().backward()
    for weight in [data.edge_weight_text, data.edge_weight_image]:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0


def test_reliability_and_empty_supervision():
    text = torch.tensor([[8., -8., -8.]], requires_grad=True)
    image = torch.zeros(1, 3, requires_grad=True)
    fused, weights = reliability_fusion(text, image)
    assert weights[0, 0] > weights[0, 1]
    assert not weights.requires_grad
    with unittest.TestCase().assertRaises(ValueError):
        reliability_fusion(text, image, temperature=0)
    modalities = torch.stack([text, image], 1)
    for mask, available in [(torch.tensor([False]), None), (torch.tensor([True]), torch.zeros(1, 2, dtype=torch.bool))]:
        loss = multimodal_supervised_loss(modalities, fused, torch.tensor([0]), mask, available)
        assert loss.item() == 0
        assert loss.requires_grad

class MultimodalModelTests(unittest.TestCase):
    def test_reliability(self):
        test_reliability_and_empty_supervision()


def _case(function, kind, name):
    def run(self):
        function(kind, name)
    return run


for _kind, _name in MODELS:
    for _function in [test_independent_branches_and_backward, test_masked_inputs_and_both_missing]:
        setattr(MultimodalModelTests, _function.__name__ + '_' + _kind, _case(_function, _kind, _name))
    if _kind != 'mlp':
        setattr(MultimodalModelTests, 'test_topology_' + _kind, _case(test_separate_topologies, _kind, _name))
    if _kind in ('gcn', 'sgc'):
        setattr(MultimodalModelTests, 'test_weights_' + _kind, _case(test_edge_weights_differentiable, _kind, _name))
