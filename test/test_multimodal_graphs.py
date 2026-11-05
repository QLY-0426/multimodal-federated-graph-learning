import unittest
import math
import torch
from torch_geometric.data import Data

from openfgl.flcore.multimodal_fedgvd.contrastive import get_coordinated_data


def graph(text, image, edges=None, weights=None, mask=None):
    return Data(x_text=torch.tensor(text, dtype=torch.float),
                x_image=torch.tensor(image, dtype=torch.float),
                y=torch.arange(len(text)),
                edge_index=torch.tensor(edges if edges is not None else [[], []], dtype=torch.long),
                edge_weight=None if weights is None else torch.tensor(weights),
                modality_mask=None if mask is None else torch.tensor(mask))


def edge_map(data, modality):
    edges = getattr(data, 'edge_index_' + modality)
    weights = getattr(data, 'edge_weight_' + modality)
    return {tuple(edge): float(w) for edge, w in zip(edges.t().tolist(), weights)}


def test_modalities_choose_different_cross_client_neighbors_without_labels():
    a = graph([[1, 0]], [[1, 0]])
    b = graph([[1, 0], [0, 1]], [[0, 1], [1, 0]])
    data, slices = get_coordinated_data({9: b, 2: a}, cross_k=1, similarity_threshold=.5)
    assert slices == {2: slice(0, 1), 9: slice(1, 3)}
    assert set(edge_map(data, 'text')) == {(0, 1), (1, 0)}
    assert set(edge_map(data, 'image')) == {(0, 2), (2, 0)}
    a.y.fill_(88)
    b.y.fill_(-1)
    changed, _ = get_coordinated_data({2: a, 9: b}, cross_k=1, similarity_threshold=.5)
    assert torch.equal(data.edge_index_text, changed.edge_index_text)
    assert torch.equal(data.edge_index_image, changed.edge_index_image)
    assert data.num_nodes == 3
    assert torch.all(data.modality_mask)


def test_local_offsets_weights_and_no_same_client_added_edges():
    a = graph([[1., 0], [1, 0]], [[1., 0], [1, 0]], [[0], [1]], [.7])
    b = graph([[1., 0], [1, 0], [1, 0]], [[1., 0], [1, 0], [1, 0]], [[0], [2]], [.3])
    data, _ = get_coordinated_data({8: b, 3: a}, cross_k=1)
    assert torch.equal(data.edge_index, torch.tensor([[0, 2], [1, 4]]))
    for modality in ('text', 'image'):
        edges = edge_map(data, modality)
        assert math.isclose(edges[(0, 1)], .7, abs_tol=1e-6)
        assert math.isclose(edges[(1, 0)], .7, abs_tol=1e-6)
        assert math.isclose(edges[(2, 4)], .3, abs_tol=1e-6)
        for (u, v), weight in edges.items():
            assert u != v
            assert math.isclose(edges[(v, u)], weight, abs_tol=1e-6)
            if (u < 2) == (v < 2):
                assert (u, v) in {(0, 1), (1, 0), (2, 4), (4, 2)}


def test_missing_modalities_remove_all_incident_edges():
    a = graph([[1., 0], [1, 0]], [[1., 0], [1, 0]], [[0], [1]], mask=[[True, False], [False, True]])
    b = graph([[1., 0]], [[1., 0]])
    data, _ = get_coordinated_data({0: a, 1: b}, cross_k=2)
    assert 1 not in data.edge_index_text
    assert 0 not in data.edge_index_image
    assert set(edge_map(data, 'text')) == {(0, 2), (2, 0)}
    assert set(edge_map(data, 'image')) == {(1, 2), (2, 1)}


def test_single_client_preserves_only_local_graph():
    cross_k = 2
    a = graph([[3., 0], [0, 2]], [[0., 4], [1, 0]], [[0], [1]])
    data, _ = get_coordinated_data({7: a}, cross_k=cross_k)
    assert set(edge_map(data, 'text')) == {(0, 1), (1, 0)}
    assert torch.equal(data.x[:, :2], torch.nn.functional.normalize(a.x_image, dim=1))
    assert torch.equal(data.x[:, 2:], torch.nn.functional.normalize(a.x_text, dim=1))


def test_zero_k_never_adds_cross_client_edges():
    a = graph([[1., 0]], [[1., 0]])
    data, _ = get_coordinated_data({0: a, 1: a}, cross_k=0)
    assert data.edge_index_text.shape == (2, 0)


def test_invalid_arguments():
    a = graph([[1., 0]], [[1., 0]])
    with unittest.TestCase().assertRaises(ValueError):
        get_coordinated_data({})
    for kwargs in ({'cross_k': -1}, {'similarity_threshold': -1.1}, {'similarity_threshold': 1.1}):
        with unittest.TestCase().assertRaises(ValueError):
            get_coordinated_data({0: a}, **kwargs)

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items() if name.startswith("test_") and callable(value))
