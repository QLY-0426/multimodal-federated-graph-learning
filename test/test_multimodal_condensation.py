import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from openfgl.flcore.multimodal_fedgvd.SimGC_transductive import data_gc
from openfgl.flcore.multimodal_fedgvd.utils import append_synthetic_data


def fixture():
    generator = torch.Generator().manual_seed(7)
    text = torch.randn(12, 4, generator=generator)
    image = torch.randn(12, 4, generator=generator)
    edges = torch.stack([torch.arange(12), torch.arange(12).roll(1)])
    data = Data(x=torch.cat([image, text], 1), x_text=text, x_image=image,
                y=torch.tensor([0, 1, 2] * 4), edge_index=edges,
                num_global_classes=3,
                modality_mask=torch.ones(12, 2, dtype=torch.bool))
    return dict(data=data, train_mask=torch.arange(12) < 6,
                val_mask=(torch.arange(12) >= 6) & (torch.arange(12) < 9),
                test_mask=torch.arange(12) >= 9)


def options(directory):
    return SimpleNamespace(dataset=['Movies'], seed=42, num_clients=2,
        hid_dim=8, num_layers=2, dropout=0.0, lr=.01, weight_decay=0.,
        multimodal_reduction_rate=.5, multimodal_teacher_epochs=2,
        multimodal_condense_epochs=2, multimodal_student_epochs=2,
        multimodal_validation_interval=2, multimodal_fusion_temperature=.5,
        multimodal_condensed_pattern=str(Path(directory) / '{dataset}_{client_id}.pt'))


class CondensationTests(unittest.TestCase):
    def test_joint_condensation_cache_and_test_label_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            data, args = fixture(), options(directory)
            graph = data_gc(data, 0, args, torch.device('cpu'))
            self.assertEqual(graph.x_text.shape, (3, 4))
            self.assertEqual(graph.x_image.shape, (3, 4))
            self.assertEqual(sorted(graph.y.tolist()), [0, 1, 2])
            self.assertEqual(graph.modality_mask.shape, (3, 2))
            self.assertTrue(torch.isfinite(graph.x).all())
            self.assertTrue(torch.isfinite(graph.edge_weight).all())
            self.assertFalse(graph.x_text.requires_grad)
            original = data['data'].x_text[data['train_mask']]
            self.assertFalse(torch.equal(graph.x_text, original[:3]))
            again = data_gc(data, 0, args, torch.device('cpu'))
            torch.testing.assert_close(graph.x, again.x, rtol=0, atol=0)
            data['data'].y[data['test_mask']] = 0
            again = data_gc(data, 0, args, torch.device('cpu'))
            torch.testing.assert_close(graph.x, again.x, rtol=0, atol=0)
            next(Path(directory).glob('*.pt')).unlink()
            regenerated = data_gc(data, 0, args, torch.device('cpu'))
            torch.testing.assert_close(graph.x, regenerated.x, rtol=0, atol=0)

    def test_append_preserves_real_evaluation_graph_and_offsets(self):
        data = fixture()
        syn = Data(x=torch.ones(2, 8), x_text=torch.ones(2, 4),
                   x_image=torch.ones(2, 4), y=torch.tensor([0, 1]),
                   edge_index=torch.tensor([[0, 1], [1, 0]]),
                   edge_weight=torch.tensor([.5, .5]),
                   modality_mask=torch.ones(2, 2, dtype=torch.bool))
        augmented = append_synthetic_data(data, syn)
        self.assertEqual(data['data'].num_nodes, 12)
        self.assertEqual(data['train_mask'].numel(), 12)
        self.assertEqual(augmented['data'].num_nodes, 14)
        torch.testing.assert_close(augmented['data'].edge_index[:, -2:], syn.edge_index + 12)
        self.assertFalse(augmented['val_mask'][-2:].any())
        self.assertFalse(augmented['test_mask'][-2:].any())
        self.assertTrue(augmented['train_mask'][-2:].all())
        self.assertEqual(augmented['data'].edge_weight.numel(), augmented['data'].num_edges)

    def test_cache_invalidated_by_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            data, args = fixture(), options(directory)
            data_gc(data, 0, args, torch.device('cpu'))
            path = next(Path(directory).glob('*.pt'))
            first = torch.load(path)['signature']
            data['train_mask'][0] = False
            data_gc(data, 0, args, torch.device('cpu'))
            self.assertNotEqual(first, torch.load(path)['signature'])

    def test_single_proxy_with_unavailable_image(self):
        with tempfile.TemporaryDirectory() as directory:
            data, args = fixture(), options(directory)
            data['train_mask'][:] = False
            data['train_mask'][0] = True
            data['data'].modality_mask[:, 1] = False
            graph = data_gc(data, 0, args, torch.device('cpu'))
            self.assertEqual(graph.num_nodes, 1)
            self.assertEqual(graph.edge_index.shape, (2, 0))
            self.assertFalse(graph.modality_mask[:, 1].any())
            self.assertTrue(torch.isfinite(graph.x).all())


if __name__ == '__main__':
    unittest.main()
