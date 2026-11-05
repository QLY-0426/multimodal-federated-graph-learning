import unittest
from types import SimpleNamespace

from openfgl.utils.task_utils import load_node_edge_level_default_model


class LoadingTests(unittest.TestCase):
    def test_heterogeneous_clients_and_independent_server(self):
        args = SimpleNamespace(fl_algorithm='multimodal_fedgvd',
            model=['multimodal_gat', 'multimodal_sgc', 'multimodal_mlp',
                   'multimodal_graphsage', 'multimodal_gcn'],
            num_clients=10, hid_dim=8, num_layers=2, dropout=0.,
            multimodal_server_model='multimodal_gcn', multimodal_fusion_temperature=.5)
        for cid, expected in [(0, 'MultimodalGAT'), (2, 'MultimodalSGC'),
                              (4, 'MultimodalMLP'), (6, 'MultimodalGraphSAGE'),
                              (8, 'MultimodalGCN'), (None, 'MultimodalGCN')]:
            model = load_node_edge_level_default_model(args, (4, 6), 3, cid)
            self.assertEqual(type(model).__name__, expected)

    def test_original_model_factory_unchanged(self):
        args = SimpleNamespace(fl_algorithm='fedgvd', model=['gcn'], num_clients=10,
                               hid_dim=8, num_layers=2, dropout=0.)
        model = load_node_edge_level_default_model(args, 10, 3, 0)
        self.assertEqual(type(model).__name__, 'GCN')


if __name__ == '__main__':
    unittest.main()
