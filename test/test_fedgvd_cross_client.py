"""Cross-client graph/logit routing regression tests (no condensation needed)."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

# The legacy condensation module parses argv at import time.
with patch.object(sys, 'argv', [sys.argv[0]]):
    from openfgl.flcore.fedgvd.client import FedGVDClient
    from openfgl.flcore.fedgvd.server import FedGVDServer


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.pool = {'round': 1, 'sampled_clients': [2, 0, 1],
                     'feat_syn': {}, 'label_syn': {}, 'edge_index_syn': {}, 'syn_logits': []}
        for cid, n in [(2, 3), (0, 1), (1, 2)]:
            item = {'feat_syn': torch.full((n, 2), float(cid)),
                    'label_syn': torch.full((n,), cid),
                    'edge_index_syn': torch.tensor([[0], [n - 1]])}
            self.pool[f'client_{cid}'] = item.copy()
            for key, value in item.items():
                self.pool[key][cid] = value
        self.client = FedGVDClient.__new__(FedGVDClient)
        self.client.client_id = 1
        self.client.message_pool = self.pool
        self.client.device = torch.device('cpu')
        self.client.local_data_len = 2
        for key in ('feat_syn', 'label_syn', 'edge_index_syn'):
            setattr(self.client, key, self.pool[key][1])

    def test_graph_excludes_self_and_has_stable_order(self):
        x, y, edges = self.client.cat_syn_data()
        self.assertEqual(y.tolist(), [0, 2, 2, 2])
        self.assertEqual(edges.tolist(), [[0, 1], [0, 3]])
        self.assertEqual(x[:, 0].tolist(), [0, 2, 2, 2])

    def test_execute_merges_other_graphs_without_mutating_upload(self):
        data = Data(x=torch.ones(2, 2), y=torch.tensor([0, 1]),
                    edge_index=torch.tensor([[0], [1]]))
        processed = {'data': data, 'train_mask': torch.tensor([True, False]),
                     'val_mask': torch.tensor([False, True]), 'test_mask': torch.tensor([False, True])}
        self.client.task = SimpleNamespace(processed_data=processed, train=lambda: None)
        self.client.update_syn_logits = lambda: None
        self.client.execute()
        self.assertEqual(data.y.tolist(), [0, 1, 0, 2, 2, 2])
        self.assertEqual(data.edge_index.tolist(), [[0, 2, 3], [1, 2, 5]])
        self.assertEqual(processed['train_mask'].tolist(), [True, False, True, True, True, True])
        self.assertEqual(processed['val_mask'].sum().item(), 1)
        self.assertEqual(self.client.edge_index_syn.tolist(), [[0], [1]])

    def test_server_targets_match_graph_order_excluding_virtual_nodes(self):
        server = FedGVDServer.__new__(FedGVDServer)
        server.message_pool = self.pool
        server.device = torch.device('cpu')
        graph = server._aggregate_subgraphs()
        class Model:
            def eval(self):
                pass
            def __call__(self, data):
                return data.x, data.x
        server.task = SimpleNamespace(model=Model())
        # Changing sampling order after aggregation must not change node ownership.
        self.pool['sampled_clients'] = [1, 2, 0]
        server._calculate_client_logits(graph)
        self.assertEqual(self.pool['client_1']['syn_logits'][:, 0].tolist(), [0, 2, 2, 2])
        self.assertEqual(self.pool['client_0']['syn_logits'][:, 0].tolist(), [1, 1, 2, 2, 2])
        self.assertEqual(self.pool['client_2']['syn_logits'][:, 0].tolist(), [0, 1, 1])

    def test_distillation_uses_aligned_remote_targets_and_backpropagates(self):
        self.pool['round'] = 2
        self.client.task = SimpleNamespace(default_loss_fn=torch.nn.functional.cross_entropy)
        logits = torch.randn(6, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 2, 2, 2])
        mask = torch.tensor([True, False, True, True, True, True])
        self.pool['client_1']['syn_logits'] = logits[2:].detach().clone()
        loss_fn = self.client.get_custom_loss_fn()
        loss = loss_fn(None, logits, labels, mask)
        from openfgl.flcore.fedgvd.fedgvd_config import config
        expected = (torch.nn.functional.cross_entropy(logits[:1], labels[:1])
                    + config['gama'] * torch.nn.functional.cross_entropy(logits[2:], labels[2:]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        val_mask = torch.tensor([False, True, False, False, False, False])
        torch.testing.assert_close(loss_fn(None, logits, labels, val_mask),
                                   torch.nn.functional.cross_entropy(logits[1:2], labels[1:2]))

    def test_single_client_has_empty_remote_graph(self):
        for key in ('feat_syn', 'label_syn', 'edge_index_syn'):
            self.pool[key] = {1: self.pool[key][1]}
        x, y, edges = self.client.cat_syn_data()
        self.assertEqual(tuple(x.shape), (0, 2))
        self.assertEqual(tuple(edges.shape), (2, 0))
        self.assertEqual(y.numel(), 0)


if __name__ == '__main__':
    unittest.main()
