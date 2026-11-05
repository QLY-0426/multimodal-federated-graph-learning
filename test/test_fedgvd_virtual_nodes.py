"""Paper section 3.2: differentiable integrators and dynamic sparse topology."""
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from openfgl.flcore.fedgvd.virtual_nodes import SubgraphIntegrators
from openfgl.flcore.fedgvd.server import FedGVDServer
from openfgl.model.gcn import GCN


class VirtualNodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.graphs = [
            Data(x=torch.tensor([[1., 0., .2], [.1, 1., .3]]),
                 y=torch.tensor([0, 1]), edge_index=torch.tensor([[0, 1], [1, 0]])),
            Data(x=torch.tensor([[0., 1., 1.]]), y=torch.tensor([1]),
                 edge_index=torch.empty(2, 0, dtype=torch.long)),
            Data(x=torch.tensor([[-1., .2, 0.]]), y=torch.tensor([0]),
                 edge_index=torch.empty(2, 0, dtype=torch.long)),
        ]
        self.module = SubgraphIntegrators(self.graphs, hidden_dim=4, top_k=1, temperature=1.)

    def test_features_are_cosine_attention_mlp_and_edges_preserve_ownership(self):
        m = self.module
        weights = torch.softmax(torch.nn.functional.cosine_similarity(
            self.graphs[0].x, m.query.unsqueeze(0), dim=1), dim=0)
        expected = m.mlp((weights[:, None] * self.graphs[0].x).sum(0))
        graph = m.build_graph()
        torch.testing.assert_close(graph.x[4], expected)
        torch.testing.assert_close(graph.x[:4], torch.cat([g.x for g in self.graphs]))
        self.assertEqual(graph.y.tolist(), [0, 1, 1, 0])
        edges = set(map(tuple, graph.edge_index.t().tolist()))
        for edge in [(0, 1), (1, 0), (0, 4), (4, 0), (1, 4), (4, 1),
                     (2, 5), (5, 2), (3, 6), (6, 3)]:
            self.assertIn(edge, edges)
        self.assertNotIn((0, 5), edges)
        self.assertNotIn((4, 2), edges)
        torch.testing.assert_close(m.class_targets(2), torch.tensor([[.5, .5], [0., 1.], [1., 0.]]))

    def test_top_k_recomputed_from_current_features_without_self_edges(self):
        a = torch.tensor([[1., 0.], [.9, .1], [-1., 0.]])
        b = torch.tensor([[1., 0.], [-1., 0.], [.9, .1]])
        ea, eb = self.module.integrator_edges(a), self.module.integrator_edges(b)
        self.assertFalse(torch.equal(ea, eb))
        self.assertEqual(ea.shape, (2, 3))
        self.assertTrue((ea[0] != ea[1]).all())
        self.assertEqual(torch.bincount(ea[1] - 4).tolist(), [1, 1, 1])
        self.assertIn([5, 4], ea.t().tolist())
        self.assertIn([6, 4], eb.t().tolist())

    def test_contrastive_matches_paper_equation_13(self):
        x = torch.tensor([[1., 0.], [0., 1.], [-1., .5]], requires_grad=True)
        unit = torch.nn.functional.normalize(x, dim=1)
        center = torch.nn.functional.normalize(x.mean(0), dim=0)
        expected = torch.stack([
            torch.logsumexp(torch.stack([unit[i] @ unit[j] for j in range(3) if i != j]), 0)
            - unit[i] @ center for i in range(3)]).mean()
        torch.testing.assert_close(self.module.contrastive_loss(x), expected)

    def test_server_updates_query_mlp_and_model_across_repeated_rounds(self):
        server = FedGVDServer.__new__(FedGVDServer)
        server.integrators = self.module
        server.device = torch.device('cpu')
        model = GCN(3, 4, 2, dropout=0.)
        server.task = SimpleNamespace(model=model, optim=torch.optim.Adam(model.parameters(), lr=.01))
        before = {name: p.detach().clone() for name, p in self.module.named_parameters()}
        model_before = [p.detach().clone() for p in model.parameters()]
        server._train_global_model(self.module.build_graph())
        server._train_global_model(server.global_data)
        opt_ids = [id(p) for group in server.task.optim.param_groups for p in group['params']]
        for name, p in self.module.named_parameters():
            self.assertEqual(opt_ids.count(id(p)), 1)
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertFalse(torch.equal(before[name], p), name)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(model_before, model.parameters())))
        self.assertFalse(server.global_data.x.requires_grad)
        torch.testing.assert_close(self.module.original_x, torch.cat([g.x for g in self.graphs]))

    def test_single_client_and_zero_vectors_have_finite_loss(self):
        m = SubgraphIntegrators([self.graphs[0]], hidden_dim=4, top_k=3, temperature=1.)
        x = torch.zeros(1, 3, requires_grad=True)
        self.assertEqual(m.integrator_edges(x).shape, (2, 0))
        loss = m.contrastive_loss(x)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
