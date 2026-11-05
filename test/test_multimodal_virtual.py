import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from openfgl.flcore.multimodal_fedgvd import contrastive
from openfgl.flcore.multimodal_fedgvd.server import MultimodalFedGVDServer
from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload
from openfgl.model.multimodal_gcn import MultimodalGCN


def graphs():
    result = {}
    for cid in (7, 2, 9):
        text = torch.tensor([[1., .2, .3], [.1, 1., .5], [.4, .2, 1.]]) + cid / 20
        image = torch.tensor([[.3, 1.], [1., .1], [.6, .8]]) - cid / 30
        result[cid] = Data(x=torch.cat((image, text), 1), x_text=text, x_image=image, y=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1], [1, 0]]), edge_weight=torch.ones(2),
            modality_mask=torch.ones(3, 2, dtype=torch.bool))
    return result


class VirtualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def builder(self, items=None, k=1):
        # Assert the missing feature before importing it, so RED is a clear failure.
        from openfgl.flcore.multimodal_fedgvd import server
        self.assertTrue(hasattr(server, 'VirtualGraphBuilder'), 'server must support learnable virtual nodes')
        base, slices = contrastive.get_coordinated_data(items or graphs(), cross_k=0)
        return server.VirtualGraphBuilder(base, slices, num_classes=2, hidden_dim=8, cross_k=k)

    def test_aligned_nodes_stars_and_no_direct_cross_client_edges(self):
        builder = self.builder()
        data = builder()
        self.assertEqual(data.num_nodes, 12)
        self.assertEqual(data.y.tolist(), [0, 1, 0] * 3 + [-1] * 3)
        torch.testing.assert_close(data.virtual_targets, torch.tensor([[2/3, 1/3]] * 3))
        self.assertEqual(int(data.synthetic_mask.sum()), 9)
        for name in ('text', 'image'):
            edges = set(map(tuple, getattr(data, 'edge_index_' + name).t().tolist()))
            for index, part in enumerate(builder.client_slices.values()):
                for node in range(part.start, part.stop):
                    self.assertIn((node, 9 + index), edges)
                    self.assertIn((9 + index, node), edges)
            for a, b in edges:
                if a < 9 and b < 9:
                    self.assertEqual(a // 3, b // 3)
                self.assertIn((b, a), edges)

    def test_query_and_mlp_receive_gradient_from_original_node_predictions(self):
        builder = self.builder()
        model = MultimodalGCN((3, 2), 8, 2, dropout=0.)
        before = {name: p.detach().clone() for name, p in builder.named_parameters()}
        optimizer = torch.optim.Adam(builder.parameters(), lr=.02)
        for _ in range(2):
            optimizer.zero_grad()
            data = builder()
            _, logits, _, _ = model.forward_modalities(data)
            F.cross_entropy(logits[data.synthetic_mask], data.y[data.synthetic_mask]).backward()
            for name, parameter in builder.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(builder.queries[0].grad.norm()), 0)
            self.assertGreater(float(builder.queries[1].grad.norm()), 0)
            optimizer.step()
        self.assertTrue(all(not torch.equal(before[name], p) for name, p in builder.named_parameters()))

    def test_missing_modality_and_single_client_are_finite(self):
        items = {2: graphs()[2]}
        items[2].modality_mask[:, 1] = False
        builder = self.builder(items, k=10)
        data = builder()
        self.assertFalse(data.modality_mask[-1, 1])
        self.assertEqual(data.edge_index_image.numel(), 0)
        self.assertTrue(torch.equal(data.x_image[-1], torch.zeros(2)))
        from openfgl.flcore.multimodal_fedgvd.virtual import integrator_contrastive_loss
        for mode in ('paper', 'infonce'):
            loss = integrator_contrastive_loss(data, temperature=.5, mode=mode)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(loss), 0)
            loss.backward(retain_graph=True)

    def test_contrastive_modes_match_analytical_values(self):
        self.builder()
        from openfgl.flcore.multimodal_fedgvd.virtual import integrator_contrastive_loss
        data = Data(x_text=torch.eye(2, requires_grad=True), x_image=torch.eye(2, requires_grad=True),
                    modality_mask=torch.ones(2, 2, dtype=torch.bool), virtual_start=0)
        positive = 2 ** -.5 / .5
        paper = integrator_contrastive_loss(data, .5, 'paper')
        info = integrator_contrastive_loss(data, .5, 'infonce')
        self.assertAlmostEqual(float(paper), -positive, places=5)
        self.assertAlmostEqual(float(info), float(F.softplus(torch.tensor(-positive))), places=5)
        for temperature in (0., float('nan')):
            with self.assertRaises(ValueError):
                integrator_contrastive_loss(data, temperature, 'paper')

    def test_soft_targets_ignore_unobserved_nodes_and_match_soft_cross_entropy(self):
        items = {2: graphs()[2]}
        items[2].modality_mask[0] = False
        builder = self.builder(items)
        data = builder()
        torch.testing.assert_close(data.virtual_targets, torch.tensor([[.5, .5]]))
        from openfgl.flcore.multimodal_fedgvd.virtual import virtual_classification_loss
        logits = torch.tensor([[0., 0.]] * 3 + [[2., -1.]], requires_grad=True)
        loss = virtual_classification_loss(data, logits, torch.stack((logits, logits), 1), .5)
        expected = -1.5 * F.log_softmax(logits[-1], 0).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[:3].norm()), 0.)
        self.assertGreater(float(logits.grad[-1].norm()), 0.)

    def test_all_modalities_missing_has_no_virtual_edges_or_auxiliary_loss(self):
        items = {2: graphs()[2]}
        items[2].modality_mask[:] = False
        data = self.builder(items)()
        from openfgl.flcore.multimodal_fedgvd.virtual import virtual_classification_loss
        self.assertFalse(data.modality_mask.any())
        self.assertEqual(data.edge_index_text.numel(), 0)
        self.assertEqual(data.edge_index_image.numel(), 0)
        model = MultimodalGCN((3, 2), 8, 2, dropout=0.)
        _, fused, modal, _ = model.forward_modalities(data)
        loss = virtual_classification_loss(data, fused, modal)
        self.assertEqual(float(loss), 0.)
        loss.backward()

    def test_topology_recomputed_from_current_virtual_features(self):
        builder = self.builder(k=1)
        # Replace only aggregation to isolate the dynamic-edge contract.
        first = torch.tensor([[1., 0.], [.9, .1], [0., 1.]])
        second = torch.tensor([[1., 0.], [0., 1.], [.9, .1]])
        with patch.object(builder, '_aggregate', side_effect=[F.pad(first, (0, 1)), first,
                                                            F.pad(second, (0, 1)), second]):
            a, b = builder(), builder()
        self.assertFalse(torch.equal(a.edge_index_text, b.edge_index_text))

    def test_server_updates_virtual_parameters_and_excludes_them_from_payload(self):
        self.builder()
        items = graphs()
        pool = {'sampled_clients': [7, 2], **{f'client_{cid}': {'graph': graph_to_payload(items[cid])} for cid in (7, 2)}}
        model = MultimodalGCN((3, 2), 8, 2, dropout=0.)
        task = SimpleNamespace(model=model, optim=torch.optim.Adam(model.parameters(), lr=.02))
        args = SimpleNamespace(multimodal_server_mode='dual_virtual', multimodal_server_epochs=2,
                               hid_dim=8, multimodal_virtual_ce_weight=1., multimodal_contrastive_weight=1.)
        def init(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, task
        with patch('openfgl.flcore.base.BaseServer.__init__', init):
            server = MultimodalFedGVDServer(args, None, '', pool, 'cpu')
        server.execute()
        before = {name: p.detach().clone() for name, p in server.integrator.named_parameters()}
        pool.update(sampled_clients=[9], client_9={'graph': graph_to_payload(items[9])}, client_7={}, client_2={})
        server.execute()
        server.send_message()
        self.assertEqual(server.global_data.num_nodes, 12)
        self.assertEqual(len(task.optim.param_groups), 2)
        self.assertTrue(any(not torch.equal(before[name], p) for name, p in server.integrator.named_parameters()))
        with torch.no_grad():
            _, fused, _, _ = model.forward_modalities(server.global_data)
        for cid, part in server.client_slices.items():
            self.assertEqual(server.syn_logits[cid].shape, (3, 2))
            torch.testing.assert_close(server.syn_logits[cid], fused[part])
        for payload in pool['server']['graphs'][9].values():
            self.assertEqual(len(payload['y']), 3)
        self.assertGreater(server.diagnostics[-1]['query_grad_norm'], 0)


if __name__ == '__main__':
    unittest.main()
