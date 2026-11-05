import unittest
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from openfgl.utils.task_utils import load_node_edge_level_default_model


def args():
    return SimpleNamespace(fl_algorithm='multimodal_fedtad', model=['multimodal_gcn'],
                           hid_dim=8, num_layers=2, dropout=0., num_clients=2,
                           weight_decay=0., num_epochs=1)


def graph():
    return Data(x=torch.randn(6, 7), x_text=torch.randn(6, 3),
                x_image=torch.randn(6, 4), y=torch.tensor([0, 1, 0, 1, 0, 1]),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]))


class FedTADTests(unittest.TestCase):
    def test_ckr_numerically_matches_original_with_same_topology_embedding(self):
        from openfgl.flcore.fedtad.client import FedTADClient
        from openfgl.flcore.multimodal_fedtad._utils import class_reliability
        from openfgl.flcore.fedtad.fedtad_config import config
        client = FedTADClient.__new__(FedTADClient)
        client.client_id, client.device = 0, 'cpu'
        data = graph()
        mask = torch.tensor([True, True, True, True, False, False])
        topology = torch.arange(18).reshape(6, 3)
        with tempfile.TemporaryDirectory() as directory:
            client.task = SimpleNamespace(data_dir=directory, data=data, num_samples=6,
                                          num_global_classes=2, train_mask=mask)
            with patch('openfgl.flcore.fedtad.client.cal_topo_emb', return_value=topology), patch.dict(config, save_ckr=False):
                client.fedtad_initialization()
            with patch('openfgl.flcore.multimodal_fedtad._utils.cal_topo_emb', return_value=topology):
                actual = class_reliability(data, mask, 2, 5)
            torch.testing.assert_close(actual, client.ckr, atol=0, rtol=0)

    def test_ckr_cache_save_load_and_default_no_save(self):
        from openfgl.flcore.multimodal_fedtad.client import MultimodalFedTADClient
        from openfgl.flcore.multimodal_fedtad.fedtad_config import config
        client = MultimodalFedTADClient.__new__(MultimodalFedTADClient)
        client.client_id, client.device = 0, 'cpu'
        with tempfile.TemporaryDirectory() as directory:
            client.task = SimpleNamespace(data_dir=directory, data=graph(),
                train_mask=torch.tensor([True, True, False, False, False, False]), num_global_classes=2)
            path = Path(directory) / 'ckr_multimodal_fedtad' / 'client_0.pt'
            value = torch.tensor([.3, .7])
            with patch('openfgl.flcore.multimodal_fedtad.client.class_reliability', return_value=value) as compute:
                with patch.dict(config, save_ckr=False):
                    client.fedtad_initialization()
                    self.assertFalse(path.exists())
                with patch.dict(config, save_ckr=True):
                    client.fedtad_initialization()
                    self.assertTrue(path.exists())
                compute.reset_mock()
                with patch.dict(config, save_ckr=False):
                    client.fedtad_initialization()
                compute.assert_not_called()
                torch.testing.assert_close(client.ckr, value)

    def test_pseudo_graph_and_branch_inputs_match_original_feature_scale(self):
        from openfgl.flcore.fedtad._utils import construct_graph
        from openfgl.flcore.multimodal_fedtad._utils import construct_modal_graph
        features = torch.randn(6, 7) * torch.tensor([1., 2., 3., 4., 20., 30., 40.])
        normalized = torch.nn.functional.normalize(features, dim=1)
        original = construct_graph(features, normalized @ normalized.t(), 3)
        adapted = construct_modal_graph(features, (3, 4), 3)
        torch.testing.assert_close(adapted.x, original.x)
        torch.testing.assert_close(adapted.edge_index, original.edge_index)
        model = load_node_edge_level_default_model(args(), (3, 4), 2)
        # Original FedTAD only supplies x and edge_index, including for pseudo graphs.
        captured = []
        handle = model.text_branch.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach()))
        model(original)
        handle.remove()
        torch.testing.assert_close(captured[0], features[:, 4:])

    def test_server_updates_numerically_match_original_fedtad(self):
        from openfgl.flcore.fedtad.server import FedTADServer
        from openfgl.flcore.multimodal_fedtad.server import MultimodalFedTADServer
        from openfgl.flcore.fedtad.fedtad_config import config as original_config
        from openfgl.flcore.multimodal_fedtad.fedtad_config import config
        settings = args()
        settings.dropout = .3
        data = graph()
        model = load_node_edge_level_default_model(settings, (3, 4), 2)
        task = SimpleNamespace(model=model, data=data, num_feats=7, num_global_classes=2)
        pool = {'sampled_clients': [0, 1]}
        for i in range(2):
            teacher = load_node_edge_level_default_model(settings, (3, 4), 2, i)
            pool[f'client_{i}'] = dict(num_samples=i + 1, model=teacher,
                weight=list(teacher.parameters()), ckr=torch.tensor([.2 + .2*i, .8 - .2*i]))
        def initialize(obj, a, d, directory, messages, device):
            obj.args, obj.task, obj.message_pool, obj.device = a, copy.deepcopy(task), messages, device
        results = []
        options = dict(num_gen=6, glb_epochs=2, it_g=2, it_d=2)
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize), patch.dict(original_config, options), patch.dict(config, options):
            for server_cls in (FedTADServer, MultimodalFedTADServer):
                torch.manual_seed(71)
                server = server_cls(settings, data, '', copy.deepcopy(pool), 'cpu')
                for _ in range(2):
                    server.execute()
                results.append(server)
        for left, right in zip(results[0].task.model.parameters(), results[1].task.model.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for left, right in zip(results[0].generator.parameters(), results[1].generator.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_ckr_uses_training_labels_only_and_handles_one_training_node(self):
        from openfgl.flcore.multimodal_fedtad._utils import class_reliability
        data = graph()
        mask = torch.tensor([True, False, False, False, False, False])
        with patch('openfgl.flcore.multimodal_fedtad._utils.cal_topo_emb', return_value=torch.ones(6, 3)):
            expected = class_reliability(data, mask, 2, 2)
            data.y[~mask] = 1 - data.y[~mask]
            actual = class_reliability(data, mask, 2, 2)
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(actual, torch.tensor([1., 0.]))

    def test_heterogeneous_configuration_is_rejected(self):
        settings = args()
        settings.model = ['multimodal_gcn', 'multimodal_gat']
        with self.assertRaisesRegex(ValueError, 'shared model'):
            load_node_edge_level_default_model(settings, (3, 4), 2)

    def test_client_downloads_weights_each_round_and_uploads_ckr(self):
        from openfgl.flcore.multimodal_fedtad.client import MultimodalFedTADClient
        settings, data = args(), graph()
        model = load_node_edge_level_default_model(settings, (3, 4), 2, 0)
        task = SimpleNamespace(model=model, data=data, data_dir=tempfile.gettempdir(), num_samples=6, num_global_classes=2,
                               train_mask=torch.tensor([True, True, False, False, False, False]))
        seen = []
        def train():
            seen.append(next(model.parameters()).detach().clone())
        task.train = train
        pool = {'server': {'weight': [torch.full_like(p, .25) for p in model.parameters()]}}
        def initialize(obj, a, cid, d, directory, messages, device):
            obj.args, obj.client_id, obj.task, obj.message_pool, obj.device = a, cid, task, messages, device
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize), patch(
                'openfgl.flcore.multimodal_fedtad.client.class_reliability', return_value=torch.tensor([.4, .6])):
            client = MultimodalFedTADClient(settings, 0, data, '', pool, 'cpu')
        client.execute()
        client.send_message()
        torch.testing.assert_close(seen[0], torch.full_like(seen[0], .25))
        pool['server']['weight'] = [torch.full_like(p, .75) for p in model.parameters()]
        client.execute()
        torch.testing.assert_close(seen[1], torch.full_like(seen[1], .75))
        torch.testing.assert_close(pool['client_0']['ckr'], torch.tensor([.4, .6]))

    def test_model_uses_equal_available_modality_weights(self):
        model = load_node_edge_level_default_model(args(), (3, 4), 2)
        data = graph()
        data.modality_mask = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 1]], dtype=torch.bool)
        _, fused, modal, weights = model.forward_modalities(data)
        torch.testing.assert_close(weights[:4], torch.tensor([[.5, .5], [1., 0], [0, 1.], [0, 0]]))
        torch.testing.assert_close(fused[0], modal[0].mean(0))
        self.assertEqual(fused[3].abs().sum().item(), 0)

    def test_generator_graph_keeps_paired_features_and_gradients(self):
        from openfgl.flcore.multimodal_fedtad.generator import MultimodalFedTADGenerator
        from openfgl.flcore.multimodal_fedtad._utils import construct_modal_graph
        generator = MultimodalFedTADGenerator(4, (3, 4), 2, 0.)
        features = generator(torch.randn(6, 4), torch.arange(6) % 2)
        data = construct_modal_graph(features, (3, 4), 5)
        self.assertEqual(tuple(data.x_text.shape), (6, 3))
        self.assertEqual(tuple(data.x_image.shape), (6, 4))
        model = load_node_edge_level_default_model(args(), (3, 4), 2)
        model(data)[1].square().sum().backward()
        self.assertGreater(generator.nodes_layer.weight.grad.abs().sum().item(), 0)

    def test_server_distillation_preserves_teacher_weights_and_broadcasts_parameters(self):
        from openfgl.flcore.multimodal_fedtad.server import MultimodalFedTADServer
        from openfgl.flcore.multimodal_fedtad.fedtad_config import config
        settings = args()
        data = graph()
        model = load_node_edge_level_default_model(settings, (3, 4), 2)
        teachers = [load_node_edge_level_default_model(settings, (3, 4), 2, i) for i in range(2)]
        pool = {'sampled_clients': [0, 1]}
        for i, teacher in enumerate(teachers):
            pool[f'client_{i}'] = dict(num_samples=i + 1, model=teacher,
                                      weight=list(teacher.parameters()), ckr=torch.tensor([.4, .6]))
        task = SimpleNamespace(model=model, data=data, num_feats=7, num_global_classes=2)
        def initialize(obj, a, d, directory, messages, device):
            obj.args, obj.task, obj.message_pool, obj.device = a, task, messages, device
        before = [[p.detach().clone() for p in t.parameters()] for t in teachers]
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize), patch.dict(config, num_gen=6, glb_epochs=1, it_g=1, it_d=1):
            server = MultimodalFedTADServer(settings, data, '', pool, 'cpu')
            with patch.dict(config, glb_epochs=0):
                server.execute()
            for p, left, right in zip(model.parameters(), before[0], before[1]):
                torch.testing.assert_close(p, (left + 2 * right) / 3)
            gen_before = server.generator.nodes_layer.weight.detach().clone()
            server.execute()
            self.assertFalse(torch.equal(gen_before, server.generator.nodes_layer.weight))
            server.send_message()
        self.assertIn('weight', pool['server'])
        for old, teacher in zip(before, teachers):
            for expected, actual in zip(old, teacher.parameters()):
                torch.testing.assert_close(expected, actual)
                self.assertTrue(actual.requires_grad)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
