"""FED-PUB mechanism checks, including numerical parity with single-modality code."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from openfgl.utils.task_utils import load_node_edge_level_default_model


def args():
    return SimpleNamespace(fl_algorithm='multimodal_fedpub', model=['multimodal_gcn'],
                           hid_dim=5, num_layers=2, dropout=0., num_clients=3,
                           num_epochs=1, seed=42)


def graph():
    text, image = torch.randn(6, 3), torch.randn(6, 4)
    return Data(x=torch.cat((image, text), 1), x_text=text, x_image=image,
                y=torch.tensor([0, 1, 0, 1, 0, 1]),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]))


def model():
    return load_node_edge_level_default_model(args(), (3, 4), 2)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def make_client(pool, cid=0):
    from openfgl.flcore.multimodal_fedpub.client import MultimodalFedPubClient
    client = MultimodalFedPubClient.__new__(MultimodalFedPubClient)
    client.args, client.client_id, client.device, client.message_pool = args(), cid, 'cpu', pool
    client.task = SimpleNamespace(model=model(), data=graph(), num_samples=6, train=lambda: None)
    return client


def make_server(pool):
    from openfgl.flcore.multimodal_fedpub.server import MultimodalFedPubServer
    task = SimpleNamespace(model=model(), data=graph(), num_global_classes=2)
    def initialize(obj, settings, data, directory, messages, device, personalized=False):
        obj.args, obj.task, obj.message_pool, obj.device = settings, task, messages, device
        obj.personalized = personalized
    with patch('openfgl.flcore.base.BaseServer.__init__', initialize):
        return MultimodalFedPubServer(args(), task.data, '', pool, 'cpu')


class MultimodalFedPubTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_loader_builds_two_masked_branches_with_unequal_dimensions(self):
        network = model()
        self.assertIsNotNone(network, 'multimodal_fedpub must be registered in the model loader')
        embedding, logits = network(graph())
        self.assertEqual(embedding.shape, (6, 10))
        self.assertEqual(logits.shape, (6, 2))
        names = dict(network.named_parameters())
        for branch in ('text_branch', 'image_branch'):
            for layer in ('conv1', 'conv2', 'clsif'):
                self.assertIn(f'{branch}.{layer}.mask', names)

    def test_each_branch_matches_original_maskedgcn_on_raw_inputs(self):
        from openfgl.flcore.fedpub.maskedgcn import MaskedGCN
        network, data = model().eval(), graph()
        embedding, fused, modal, weights = network.forward_modalities(data)
        for i, (name, features) in enumerate((('text', data.x_text), ('image', data.x_image))):
            original = MaskedGCN(features.size(1), 5, 2).eval()
            original.load_state_dict(getattr(network, name + '_branch').state_dict())
            expected_embedding, expected_logits = original(Data(x=features, edge_index=data.edge_index))
            torch.testing.assert_close(embedding[:, i * 5:(i + 1) * 5], expected_embedding)
            torch.testing.assert_close(modal[:, i], expected_logits)
        torch.testing.assert_close(fused, modal.mean(1))
        torch.testing.assert_close(weights, torch.full((6, 2), .5))

    def test_missing_modalities_cannot_affect_present_nodes(self):
        network, data = model().eval(), graph()
        data.modality_mask = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 1]], dtype=torch.bool)
        embedding, fused, modal, weights = network.forward_modalities(data)
        torch.testing.assert_close(weights[:4], torch.tensor([[.5, .5], [1., 0], [0, 1.], [0, 0]]))
        self.assertEqual(embedding[3].abs().sum().item(), 0)
        self.assertEqual(fused[3].abs().sum().item(), 0)
        data.x[~data.modality_mask[:, 1], :4] = 1e6
        data.x[~data.modality_mask[:, 0], 4:] = -1e6
        torch.testing.assert_close(network(data)[1], fused)
        self.assertEqual(modal[1, 1].abs().sum().item(), 0)

    def test_masks_prune_only_in_eval(self):
        network, data = model(), graph()
        with torch.no_grad():
            for name, parameter in network.named_parameters():
                if 'mask' in name:
                    parameter.fill_(1e-4)
        network.eval()
        pruned = network(data)[1]
        with torch.no_grad():
            for name, parameter in network.named_parameters():
                if 'mask' in name:
                    parameter.zero_()
        torch.testing.assert_close(pruned, network(data)[1])
        # Keep upstream features nonzero when testing a sub-threshold classifier mask.
        with torch.no_grad():
            for name, parameter in network.named_parameters():
                if 'mask' in name:
                    parameter.fill_(1e-4 if 'clsif' in name else 1.)
        network.train()
        network(data)[1].square().sum().backward()
        self.assertGreater(network.text_branch.clsif.mask.grad.abs().sum().item(), 0)

    def test_heterogeneous_or_non_gcn_models_are_rejected(self):
        for choices in (['multimodal_gcn', 'multimodal_gat'], ['multimodal_mlp']):
            settings = args()
            settings.model = choices
            with self.assertRaisesRegex(ValueError, 'shared.*[Mm]askedGCN'):
                load_node_edge_level_default_model(settings, (3, 4), 2)

    def test_download_preserves_local_masks_and_message_snapshots(self):
        pool = {'round': 1, 'server': {}}
        client = make_client(pool)
        original = snapshot(client.task.model)
        incoming = {name: torch.full_like(value, .4) for name, value in original.items()}
        pool['server'] = {'weight': incoming, 'personalized_0': {name: value + .2 for name, value in incoming.items()}}
        before = copy.deepcopy(pool['server'])
        client.execute()
        for name, value in client.task.model.state_dict().items():
            torch.testing.assert_close(value, original[name] if 'mask' in name else before['personalized_0'][name])
            torch.testing.assert_close(pool['server']['weight'][name], before['weight'][name])
            torch.testing.assert_close(pool['server']['personalized_0'][name], before['personalized_0'][name])
        anchor = copy.deepcopy(client.prev_w)
        with torch.no_grad():
            for parameter in client.task.model.parameters():
                parameter.add_(1)
        for name, value in anchor.items():
            torch.testing.assert_close(client.prev_w[name], value)

    def test_regularizers_have_gradients_and_l2_starts_after_round_zero(self):
        from openfgl.flcore.multimodal_fedpub.fedpub_config import config
        pool = {'round': 0, 'server': {}}
        client = make_client(pool)
        pool['server']['weight'] = snapshot(client.task.model)
        client.execute()
        logits = torch.zeros(6, 2, requires_grad=True)
        mask = torch.ones(6, dtype=torch.bool)
        with patch.dict(config, l1=.03, loc_l2=.07):
            loss = client.get_custom_loss_fn()(None, logits, client.task.data.y, mask)
            loss.backward()
            torch.testing.assert_close(client.task.model.text_branch.conv1.mask.grad,
                                       torch.full_like(client.task.model.text_branch.conv1.mask, .03))
            self.assertIsNone(client.task.model.text_branch.conv1.lin.weight.grad)
            pool['round'] = 1
            client.task.model.zero_grad()
            parameter = client.task.model.text_branch.conv1.lin.weight
            with torch.no_grad():
                parameter.add_(.25)
            loss = client.get_custom_loss_fn()(None, logits, client.task.data.y, mask)
            loss.backward()
            expected = torch.full_like(parameter, .07 / parameter.numel() ** .5)
            torch.testing.assert_close(parameter.grad, expected)

    def test_functional_embedding_is_mean_hidden_output_and_upload_is_snapshot(self):
        pool = {'round': 0, 'server': {'proxy': graph()}}
        client = make_client(pool)
        pool['server']['weight'] = snapshot(client.task.model)
        client.execute()
        client.send_message()
        payload = pool['client_0']
        client.task.model.eval()
        expected = client.task.model(pool['server']['proxy'])[0].mean(0).detach().numpy()
        np.testing.assert_allclose(payload['functional_embedding'], expected)
        self.assertEqual(payload['num_samples'], 6)
        uploaded = copy.deepcopy(payload['weight'])
        with torch.no_grad():
            next(client.task.model.parameters()).add_(1)
        for name, value in uploaded.items():
            torch.testing.assert_close(payload['weight'][name], value)
            self.assertFalse(payload['weight'][name].requires_grad)

    def test_proxy_matches_original_random_partition_and_gaussian_features(self):
        from openfgl.flcore.fedpub.server import FedPubServer
        from openfgl.flcore.fedpub.fedpub_config import config as original_config
        from openfgl.flcore.multimodal_fedpub.fedpub_config import config
        pool = {'round': 0, 'sampled_clients': [0]}
        with patch.dict(config, n_proxy=2), patch.dict(original_config, n_proxy=2):
            server = make_server(pool)
            torch.manual_seed(38)
            expected = FedPubServer.get_proxy_data(server, 7)
            torch.manual_seed(38)
            actual = server.get_proxy_data((3, 4))
        torch.testing.assert_close(actual.edge_index, expected.edge_index)
        torch.testing.assert_close(actual.x, expected.x)
        torch.testing.assert_close(actual.x_image, actual.x[:, :4])
        torch.testing.assert_close(actual.x_text, actual.x[:, 4:])
        self.assertIsNone(actual.y)

    def test_aggregation_numerically_matches_original_fedpub(self):
        from openfgl.flcore.fedpub.server import FedPubServer
        from openfgl.flcore.fedpub.fedpub_config import config as original_config
        from openfgl.flcore.multimodal_fedpub.fedpub_config import config
        for normalization in ('exp', 'none'):
            pool = {'round': 0, 'sampled_clients': [0, 2]}
            server = make_server(pool)
            for cid, value, embedding in ((0, .2, [1., .2]), (2, .8, [.1, 1.])):
                pool[f'client_{cid}'] = dict(num_samples=cid + 1,
                    weight={k: torch.full_like(v, value) for k, v in server.task.model.state_dict().items()},
                    functional_embedding=np.array(embedding))
            original = FedPubServer.__new__(FedPubServer)
            original.message_pool, original.task = copy.deepcopy(pool), copy.deepcopy(server.task)
            with patch.dict(config, agg_norm=normalization), patch.dict(original_config, agg_norm=normalization):
                server.execute()
                original.execute()
            for name, value in server.task.model.state_dict().items():
                torch.testing.assert_close(value, original.task.model.state_dict()[name])
            for index, cid in enumerate(pool['sampled_clients']):
                for name, value in server.update_weights[cid].items():
                    torch.testing.assert_close(value, original.update_weights[index][name])

    def test_partial_participation_routes_weights_by_id_and_falls_back_to_global(self):
        pool = {'round': 0, 'sampled_clients': [0, 2]}
        server = make_server(pool)
        server.send_message()
        for cid in (0, 2):
            pool[f'client_{cid}'] = dict(num_samples=1,
                weight={k: torch.full_like(v, cid + .1) for k, v in server.task.model.state_dict().items()},
                functional_embedding=np.array([1., cid]))
        server.execute()
        previous = copy.deepcopy(server.update_weights)
        pool.update(round=1, sampled_clients=[1, 2])
        server.send_message()
        self.assertNotIn('personalized_1', pool['server'])
        self.assertNotIn('personalized_0', pool['server'])
        for name, value in previous[2].items():
            torch.testing.assert_close(pool['server']['personalized_2'][name], value)
        newcomer = make_client(pool, cid=1)
        newcomer.execute()
        for name, value in newcomer.task.model.state_dict().items():
            if 'mask' not in name:
                torch.testing.assert_close(value, pool['server']['weight'][name])

    def test_zero_functional_embeddings_produce_finite_aggregation(self):
        pool = {'round': 0, 'sampled_clients': [0, 1]}
        server = make_server(pool)
        for cid in (0, 1):
            pool[f'client_{cid}'] = dict(num_samples=1, weight=snapshot(server.task.model),
                                       functional_embedding=np.zeros(10))
        server.execute()
        for weights in server.update_weights.values():
            self.assertTrue(all(torch.isfinite(value).all() for value in weights.values()))


if __name__ == '__main__':
    unittest.main()
