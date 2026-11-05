import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from openfgl.flcore.multimodal_fedgvd.client import MultimodalFedGVDClient
from openfgl.flcore.multimodal_fedgvd.server import MultimodalFedGVDServer
from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload


def graph():
    return Data(x=torch.randn(3, 4), x_text=torch.randn(3, 2),
                x_image=torch.randn(3, 2), y=torch.tensor([0, 1, 0]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.ones(2), modality_mask=torch.ones(3, 2, dtype=torch.bool))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward_modalities(self, data):
        logits = self.linear(data.x)
        return data.x, logits, torch.stack((logits, logits), 1), logits.new_full((len(logits), 2), .5)


def task():
    data = graph()
    model = ToyModel()
    split = {'data': data, 'train_mask': torch.tensor([True, True, False]),
             'val_mask': torch.tensor([False, False, True]), 'test_mask': torch.zeros(3, dtype=torch.bool)}
    return SimpleNamespace(model=model, optim=torch.optim.SGD(model.parameters(), lr=.1),
                           processed_data=split, splitted_data=split, data=data, num_samples=3)


class FederationTests(unittest.TestCase):
    def test_round_trip_with_late_participation(self):
        pool = {}
        args = SimpleNamespace(num_epochs=1, multimodal_server_epochs=1)
        clients = {}
        for cid in (2, 4, 7):
            local_task = task()
            def initialize(obj, args, client_id, data, data_dir, message_pool, device):
                obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, client_id, message_pool, device, local_task
            with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
                clients[cid] = MultimodalFedGVDClient(args, cid, None, '', pool, 'cpu')
        server_task = task()
        def initialize_server(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, server_task
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize_server):
            server = MultimodalFedGVDServer(args, None, '', pool, 'cpu')
        for selected in ([4, 7], [4, 7], [2, 4], [2, 4, 7], [2, 4, 7]):
            pool['sampled_clients'] = selected
            server.send_message()
            for cid in selected:
                with patch('openfgl.flcore.multimodal_fedgvd.client.data_gc', return_value=graph()):
                    clients[cid].execute()
                clients[cid].send_message()
            server.execute()
        for cid, client in clients.items():
            self.assertEqual(set(client.peer_graphs), {2, 4, 7} - {cid})
            self.assertEqual(client.task.processed_data['data'].num_nodes, 9)
            self.assertEqual(client.task.splitted_data['data'].num_nodes, 3)
            self.assertNotIn('graph', pool[f'client_{cid}'])
        server.send_message()
        self.assertTrue(all(not value for value in pool['server']['graphs'].values()))

    def test_client_participation_and_evaluation_isolation(self):
        peer = graph()
        pool = {'round': 12, 'server': {
            'syn_logits': {4: torch.full((3, 2), 99.), 7: torch.zeros(3, 2)},
            'graphs': {4: {7: graph_to_payload(peer)}}}}
        local_task = task()
        original = local_task.splitted_data
        def initialize(obj, args, client_id, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, client_id, message_pool, device, local_task
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            client = MultimodalFedGVDClient(SimpleNamespace(num_epochs=1), 4, None, '', pool, 'cpu')
        with patch('openfgl.flcore.multimodal_fedgvd.client.data_gc', return_value=graph()) as condense:
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_not_called()
            client.send_message()
            self.assertIn('graph', pool['client_4'])
            before = local_task.model.linear.weight.detach().clone()
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_not_called()
            client.send_message()
            self.assertNotIn('graph', pool['client_4'])
            self.assertEqual(pool['client_4']['known_graph_ids'], [7])
            torch.testing.assert_close(local_task.processed_data['data'].x_text[3:], peer.x_text)
            pool['server']['graphs'] = {}
            self.assertFalse(torch.equal(before, local_task.model.linear.weight))
            with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
                client.execute()
                mse.assert_called_once()
                torch.testing.assert_close(mse.call_args.args[1], pool['server']['syn_logits'][7])
            self.assertEqual(condense.call_count, 1)
        self.assertIs(local_task.splitted_data, original)
        self.assertEqual(original['data'].num_nodes, 3)
        self.assertEqual(local_task.processed_data['data'].num_nodes, 6)

    def test_server_caches_clients_and_returns_detached_slices(self):
        from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload
        pool = {'sampled_clients': [7], 'client_7': {'graph': graph_to_payload(graph())}}
        server_task = task()
        def initialize(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, server_task
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize):
            server = MultimodalFedGVDServer(SimpleNamespace(multimodal_server_epochs=1), None, '', pool, 'cpu')
        server.send_message()
        self.assertEqual(pool['server']['syn_logits'], {})
        server.execute()
        pool.update(sampled_clients=[2], client_2={'graph': graph_to_payload(graph())})
        server.execute()
        server.send_message()
        with torch.no_grad():
            expected = server_task.model.forward_modalities(server.global_data)[1]
        for cid in (2, 7):
            self.assertEqual(tuple(pool['server']['syn_logits'][cid].shape), (3, 2))
            self.assertFalse(pool['server']['syn_logits'][cid].requires_grad)
            torch.testing.assert_close(pool['server']['syn_logits'][cid], expected[server.client_slices[cid]])
        self.assertEqual(set(pool['server']), {'syn_logits', 'graphs'})
        self.assertEqual(set(pool['server']['graphs'][2]), {7})
        coordinated = server.global_data
        pool.update(sampled_clients=[7], client_7={})
        server.execute()
        self.assertIs(server.global_data, coordinated)
        self.assertEqual(set(server.graph_cache), {2, 7})

    def test_peer_order_and_late_arrival_rebuild_without_duplicates(self):
        local_task = task()
        original = local_task.processed_data
        peers = {cid: graph_to_payload(graph()) for cid in (7, 2)}
        targets = {cid: torch.full((3, 2), float(cid)) for cid in (4, 7, 2, 1)}
        pool = {'server': {'graphs': {4: peers}, 'syn_logits': targets}}
        def initialize(obj, args, client_id, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, client_id, message_pool, device, local_task
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            client = MultimodalFedGVDClient(SimpleNamespace(num_epochs=1), 4, None, '', pool, 'cpu')
        with patch('openfgl.flcore.multimodal_fedgvd.client.data_gc', return_value=graph()):
            client.execute()
        client.execute()
        for cid, start in ((2, 3), (7, 6)):
            torch.testing.assert_close(local_task.processed_data['data'].x_text[start:start+3], peers[cid]['x_text'])
        late = graph_to_payload(graph())
        pool['server']['graphs'] = {4: {1: late}}
        with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
            client.execute()
            torch.testing.assert_close(mse.call_args.args[1], torch.cat([targets[cid] for cid in (1, 2, 7)]))
        data = local_task.processed_data['data']
        self.assertEqual(data.num_nodes, 12)
        torch.testing.assert_close(data.x_text[3:6], late['x_text'])
        torch.testing.assert_close(data.edge_index[:, 2:4], late['edge_index'] + 3)
        client.execute()  # Retransmission cannot append the same graph twice.
        self.assertEqual(local_task.processed_data['data'].num_nodes, 12)
        self.assertEqual(original['data'].num_nodes, 3)
        self.assertEqual(local_task.processed_data['val_mask'].sum().item(), 1)

    def test_no_peers_trains_real_nodes_only(self):
        local_task = task()
        pool = {'server': {'graphs': {}, 'syn_logits': {4: torch.zeros(3, 2)}}}
        def initialize(obj, args, client_id, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, client_id, message_pool, device, local_task
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            client = MultimodalFedGVDClient(SimpleNamespace(num_epochs=1), 4, None, '', pool, 'cpu')
        with patch('openfgl.flcore.multimodal_fedgvd.client.data_gc', return_value=graph()):
            client.execute()
        before = local_task.model.linear.weight.detach().clone()
        with patch('openfgl.flcore.multimodal_fedgvd.client.F.mse_loss', wraps=torch.nn.functional.mse_loss) as mse:
            client.execute()
            client.execute()
            mse.assert_not_called()
        self.assertEqual(local_task.processed_data['data'].num_nodes, 3)
        self.assertFalse(torch.equal(before, local_task.model.linear.weight))
        self.assertTrue(torch.isfinite(local_task.model.linear.weight).all())

    def test_server_sends_only_unacknowledged_peer_graphs(self):
        server_task = task()
        pool = {'sampled_clients': [4, 7], 'client_4': {'known_graph_ids': [7]}, 'client_7': {}}
        def initialize(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, server_task
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize):
            server = MultimodalFedGVDServer(SimpleNamespace(), None, '', pool, 'cpu')
        server.graph_cache = {cid: graph() for cid in (2, 4, 7)}
        server.send_message()
        messages = pool['server']['graphs']
        self.assertEqual(set(messages[4]), {2})
        self.assertEqual(set(messages[7]), {2, 4})
        pool['client_4']['known_graph_ids'] = [2, 7]
        pool['client_7']['known_graph_ids'] = [2, 4]
        server.send_message()
        self.assertFalse(pool['server']['graphs'][4])
        self.assertFalse(pool['server']['graphs'][7])


if __name__ == '__main__':
    unittest.main()
