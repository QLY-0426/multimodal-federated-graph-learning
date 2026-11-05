import unittest
from types import SimpleNamespace

import torch

from openfgl.utils.basic_utils import total_size
from openfgl.utils.logger import Logger


class CommunicationTests(unittest.TestCase):
    def test_each_sampled_client_downloads_peer_logits_and_new_graphs(self):
        args = SimpleNamespace(debug=True, comm_cost=True, log_root='/tmp',
                               log_name='unused', fl_algorithm='multimodal_fedgvd')
        targets = {2: torch.ones(3, 4), 7: torch.ones(5, 4), 8: torch.ones(2, 4)}
        pool = {'sampled_clients': [2, 7], 'client_2': {}, 'client_7': {},
                'server': {'syn_logits': targets, 'graphs': {
                    2: {7: {'x_text': torch.ones(5, 6)}}, 7: {}}}}
        logger = Logger(args, pool, '/tmp')
        logger.add_log({'current_round': 1})
        expected = total_size(pool['client_2']) + total_size(pool['client_7'])
        expected += total_size(targets[7]) + total_size(targets[8])
        expected += total_size(targets[2]) + total_size(targets[8])
        expected += total_size(pool['server']['graphs'][2]) + total_size(pool['server']['graphs'][7])
        self.assertEqual(logger.comm_cost, [expected])


if __name__ == '__main__':
    unittest.main()
