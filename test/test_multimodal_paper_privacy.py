import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from openfgl.flcore.multimodal_fedgvd.paper_privacy import (
    retention_probability, randomized_mismatch, prototype_filter, normalized_kl)


class PaperPrivacyTests(unittest.TestCase):
    def test_cli_defaults_select_complete_paper_mechanism(self):
        from openfgl.config import parser
        defaults = parser.parse_args([])
        self.assertEqual(defaults.multimodal_logit_protection, 'paper_mismatch')
        self.assertTrue(defaults.multimodal_prototype_filter)
        self.assertEqual(defaults.multimodal_distill_loss, 'normalized_kl')
        self.assertEqual(defaults.multimodal_mismatch_epsilon, 1.)
        self.assertEqual(defaults.multimodal_prototype_quantile, .95)
        self.assertEqual(defaults.multimodal_distill_temperature, 1.)
        self.assertEqual(defaults.multimodal_server_mode, 'dual_virtual')
        self.assertEqual(defaults.multimodal_virtual_ce_weight, 1.)
        self.assertEqual(defaults.multimodal_contrastive_weight, 1.)
        legacy = parser.parse_args(['--multimodal_logit_protection', 'none',
            '--no_multimodal_prototype_filter', '--multimodal_distill_loss', 'mse'])
        self.assertFalse(legacy.multimodal_prototype_filter)

    def test_retention_formula_and_validation(self):
        self.assertEqual(retention_probability(0, 20), 0)
        self.assertAlmostEqual(retention_probability(1, 20), (math.e-1)/(math.e-1+20))
        self.assertEqual(retention_probability(1000, 20), 1)
        for eps in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                retention_probability(eps, 20)

    def test_permutation_preserves_values_not_guaranteed_dp(self):
        x = torch.arange(5.).repeat(20000, 1)
        original = x.clone()
        state = torch.get_rng_state().clone()
        y, info = randomized_mismatch(x, 1., torch.Generator().manual_seed(14))
        torch.testing.assert_close(x, original)
        torch.testing.assert_close(y.sort(1).values, x)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        beta = retention_probability(1, 5)
        self.assertAlmostEqual(info['selected_rows']/len(x), 1-beta, delta=.015)
        # Uniform permutation includes identity and has 1/C chance to retain argmax.
        keep = (y.argmax(1) == x.argmax(1)).float().mean().item()
        self.assertAlmostEqual(keep, beta + (1-beta)/5, delta=.015)

    def test_one_class_and_empty(self):
        x = torch.ones(3, 1)
        y, info = randomized_mismatch(x, 0, torch.Generator().manual_seed(0))
        torch.testing.assert_close(y, x)
        self.assertEqual(info['changed_rows'], 0)
        y, info = randomized_mismatch(torch.empty(0, 2), 1, torch.Generator())
        self.assertEqual(y.shape, (0, 2))

    def test_prototypes_use_only_real_training_labels(self):
        h = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[-1.,0.],[1.,0.],[0.,1.],[-1.,0.]])
        labels = torch.tensor([0,0,1,1,0,0,0])
        train = torch.tensor([True,True,True,False,False,False,False])
        teacher = torch.tensor([[3.,0.],[0.,3.],[3.,0.]])
        keep, info = prototype_filter(h, labels, train, 4, teacher, .95)
        self.assertEqual(keep.tolist(), [True,True,False])
        self.assertAlmostEqual(info['threshold'], 0., places=6)
        labels[3:] = 0
        keep2, _ = prototype_filter(h, labels, train, 4, teacher, .95)
        self.assertTrue(torch.equal(keep, keep2))

    def test_missing_class_and_zero_embeddings_reject(self):
        h = torch.tensor([[1.,0.],[0.,0.],[0.,1.],[0.,0.]])
        train = torch.tensor([True,True,False,False])
        keep, _ = prototype_filter(h, torch.zeros(4,dtype=torch.long), train, 2,
                                   torch.tensor([[0.,1.],[1.,0.]]), .95)
        self.assertFalse(keep.any())

    def test_filter_rejects_invalid_quantile(self):
        with self.assertRaises(ValueError):
            prototype_filter(torch.ones(2,2), torch.zeros(2,dtype=torch.long),
                             torch.tensor([True,False]), 1, torch.ones(1,2), 1.1)

    def test_kl_direction_normalization_and_empty_backward(self):
        student = torch.tensor([[1.,2.,4.],[3.,1.,2.]], requires_grad=True)
        teacher = torch.tensor([[2.,1.,3.],[1.,5.,2.]], requires_grad=True)
        keep = torch.tensor([True,False])
        loss = normalized_kl(student, teacher, keep, 2.)
        s = (student[0]-student[0].mean())/student[0].std(unbiased=False)/2
        t = (teacher[0]-teacher[0].mean())/teacher[0].std(unbiased=False)/2
        p = t.softmax(0)
        torch.testing.assert_close(loss, (p*(p.log()-s.log_softmax(0))).sum())
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[1].abs().sum(), 0)
        student.grad = None
        zero = normalized_kl(student, teacher, torch.zeros(2,dtype=torch.bool), 1.)
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertEqual(student.grad.abs().sum(), 0)

    def test_server_broadcast_keeps_clean_logits_and_reuses_same_draw(self):
        from openfgl.flcore.multimodal_fedgvd.server import MultimodalFedGVDServer
        from test_multimodal_federation import task
        pool = {'round': 3, 'sampled_clients': []}
        def initialize(obj, args, data, data_dir, message_pool, device):
            obj.args, obj.message_pool, obj.device, obj.task = args, message_pool, device, task()
        with patch('openfgl.flcore.base.BaseServer.__init__', initialize):
            server = MultimodalFedGVDServer(SimpleNamespace(multimodal_logit_protection='paper_mismatch',
                multimodal_mismatch_epsilon=0, seed=42), None, '', pool, 'cpu')
        original = torch.arange(4.).repeat(100,1)
        server.syn_logits = {0: original.clone()}
        server.send_message()
        first = pool['server']['syn_logits'][0].clone()
        self.assertFalse(torch.equal(first, original))
        torch.testing.assert_close(server.syn_logits[0], original)
        server.send_message()
        torch.testing.assert_close(first, pool['server']['syn_logits'][0])
        self.assertEqual(len(server.protection_diagnostics), 1)
        pool['round'] = 4
        server.send_message()
        self.assertFalse(torch.equal(first, pool['server']['syn_logits'][0]))

    def test_client_filter_only_distillation_with_all_rejected(self):
        from openfgl.flcore.multimodal_fedgvd.client import MultimodalFedGVDClient
        from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload
        from test_multimodal_federation import task, graph
        local = task()
        pool = {'round':2, 'server':{'graphs':{0:{1:graph_to_payload(graph())}},
                                  'syn_logits':{1:torch.zeros(3,2)}}}
        def initialize(obj, args, cid, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, cid, message_pool, device, local
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            client = MultimodalFedGVDClient(SimpleNamespace(num_epochs=1, multimodal_prototype_filter=True,
                multimodal_distill_loss='normalized_kl'), 0, None, '', pool, 'cpu')
        client.synthetic, client.local_trainings = graph(), 1
        before = local.model.linear.weight.detach().clone()
        info = {'threshold':0.,'candidate_rows':3,'accepted_rows':0}
        with patch('openfgl.flcore.multimodal_fedgvd.client.prototype_filter',
                   return_value=(torch.zeros(3,dtype=torch.bool),info)) as filtering:
            client.execute()
        self.assertFalse(torch.equal(before, local.model.linear.weight))
        self.assertFalse(filtering.call_args.args[2][3:].any())
        self.assertEqual(client.distillation_diagnostics[0]['distillation_loss'], 0.)
        self.assertEqual(local.splitted_data['data'].num_nodes,3)
        client.send_message()
        self.assertEqual(set(pool['client_0']), {'num_samples','known_graph_ids'})

    def test_client_invalid_filter_configuration_fails_early(self):
        from openfgl.flcore.multimodal_fedgvd.client import MultimodalFedGVDClient
        from test_multimodal_federation import task
        def initialize(obj, args, cid, data, data_dir, message_pool, device):
            obj.args, obj.client_id, obj.message_pool, obj.device, obj.task = args, cid, message_pool, device, task()
        with patch('openfgl.flcore.base.BaseClient.__init__', initialize):
            with self.assertRaises(ValueError):
                MultimodalFedGVDClient(SimpleNamespace(multimodal_prototype_quantile=2.),0,None,'',{},'cpu')


if __name__ == '__main__':
    unittest.main()
