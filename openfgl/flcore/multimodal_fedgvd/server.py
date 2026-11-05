"""Train on cached synthetic modal graphs and return per-client logits."""
import math
import torch

from openfgl.flcore.base import BaseServer
from openfgl.model.multimodal_utils import multimodal_supervised_loss
from .contrastive import get_coordinated_data
from .fedgvd_config import config
from .utils import graph_from_payload, graph_to_payload
from .virtual import VirtualGraphBuilder, integrator_contrastive_loss, virtual_classification_loss
from .paper_privacy import retention_probability, randomized_mismatch


class MultimodalFedGVDServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.graph_cache = {}
        self.global_data = None
        self.client_slices = {}
        self.syn_logits = {}
        self.mode = getattr(args, 'multimodal_server_mode', 'dual_knn')
        if self.mode not in ('dual_knn', 'dual_virtual'):
            raise ValueError('unsupported multimodal server mode')
        self.integrator = None
        self.diagnostics = []
        self.logit_protection = getattr(args, 'multimodal_logit_protection', 'none')
        if self.logit_protection not in ('none', 'paper_mismatch'):
            raise ValueError('unsupported logit protection mode')
        self.mismatch_epsilon = getattr(args, 'multimodal_mismatch_epsilon', 1.)
        retention_probability(self.mismatch_epsilon, 1)
        mismatch_seed = getattr(args, 'experiment_training_seed', getattr(args, 'seed', 0))
        self.mismatch_generator = torch.Generator(device='cpu').manual_seed(mismatch_seed + 78531)
        self.protection_diagnostics = []
        self._protected_source = None
        self._protected_round = None
        self._protected_logits = {}
        self.virtual_ce_weight = getattr(args, 'multimodal_virtual_ce_weight', 1.)
        self.contrastive_weight = getattr(args, 'multimodal_contrastive_weight', 1.)
        if any(not math.isfinite(w) or w < 0 for w in (self.virtual_ce_weight, self.contrastive_weight)):
            raise ValueError('virtual loss weights must be finite and nonnegative')

    def execute(self):
        changed = False
        for cid in self.message_pool['sampled_clients']:
            message = self.message_pool.get(f'client_{cid}', {})
            if 'graph' in message:
                self.graph_cache[cid] = graph_from_payload(message['graph'], self.device)
                changed = True
        if not self.graph_cache:
            return
        if changed:
            base, self.client_slices = get_coordinated_data(
                self.graph_cache,
                cross_k=getattr(self.args, 'multimodal_cross_k', 2) if self.mode == 'dual_knn' else 0,
                similarity_threshold=getattr(self.args, 'multimodal_similarity_threshold', 0.0))
            if self.mode == 'dual_virtual':
                if self.integrator is None:
                    # Preserve the training RNG stream while initializing new parameters.
                    device = base.x_text.device
                    devices = [device.index] if device.type == 'cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        with torch.no_grad():
                            classes = self.task.model.forward_modalities(base)[1].size(-1)
                        self.integrator = VirtualGraphBuilder(
                            base, self.client_slices, classes, getattr(self.args, 'hid_dim', 128),
                            getattr(self.args, 'multimodal_cross_k', 2)).to(device)
                    self.task.optim.add_param_group({'params': self.integrator.parameters()})
                else:
                    self.integrator.update_graph(base, self.client_slices)
            self.global_data = base
        if self.mode == 'dual_virtual':
            self._train_virtual()
            return
        data = self.global_data
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=self.device)
        self.task.model.train()
        for _ in range(getattr(self.args, 'multimodal_server_epochs', 10)):
            self.task.optim.zero_grad()
            _, logits, modal_logits, _ = self.task.model.forward_modalities(data)
            loss = multimodal_supervised_loss(
                modal_logits, logits, data.y, mask,
                modality_mask=data.modality_mask, branch_weight=config['branch_weight'])
            loss.backward()
            self.task.optim.step()
        self.task.model.eval()
        with torch.no_grad():
            _, logits, _, _ = self.task.model.forward_modalities(data)
        self.syn_logits = {cid: logits[part].detach().cpu().clone()
                           for cid, part in self.client_slices.items()}

    def _train_virtual(self):
        self.task.model.train()
        previous_edges = None
        topology_changes = 0
        for _ in range(getattr(self.args, 'multimodal_server_epochs', 10)):
            self.task.optim.zero_grad()
            data = self.integrator()
            edges = tuple(getattr(data, 'edge_index_' + m) for m in ('text', 'image'))
            if previous_edges is not None:
                topology_changes += int(any(not torch.equal(a, b) for a, b in zip(previous_edges, edges)))
            previous_edges = edges
            _, logits, modal_logits, _ = self.task.model.forward_modalities(data)
            supervised = multimodal_supervised_loss(
                modal_logits, logits, data.y, data.synthetic_mask,
                modality_mask=data.modality_mask, branch_weight=config['branch_weight'])
            virtual_ce = virtual_classification_loss(data, logits, modal_logits, config['branch_weight'])
            contrastive = integrator_contrastive_loss(
                data, getattr(self.args, 'multimodal_contrastive_temperature', .5),
                getattr(self.args, 'multimodal_contrastive_mode', 'paper'))
            loss = supervised + self.virtual_ce_weight * virtual_ce + self.contrastive_weight * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite virtual-node server loss')
            loss.backward()
            query_grad = sum(float(p.grad.detach().norm()) for p in self.integrator.queries if p.grad is not None)
            self.task.optim.step()
        self.task.model.eval()
        with torch.no_grad():
            self.global_data = self.integrator()
            _, logits, _, _ = self.task.model.forward_modalities(self.global_data)
        self.syn_logits = {cid: logits[part].detach().cpu().clone()
                           for cid, part in self.client_slices.items()}
        self.diagnostics.append({'round': self.message_pool.get('round', len(self.diagnostics)),
            'supervised': float(supervised.detach()), 'virtual_ce': float(virtual_ce.detach()),
            'contrastive': float(contrastive.detach()), 'total': float(loss.detach()),
            'query_grad_norm': query_grad, 'topology_changes': topology_changes,
            'virtual_nodes': len(self.client_slices), 'synthetic_nodes': data.virtual_start,
            'text_edges': self.global_data.edge_index_text.size(1),
            'image_edges': self.global_data.edge_index_image.size(1)})

    def send_message(self):
        graphs = {}
        for recipient in self.message_pool.get('sampled_clients', []):
            known = self.message_pool.get(f'client_{recipient}', {}).get('known_graph_ids', [])
            graphs[recipient] = {
                cid: graph_to_payload(graph)
                for cid, graph in self.graph_cache.items()
                if cid != recipient and cid not in known
            }
        logits = self.syn_logits
        if self.logit_protection == 'paper_mismatch' and logits:
            round_id = self.message_pool.get('round')
            if self._protected_source is not logits or self._protected_round != round_id:
                self._protected_logits = {}
                for cid in sorted(logits):
                    protected, info = randomized_mismatch(logits[cid], self.mismatch_epsilon, self.mismatch_generator)
                    self._protected_logits[cid] = protected
                    self.protection_diagnostics.append({'round': round_id, 'source_client': cid, **info})
                self._protected_source, self._protected_round = logits, round_id
            logits = self._protected_logits
        self.message_pool['server'] = {'syn_logits': logits, 'graphs': graphs}
