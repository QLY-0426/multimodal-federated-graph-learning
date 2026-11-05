"""Public random proxy graphs and functional-similarity personalized averaging."""
import networkx as nx
import numpy as np
import torch

from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedpub.server import FedPubServer, from_networkx
from .fedpub_config import config


class MultimodalFedPubServer(FedPubServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        # BaseTask's loader constructs the dual masked model on both sides.
        BaseServer.__init__(self, args, global_data, data_dir, message_pool, device, personalized=True)
        self.proxy = self.get_proxy_data(self.task.model.modality_dims).to(device)
        self.update_weights = {}

    @torch.no_grad()
    def execute(self):
        clients = self.message_pool['sampled_clients']
        if not clients:
            raise ValueError('FED-PUB requires at least one sampled client')
        messages = [self.message_pool[f'client_{cid}'] for cid in clients]
        local_weights = [message['weight'] for message in messages]
        embeddings = np.asarray([message['functional_embedding'] for message in messages], dtype=np.float64)
        if not np.isfinite(embeddings).all():
            raise ValueError('FED-PUB functional embeddings must be finite')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
        similarity = np.clip(normalized @ normalized.T, -1., 1.)
        if config['agg_norm'] == 'exp':
            scores = config['norm_scale'] * similarity
            # Row shift preserves the original normalized exponential exactly in algebra.
            similarity = np.exp(scores - scores.max(axis=1, keepdims=True))
        row_sums = similarity.sum(axis=1, keepdims=True)
        ratios = np.divide(similarity, row_sums,
            out=np.full_like(similarity, 1. / len(clients)), where=np.abs(row_sums) > 1e-12)
        samples = np.asarray([message['num_samples'] for message in messages], dtype=np.float64)
        if not np.isfinite(samples).all() or (samples < 0).any() or samples.sum() <= 0:
            raise ValueError('FED-PUB requires a positive total sample count')
        self.task.model.load_state_dict(self.aggregate(local_weights, samples / samples.sum()))
        for index, cid in enumerate(clients):
            self.update_weights[cid] = self.aggregate(local_weights, ratios[index])

    def send_message(self):
        message = {
            'weight': {name: value.detach().clone() for name, value in self.task.model.state_dict().items()},
            'proxy': self.proxy,
        }
        if self.message_pool['round'] > 0:
            for cid in self.message_pool['sampled_clients']:
                if cid in self.update_weights:
                    message[f'personalized_{cid}'] = {
                        name: value.detach().clone() for name, value in self.update_weights[cid].items()
                    }
        self.message_pool['server'] = message

    def get_proxy_data(self, modality_dims):
        text_dim, image_dim = modality_dims
        # Same topology, N(0,1) features and fixed public proxy across rounds as FED-PUB.
        num_nodes = 100
        graph = nx.random_partition_graph([num_nodes] * config['n_proxy'], p_in=.1, p_out=0,
                                          seed=self.args.seed)
        data = from_networkx(graph)
        data.x = torch.normal(mean=0, std=1,
                              size=(num_nodes * config['n_proxy'], text_dim + image_dim))
        data.x_image, data.x_text = data.x.split((image_dim, text_dim), dim=1)
        return data
