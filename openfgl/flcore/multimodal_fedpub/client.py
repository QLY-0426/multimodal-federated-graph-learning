"""FED-PUB local personalization, regularization and proxy functional embeddings."""
import torch
from torch.nn import functional as F

from openfgl.flcore.base import BaseClient
from .fedpub_config import config


class MultimodalFedPubClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.prev_w = None

    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            loss = F.cross_entropy(logits[mask], label[mask])
            # state_dict() detaches parameters; use live parameters for gradients.
            for name, parameter in self.task.model.named_parameters():
                if 'mask' in name:
                    loss = loss + config['l1'] * torch.norm(parameter.float(), 1)
                elif ('conv' in name or 'clsif' in name) and self.message_pool['round'] > 0:
                    if self.prev_w is not None:
                        loss = loss + config['loc_l2'] * torch.norm(parameter.float() - self.prev_w[name], 2)
            return loss
        return custom_loss_fn

    def execute(self):
        message = self.message_pool['server']
        incoming = message.get(f'personalized_{self.client_id}', message['weight'])
        # Copy both the anchor and the download; never rewrite the server message.
        self.prev_w = {name: value.detach().to(self.device).clone() for name, value in incoming.items()}
        local = self.task.model.state_dict()
        downloaded = {
            name: local[name] if any(token in name for token in ('mask', 'pre', 'pos', 'running', 'tracked')) else value
            for name, value in self.prev_w.items()
        }
        self.task.model.load_state_dict(downloaded)
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()

    @torch.no_grad()
    def get_functional_embedding(self):
        was_training = self.task.model.training
        self.task.model.eval()
        try:
            proxy = self.message_pool['server']['proxy'].to(self.device)
            embedding, _ = self.task.model(proxy)
            # Concatenation retains both modalities in the cosine-similarity input.
            return embedding.mean(dim=0).detach().cpu().numpy()
        finally:
            self.task.model.train(was_training)

    def send_message(self):
        self.message_pool[f'client_{self.client_id}'] = {
            'num_samples': self.task.num_samples,
            'weight': {name: value.detach().clone() for name, value in self.task.model.state_dict().items()},
            'functional_embedding': self.get_functional_embedding(),
        }
