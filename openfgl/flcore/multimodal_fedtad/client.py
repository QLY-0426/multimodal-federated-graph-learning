from openfgl.flcore.fedtad.client import FedTADClient
from ._utils import class_reliability
from .fedtad_config import config


class MultimodalFedTADClient(FedTADClient):
    """Reuse FedTAD's parameter download, local CE training and model upload."""

    def fedtad_initialization(self):
        ckr_root = os.path.join(self.task.data_dir, 'ckr_multimodal_fedtad')
        ckr_filename = os.path.join(ckr_root, f'client_{self.client_id}.pt')
        if os.path.exists(ckr_filename):
            ckr = torch.load(ckr_filename).to(self.device)
        else:
            ckr = class_reliability(self.task.data, self.task.train_mask,
                                    self.task.num_global_classes, config['max_walk_length'])
            if config['save_ckr']:
                os.makedirs(ckr_root, exist_ok=True)
                torch.save(ckr, ckr_filename)
        self.ckr = ckr
import os
import torch

