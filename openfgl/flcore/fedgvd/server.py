import torch
from openfgl.flcore.base import BaseServer
from .virtual_nodes import SubgraphIntegrators
from .fedgvd_config import config
from torch_geometric.data import Data
from torch import nn


class FedGVDServer(BaseServer):

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGVDServer, self).__init__(args, global_data, data_dir, message_pool, device)


    def execute(self):
        if self.message_pool["round"] == 0:
            self.global_data = self._aggregate_subgraphs()
        self._train_global_model(self.global_data)
        self._calculate_client_logits(self.global_data)

    def send_message(self):
        pass

    def _aggregate_subgraphs(self):
        datalist = []
        self.graph_client_ids = sorted(self.message_pool["sampled_clients"])
        for client_id in self.graph_client_ids:
            client_data = self.message_pool[f"client_{client_id}"]

            x = torch.tensor(client_data["feat_syn"], dtype=torch.float) \
                if not isinstance(client_data["feat_syn"], torch.Tensor) \
                else client_data["feat_syn"].float()

            edge_index = torch.tensor(client_data["edge_index_syn"], dtype=torch.long) \
                if not isinstance(client_data["edge_index_syn"], torch.Tensor) \
                else client_data["edge_index_syn"].long()

            if edge_index.dim() == 1:
                edge_index = edge_index.view(2, -1)
            elif edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()

            data = Data(x=x, edge_index=edge_index)

            if "label_syn" in client_data:
                y = torch.tensor(client_data["label_syn"], dtype=torch.long) \
                    if not isinstance(client_data["label_syn"], torch.Tensor) \
                    else client_data["label_syn"]
                data.y = y.long().view(-1)

            datalist.append(data.to(self.device))

        self.integrators = SubgraphIntegrators(
            datalist, hidden_dim=getattr(getattr(self, 'args', None), 'hid_dim', 128),
            top_k=config['virtual_top_k'], temperature=config['virtual_temperature'])
        with torch.no_grad():
            return self.integrators.build_graph()

    def _train_global_model(self, global_data):
        self.task.model.train()
        self.integrators.train()
        if not getattr(self, '_integrators_registered', False):
            self.task.optim.add_param_group({'params': list(self.integrators.parameters())})
            self._integrators_registered = True

        for _ in range(10):
            self.task.optim.zero_grad()
            global_data = self.integrators.build_graph()
            embeddings, logits = self.task.model(global_data)

            loss = self._calculate_loss(global_data, logits)

            loss.backward()
            self.task.optim.step()
        # Distillation uses the graph after the last parameter update.
        with torch.no_grad():
            self.global_data = self.integrators.build_graph()

    def _calculate_loss(self, data, logits):
        n = self.integrators.num_original
        targets = self.integrators.class_targets(logits.size(1))
        classification = nn.CrossEntropyLoss()(logits[n:], targets)
        contrastive = self.integrators.contrastive_loss(data.x[n:])
        return classification + contrastive

    def _calculate_client_logits(self, global_data):
        self.task.model.eval()
        with torch.no_grad():
            _, all_logits = self.task.model(global_data)

            # Ownership follows the fixed graph layout, not round sampling order.
            client_logits = {}
            ptr = 0
            for cid in self.graph_client_ids:
                num_nodes = self.message_pool[f"client_{cid}"]["feat_syn"].shape[0]
                client_logits[cid] = all_logits[ptr:ptr + num_nodes]
                ptr += num_nodes

            for cid in self.graph_client_ids:
                other_logits = [client_logits[other] for other in self.graph_client_ids
                                if other != cid]
                target = torch.cat(other_logits) if other_logits else all_logits[:0]
                self.message_pool[f"client_{cid}"]["syn_logits"] = target.detach().cpu()
