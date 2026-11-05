import torch
from torch_geometric.data import Batch
from copy import deepcopy


def get_coordinated_data(datalist, cross_link=1, dynamic_edge='none',
                         dynamic_prune=0.5, cross_link_ablation=False, device=None):
    """
    支持动态边计算的协调器连接代码

    参数：
    - datalist: 输入图列表（List[Data]）
    - cross_link: 每个图的协调器数量
    - dynamic_edge: 边模式 [none, internal_external, similarity]
    - dynamic_prune: 相似性阈值（0~1）
    - cross_link_ablation: 是否禁用协调器间连接（用于消融实验）

    返回：
    - coordinated_data: 整合后的全局图
    """
    # 合并原始数据
    data = Batch.from_data_list(datalist)
    data_for_simi = deepcopy(data)  # 用于相似性计算的副本
    num_graphs = data.num_graphs

    # 添加协调器节点
    new_index_list = [i for i in range(num_graphs)] * cross_link
    new_node_features = torch.ones((len(new_index_list), data.num_node_features)).to(device)
    data.x = torch.cat([data.x, new_node_features], dim=0)
    data.batch = torch.cat([data.batch, torch.tensor(new_index_list, device= data.batch.device)], dim=0)

    # 获取协调器节点索引
    coord_indices = (data.batch >= 0).nonzero()[-len(new_index_list):].flatten()

    # 模式1：协调器与原图全连接（基础逻辑）
    if dynamic_edge in ['none', 'similarity']:
        for graph_idx in range(num_graphs):
            orig_nodes = (data.batch == graph_idx).nonzero()[:-cross_link].flatten()
            coords = (data.batch == graph_idx).nonzero()[-cross_link:].flatten()

            # 双向全连接
            senders = torch.repeat_interleave(coords, len(orig_nodes))
            receivers = orig_nodes.repeat(len(coords))
            edges = torch.stack([senders, receivers], dim=0)
            data.edge_index = torch.cat([data.edge_index, edges], dim=1)

    # 模式2：动态调整内外边（internal_external）
    elif dynamic_edge == 'internal_external':
        # 计算全局相似性矩阵
        sim_matrix = torch.sigmoid(torch.mm(data.x, data.x.t()))
        adj_mask = (sim_matrix > dynamic_prune).float()

        # 仅保留协调器相关边
        coord_mask = torch.isin(torch.arange(data.num_nodes), coord_indices)
        filtered_edges = (adj_mask * coord_mask.unsqueeze(1)).nonzero().t()
        data.edge_index = torch.cat([data.edge_index, filtered_edges], dim=1)

    # 协调器间连接（非消融模式下）
    if not cross_link_ablation:
        if dynamic_edge == 'similarity':
            # 计算图级特征均值
            graph_feats = []
            for i in range(num_graphs):
                nodes = (data_for_simi.batch == i).nonzero().flatten()
                graph_feats.append(data_for_simi.x[nodes].mean(dim=0))
            graph_feats = torch.stack(graph_feats)

            # 生成图间相似性边
            sim_matrix = torch.sigmoid(torch.mm(graph_feats, graph_feats.t()))
            adj_mask = (sim_matrix > dynamic_prune).float()
            graph_edges = adj_mask.nonzero().t()

            # 映射到协调器节点
            coord_edges = []
            for i, j in graph_edges.t():
                src = coord_indices[i * cross_link: (i + 1) * cross_link]
                tgt = coord_indices[j * cross_link: (j + 1) * cross_link]
                coord_edges.append(torch.stack([src.repeat(len(tgt)), tgt.repeat_interleave(len(src))]))
            coord_edges = torch.cat(coord_edges, dim=1)
            data.edge_index = torch.cat([data.edge_index, coord_edges], dim=1)

        else:  # none/internal_external模式使用全连接
            senders, receivers = torch.meshgrid(coord_indices, coord_indices)
            data.edge_index = torch.cat([data.edge_index, torch.stack([senders.flatten(), receivers.flatten()])], dim=1)

    return data