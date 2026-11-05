"""V4 feature probe: is the condensed graph's feature content still usable?

Read-only over V1 caches and raw data. No production source is modified.

Probe: train the same 32-dim MLP used by the V3 mlp_only anchor on
(a) condensed synthetic features (no edges at all),
(b) the same number of real training nodes (quota matched),
(c) the full real training set.

If (a) is close to (b)/(c), the synthetic features are still faithful and the
problem lives in the graph/edges. If (a) collapses well below (b), the
condensation step itself destroyed the feature content.
"""

import json
import os
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
V1 = OUT.parent / 'condense_v1_local_20261102'
SOURCE = Path(r'D:\下载学习软件\韩旭涛_科研项目资料\跑FedMVP实验\10_29跑实验\FedGVD-multimodal_fedgvd\dataset\global\subgraph_fl')

(OUT / 'tmp').mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT / 'tmp')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from openfgl.flcore.multimodal_fedgvd.utils import graph_from_payload

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATASETS = ('Movies', 'Toys')
SPLIT_SEEDS = (42, 43, 44)
METHODS = ('fixed', 'legacy')
STUDENT_SEEDS = (500, 501, 502)
MLP_EPOCHS = 500
ROWS, PROBES = [], []

# V3 REPORT.md, stage B, "orig" column (GCN student on the cached synthetic graph).
GCN_ON_SYN = {
    ('Movies', 'fixed'): {42: 30.470, 43: 33.436, 44: 30.266},
    ('Movies', 'legacy'): {42: 34.969, 43: 19.632, 44: 40.798},
    ('Toys', 'fixed'): {42: 20.049, 43: 14.833, 44: 24.287},
    ('Toys', 'legacy'): {42: 20.538, 43: 19.315, 44: 43.928},
}


def load_client(dataset, seed):
    split = torch.load(V1 / f'split_{dataset}_{seed}.pt', map_location='cpu', weights_only=False)
    ids = split['node_ids']
    raw = SOURCE / dataset / 'raw'
    labels = np.load(raw / 'labels.npy', allow_pickle=False)
    text = torch.from_numpy(np.array(np.load(raw / 'text_features.npy', mmap_mode='r')[ids.numpy()], dtype=np.float32))
    image = torch.from_numpy(np.array(np.load(raw / 'image_features.npy', mmap_mode='r')[ids.numpy()], dtype=np.float32))
    edges = torch.from_numpy(np.load(raw / 'edge_index.npy', allow_pickle=False).astype(np.int64))
    edges, _ = subgraph(ids, edges, relabel_nodes=True, num_nodes=len(labels))
    real = Data(x=torch.cat((F.normalize(image), F.normalize(text)), 1),
                x_text=text, x_image=image,
                y=torch.from_numpy(labels.astype(np.int64))[ids],
                modality_mask=torch.stack((text.norm(dim=1) > 0, image.norm(dim=1) > 0), 1),
                edge_index=edges, edge_weight=torch.ones(edges.size(1)),
                num_global_classes=int(labels.max()) + 1)
    return real, split['train_mask'].bool(), split['val_mask'].bool()


def load_graph(dataset, method, seed):
    path = V1 / 'cache/20261102_224147' / f'{dataset}_{method}_{seed}.pt'
    obj = torch.load(path, map_location='cpu', weights_only=False)
    return graph_from_payload(obj['graph'], torch.device('cpu'))


def classification(pred, labels, classes):
    confusion = torch.bincount(labels.cpu() * classes + pred.cpu(),
                               minlength=classes ** 2).reshape(classes, classes)
    actual, predicted = confusion.sum(1), confusion.sum(0)
    supported = actual + predicted > 0
    return dict(predicted_counts=predicted.tolist(), actual_counts=actual.tolist(),
                majority_class=int(actual.argmax()),
                predicted_majority_share=float(predicted[actual.argmax()] / predicted.sum()),
                macro_f1=float((2 * confusion.diag()[supported]
                                / (actual + predicted)[supported]).mean()))


def mlp_eval(x_train, y_train, real, val_mask, classes, seed):
    with torch.random.fork_rng(devices=[0] if DEVICE.type == 'cuda' else []):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x_train.size(1), 32), nn.ReLU(),
                              nn.Linear(32, classes)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.)
        for _ in range(MLP_EPOCHS):
            optimizer.zero_grad()
            F.cross_entropy(model(x_train), y_train).backward()
            optimizer.step()
        with torch.no_grad():
            pred = model(real.x.to(DEVICE)[val_mask.to(DEVICE)]).argmax(1)
    truth = real.y[val_mask].to(DEVICE)
    return float((pred == truth).float().mean()), classification(pred, truth, classes)


def quota_subset(real, train, syn_y, seed):
    """Pick real training nodes with the same per-class counts as the synthetic graph."""
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for label in syn_y.unique(sorted=True):
        candidates = (train & (real.y == label)).nonzero(as_tuple=True)[0]
        count = int((syn_y == label).sum())
        perm = torch.randperm(candidates.numel(), generator=generator)
        parts.append(candidates[perm[:count]])
    return torch.cat(parts)


def nearest_neighbour_probe(syn_x, syn_y, real_x, real_y, train):
    """How far the synthetic features drifted from the real training manifold."""
    reference = F.normalize(real_x[train].cpu(), dim=1)
    query = F.normalize(syn_x.detach().cpu(), dim=1)
    similarity = query @ reference.T
    best, index = similarity.max(dim=1)
    neighbours = real_y[train][index]
    return dict(nn_cos_mean=float(best.mean()), nn_cos_min=float(best.min()),
                nn_same_label_rate=float((neighbours == syn_y).float().mean()))


def record(dataset, method, seed, source, x_train, y_train, real, val, classes):
    for student_seed in STUDENT_SEEDS:
        score, metrics = mlp_eval(x_train, y_train, real, val, classes, student_seed)
        ROWS.append(dict(dataset=dataset, method=method, split_seed=seed, source=source,
                         student_seed=student_seed, acc=score,
                         gcn_on_syn_pct=GCN_ON_SYN[(dataset, method)][seed],
                         **metrics))
        print(f'MLP {dataset} {method} {seed} {source} {student_seed} acc={score*100:.3f} '
              f'F1={metrics["macro_f1"]*100:.3f}', flush=True)


def run():
    for dataset in DATASETS:
        for seed in SPLIT_SEEDS:
            real, train, val = load_client(dataset, seed)
            classes = real.num_global_classes
            for method in METHODS:
                graph = load_graph(dataset, method, seed).to(DEVICE)
                syn_x, syn_y = graph.x, graph.y
                record(dataset, method, seed, 'syn_feat', syn_x, syn_y,
                       real, val, classes)
                subset = quota_subset(real, train, syn_y.cpu(), seed)
                record(dataset, method, seed, 'real_subset', real.x[subset].to(DEVICE),
                       real.y[subset].to(DEVICE), real, val, classes)
                record(dataset, method, seed, 'real_full', real.x[train].to(DEVICE),
                       real.y[train].to(DEVICE), real, val, classes)
                probe = nearest_neighbour_probe(syn_x, syn_y.cpu(), real.x, real.y, train)
                PROBES.append(dict(dataset=dataset, method=method, split_seed=seed,
                                   nodes=graph.num_nodes, edges=graph.num_edges, **probe))
                print('PROBE', PROBES[-1], flush=True)
                dump()


def dump():
    (OUT / 'results.json').write_text(json.dumps(
        dict(rows=ROWS, probes=PROBES,
             device=str(DEVICE), torch=torch.__version__,
             student_seeds=list(STUDENT_SEEDS), mlp_epochs=MLP_EPOCHS),
        ensure_ascii=False, indent=2), encoding='utf-8')


def fmt(values):
    sd = statistics.stdev(values) if len(values) > 1 else 0.
    return f'{statistics.mean(values):.3f} ± {sd:.3f}'


def report():
    lines = ['# V4 合成特征探针（2026-11-03）', '',
             '目的：判断凝缩图失效的根因在“合成特征内容”还是在“图/边结构”。',
             '',
             '协议：全部只读 V1 缓存与原始数据，不改生产源码。把同一套 32 维 MLP'
             '（与 V3 mlp_only 锚点完全一致：lr=.01、weight_decay=0、500 轮、'
             '种子 500/501/502）分别喂给三种特征，在同一个真实客户端 val 上打分：'
             '- `syn_feat`：缓存合成图的特征，完全不用任何边（等价 pure-feature MLP）；',
             '- `real_subset`：真实训练节点中按合成图完全相同的每类配额抽出的子集（样本量对齐）；',
             '- `real_full`：全部真实训练节点（样本量上限参照）。',
             '',
             '这三者之间唯一的差别是“特征从哪来/有多少”，不含任何图结构，因此可以直接'
             '回答“合成特征本身还能不能用”。同时给出 `gcn_on_syn` 一列，即 V3 阶段 B 里'
             '同一个合成图配 GCN 学生的分数，用于对照“加边到底帮了还是害了”。', '']

    lines += ['## 主表（单位 %）', '',
              '| 数据集 | 方法 | 划分seed | syn_feat(MLP) | real_subset(MLP) | real_full(MLP) | gcn_on_syn |',
              '|---|---|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SPLIT_SEEDS:
                cells = []
                for source in ('syn_feat', 'real_subset', 'real_full'):
                    group = [r for r in ROWS if (r['dataset'], r['method'], r['split_seed'], r['source'])
                             == (dataset, method, seed, source)]
                    cells.append(fmt([r['acc'] * 100 for r in group]))
                lines.append(f'| {dataset} | {method} | {seed} | ' + ' | '.join(cells)
                             + f' | {GCN_ON_SYN[(dataset, method)][seed]:.3f} |')

    lines += ['', '## 按数据集汇总（三种子 × 三学生种子，单位 %）', '',
              '| 数据集 | 方法 | syn_feat | real_subset | real_full | gcn_on_syn 均值 |',
              '|---|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for method in METHODS:
            row = []
            for source in ('syn_feat', 'real_subset', 'real_full'):
                group = [r for r in ROWS if (r['dataset'], r['method'], r['source']) == (dataset, method, source)]
                row.append(fmt([r['acc'] * 100 for r in group]))
            gcn = statistics.mean(GCN_ON_SYN[(dataset, method)].values())
            lines.append(f'| {dataset} | {method} | ' + ' | '.join(row) + f' | {gcn:.3f} |')

    lines += ['', '## 特征漂移探针', '',
              '每个合成节点在真实训练特征里找余弦最近的邻居，看最近邻是否同类。'
              '`nn_cos_mean` 越接近 1、`nn_same_label_rate` 越高，说明合成特征越保真。', '',
              '| 数据集 | 方法 | 划分seed | 节点 | 边 | nn_cos_mean | nn_cos_min | 最近邻同类率 % |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for p in PROBES:
        lines.append(f'| {p["dataset"]} | {p["method"]} | {p["split_seed"]} | {p["nodes"]} | '
                     f'{p["edges"]} | {p["nn_cos_mean"]:.4f} | {p["nn_cos_min"]:.4f} | '
                     f'{p["nn_same_label_rate"]*100:.2f} |')

    def agg(dataset, source):
        return [r['acc'] * 100 for r in ROWS if (r['dataset'], r['source']) == (dataset, source)]

    lines += ['', '## 判定', '']
    for dataset in DATASETS:
        syn = statistics.mean(agg(dataset, 'syn_feat'))
        sub = statistics.mean(agg(dataset, 'real_subset'))
        full = statistics.mean(agg(dataset, 'real_full'))
        gcn = statistics.mean([v for m in METHODS for v in GCN_ON_SYN[(dataset, m)].values()])
        drop = sub - syn
        edge_gain = gcn - syn
        verdict = []
        if drop >= 3:
            verdict.append(f'合成特征比同数量真实特征低 {drop:.2f} pp，特征内容已明显退化')
        else:
            verdict.append(f'合成特征与同数量真实特征差 {drop:+.2f} pp，特征内容基本保真')
        if edge_gain <= -3:
            verdict.append(f'但给这张图加上边之后反而再降 {-edge_gain:.2f} pp，边结构是负资产')
        elif edge_gain >= 3:
            verdict.append(f'加上边带来 {edge_gain:+.2f} pp，边结构有正贡献')
        else:
            verdict.append(f'加上边变化 {edge_gain:+.2f} pp，边结构基本无影响')
        lines.append(f'- **{dataset}**：syn_feat {syn:.2f}｜real_subset {sub:.2f}｜real_full {full:.2f}｜'
                     f'gcn_on_syn {gcn:.2f}。' + '；'.join(verdict) + '。')

    lines += ['', '## 边界', '',
              '本探针是诊断而非因果证明：它只能说明“当前缓存里的合成特征还能不能被线性/浅层模型使用”，'
              '不能直接证明是 feat_alpha、PGE 还是采样配额造成的。若 syn_feat 与 real_subset 差距明显，'
              '下一步才值得重跑凝缩并消融 feat_alpha；若两者接近，则应回头查边与传播而非特征。',
              '所有数字均在同一真实客户端 val 上取得，该 val 也参与了 V1 的选图，存在选择偏差。']

    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(f'device={DEVICE} torch={torch.__version__}', flush=True)
    run()
    report()
    print('done', flush=True)
