"""V5 feat_alpha ablation: can the condensed features keep their semantics?

Read-only over raw data; only the in-process config is overridden, no production
source file is edited. Condensation is re-run because feat_alpha acts inside
train_syn, so the V1 caches (alpha=10) cannot be reused for alpha<10.

For every condensed graph we report both:
- feature probe: MLP on synthetic features only (no edges), same protocol as V4;
- graph student: the same GCN/student used by _validate_synthetic.

Reference numbers come from V4 (alpha=10 caches) and its real-feature anchors.
"""

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from openfgl.flcore.multimodal_fedgvd import SimGC_transductive as sim
from openfgl.flcore.multimodal_fedgvd.fedgvd_config import config
from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload, graph_from_payload

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATASETS = ('Movies', 'Toys')
SPLIT_SEEDS = (42,)
ALPHAS = (1.0, 0.0)
STUDENT_SEEDS = (500, 501, 502)
EPOCHS = (200, 300, 500, 50)
BASE_CONFIG = dict(config)
ROWS, PROBES, SHIFTS = [], [], []

# V4 alpha=10 references (same split, same student seeds, same val).
ALPHA10 = {
    'Movies': dict(syn_feat=28.17, real_subset=40.08, gcn=31.60),
    'Toys': dict(syn_feat=34.90, real_subset=62.81, gcn=23.82),
}


def manifest():
    code = ROOT / 'openfgl/flcore/multimodal_fedgvd'
    files = [code / name for name in ('SimGC_transductive.py', 'fedgvd_config.py', 'utils.py')]
    files += sorted(p for p in SOURCE.rglob('*') if p.is_file())
    return {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in files}


def dump():
    (OUT / 'results.json').write_text(json.dumps(
        dict(rows=ROWS, probes=PROBES, shifts=SHIFTS, alpha10_reference=ALPHA10,
             device=str(DEVICE), torch=torch.__version__), ensure_ascii=False, indent=2),
        encoding='utf-8')


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
    return dict(data=real,
                train_mask=split['train_mask'].bool(),
                val_mask=split['val_mask'].bool())


def classification(pred, labels, classes):
    confusion = torch.bincount(labels.cpu() * classes + pred.cpu(),
                               minlength=classes ** 2).reshape(classes, classes)
    actual, predicted = confusion.sum(1), confusion.sum(0)
    supported = actual + predicted > 0
    return dict(predicted_majority_share=float(predicted[actual.argmax()] / predicted.sum()),
                macro_f1=float((2 * confusion.diag()[supported]
                                / (actual + predicted)[supported]).mean()))


def student_eval(graph, real, val, seed, classes):
    captured = {}
    original = sim._accuracy

    def capture(model, data, mask):
        score = original(model, data, mask)
        with torch.no_grad():
            usable = mask & data.modality_mask.any(dim=1)
            prediction = model(data)[1][usable].argmax(dim=1)
        captured.update(classification(prediction, data.y[usable], classes))
        return score

    args = SimpleNamespace(seed=seed, hid_dim=32, num_layers=2, dropout=0.,
                           lr=.01, weight_decay=0., multimodal_fusion_temperature=.5,
                           multimodal_student_epochs=EPOCHS[2])
    with patch.object(sim, '_accuracy', capture):
        score = sim._validate_synthetic(graph.to(DEVICE), real, val, args,
                                        (real.x_text.size(1), real.x_image.size(1)), classes)
    return score, captured


def mlp_eval(x_train, y_train, real, val, classes, seed):
    with torch.random.fork_rng(devices=[0] if DEVICE.type == 'cuda' else []):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x_train.size(1), 32), nn.ReLU(),
                              nn.Linear(32, classes)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.)
        for _ in range(EPOCHS[2]):
            optimizer.zero_grad()
            F.cross_entropy(model(x_train), y_train).backward()
            optimizer.step()
        with torch.no_grad():
            prediction = model(real.x.to(DEVICE)[val.to(DEVICE)]).argmax(1)
    truth = real.y[val].to(DEVICE)
    return float((prediction == truth).float().mean()), classification(prediction, truth, classes)


def nearest_neighbour_probe(graph, real, train):
    reference = F.normalize(real.x.detach().cpu()[train], dim=1)
    query = F.normalize(graph.x.detach().cpu(), dim=1)
    best, index = (query @ reference.T).max(dim=1)
    neighbours = real.y.detach().cpu()[train][index]
    return dict(nn_cos_mean=float(best.mean()), nn_cos_min=float(best.min()),
                nn_same_label_rate=float((neighbours == graph.y.detach().cpu()).float().mean()))


def condense(dataset, seed, alpha):
    processed = load_client(dataset, seed)
    config.clear()
    config.update(BASE_CONFIG)
    config.update(feat_alpha=alpha, adj_mode='consistent_low', alternating_optim=True)
    args = SimpleNamespace(dataset=[dataset], seed=seed, num_clients=10, hid_dim=32, num_layers=2,
                           dropout=0., lr=.01, weight_decay=0., multimodal_reduction_rate=.2,
                           multimodal_teacher_epochs=EPOCHS[0],
                           multimodal_condense_epochs=EPOCHS[1],
                           multimodal_student_epochs=EPOCHS[2],
                           multimodal_validation_interval=EPOCHS[3],
                           multimodal_fusion_temperature=.5,
                           multimodal_condensed_pattern=str(
                               OUT / 'cache' / f'{dataset}_a{alpha}_s{seed}.pt'))
    path = Path(args.multimodal_condensed_pattern)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=False)
        graph = graph_from_payload(cached['graph'], DEVICE)
        print(f'REUSED {path.name} nodes={graph.num_nodes} edges={graph.num_edges}', flush=True)
        return graph, args
    start = time.time()
    graph = sim.data_gc(processed, 0, args, DEVICE)
    graph = graph.to(DEVICE)
    torch.save({'graph': graph_to_payload(graph)}, path)
    print(f'CONDENSED {dataset} alpha={alpha} nodes={graph.num_nodes} '
          f'edges={graph.num_edges} seconds={time.time()-start:.1f}', flush=True)
    return graph, args


def experiment():
    for dataset in DATASETS:
        for seed in SPLIT_SEEDS:
            processed = load_client(dataset, seed)
            real = processed['data'].to(DEVICE)
            val = processed['val_mask'].to(DEVICE)
            train = processed['train_mask']
            classes = real.num_global_classes
            for alpha in ALPHAS:
                graph, args = condense(dataset, seed, alpha)
                for student_seed in STUDENT_SEEDS:
                    score, metrics = student_eval(graph, real, val, student_seed, classes)
                    ROWS.append(dict(dataset=dataset, split_seed=seed, alpha=alpha,
                                     student_seed=student_seed, metric='gcn', value=score, **metrics))
                    print(f'GCN {dataset} a{alpha} {student_seed} {score*100:.3f}', flush=True)
                for student_seed in STUDENT_SEEDS:
                    score, metrics = mlp_eval(graph.x, graph.y, real, val, classes, student_seed)
                    ROWS.append(dict(dataset=dataset, split_seed=seed, alpha=alpha,
                                     student_seed=student_seed, metric='syn_feat_mlp',
                                     value=score, **metrics))
                    print(f'MLP {dataset} a{alpha} {student_seed} {score*100:.3f}', flush=True)
                probe = nearest_neighbour_probe(graph, real, train)
                PROBES.append(dict(dataset=dataset, split_seed=seed, alpha=alpha, **probe,
                                   nodes=graph.num_nodes, edges=graph.num_edges))
                print('PROBE', PROBES[-1], flush=True)
                SHIFTS.append(dict(dataset=dataset, split_seed=seed, alpha=alpha,
                                   weighted_indegree=float(
                                       graph.edge_weight[graph.edge_index[0] != graph.edge_index[1]].sum()
                                   ) / graph.num_nodes))
                dump()
                del graph
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
    config.clear()
    config.update(BASE_CONFIG)


def values(dataset, alpha, metric):
    return [r['value'] * 100 for r in ROWS
            if (r['dataset'], r['alpha'], r['metric']) == (dataset, alpha, metric)]


def fmt(items):
    sd = statistics.stdev(items) if len(items) > 1 else 0.
    return f'{statistics.mean(items):.3f} ± {sd:.3f}'


def report(unchanged):
    lines = ['# V5 feat_alpha 特征消融（2026-11-03）', '',
             '目的：V4 已证明合成特征内容退化（Movies −11.9 pp、Toys −27.9 pp 相对同数量真实特征）。'
             '本轮唯一变量是 `feat_alpha`（传播后类统计对齐损失的权重），检验它是否是特征漂移的直接原因。',
             '',
             '协议：只覆盖进程内 config，不写回任何生产源码；凝缩必须重跑，因为 feat_alpha 在 train_syn '
             '内部生效，V1 的 alpha=10 缓存无法复用。数据、划分、学生超参、种子与 V1/V4 完全一致'
             f'（teacher {EPOCHS[0]} / condense {EPOCHS[1]} / student {EPOCHS[2]}，种子 500/501/502）。', '',
             f'生产源码与原始数据运行前后 SHA256 一致性：**{unchanged}**。', '',
             '## 结果（单位 %）', '',
             '| 数据集 | feat_alpha | 合成特征 MLP | GCN 学生 | 同数量真特征 MLP | 最近邻余弦 | 最近邻同类率 % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        reference = ALPHA10[dataset]
        for alpha in ALPHAS:
            probes = [p for p in PROBES if (p['dataset'], p['alpha']) == (dataset, alpha)]
            if not probes:
                lines.append(f'| {dataset} | {alpha:g} | 缺失 | 缺失 | '
                             f'{reference["real_subset"]:.2f} | — | — |')
                continue
            probe = probes[0]
            lines.append(f'| {dataset} | {alpha:g} | {fmt(values(dataset, alpha, "syn_feat_mlp"))} | '
                         f'{fmt(values(dataset, alpha, "gcn"))} | {reference["real_subset"]:.2f} | '
                         f'{probe["nn_cos_mean"]:.4f} | {probe["nn_same_label_rate"]*100:.2f} |')
        lines.append(f'| {dataset} | 10 (V1/V4) | {reference["syn_feat"]:.2f} | {reference["gcn"]:.2f} | '
                     f'{reference["real_subset"]:.2f} | — | — |')

    lines += ['', '## 判定', '']
    for dataset in DATASETS:
        reference = ALPHA10[dataset]
        best_alpha, best_mlp = None, -1.
        for alpha in ALPHAS:
            scores = values(dataset, alpha, 'syn_feat_mlp')
            if not scores:
                continue
            score = statistics.mean(scores)
            if score > best_mlp:
                best_alpha, best_mlp = alpha, score
        if best_alpha is None:
            lines.append(f'- **{dataset}**：没有可用的 feat_alpha 结果，本轮未完成。')
            continue
        baseline_mlp = statistics.mean(values(dataset, best_alpha, 'syn_feat_mlp'))
        baseline_gcn = statistics.mean(values(dataset, best_alpha, 'gcn'))
        gain_mlp = baseline_mlp - reference['syn_feat']
        gain_gcn = baseline_gcn - reference['gcn']
        remaining = reference['real_subset'] - baseline_mlp
        text = (f'- **{dataset}**：最佳 feat_alpha={best_alpha:g}，合成特征 MLP {baseline_mlp:.2f} '
                f'（相对 alpha=10 {gain_mlp:+.2f} pp），GCN {baseline_gcn:.2f}（{gain_gcn:+.2f} pp），'
                f'距离同数量真实特征仍差 {remaining:.2f} pp。')
        if gain_mlp >= 5:
            text += '降低对齐权重能显著挽回特征质量，说明 feat_alpha 是漂移的直接来源。'
        else:
            text += '降低对齐权重未能显著挽回特征质量，说明漂移还有别的来源（初始化配额、分类损失或 PGE 联动），需另找。'
        lines.append(text)

    lines += ['', '## 边界', '',
              '只跑了划分种子 42，尚未扩展到 43/44；三个学生种子只消除训练随机性，不消除划分随机性。',
              'val 既参与 V1 选图又用于评分，仍是开发集结果。',
              '本实验只证明 feat_alpha 与特征保真度的关系，不证明端到端 federated 精度会同步改善。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    torch.set_num_threads(2)
    print(f'device={DEVICE} torch={torch.__version__}', flush=True)
    before = manifest()
    try:
        experiment()
    finally:
        after = manifest()
        (OUT / 'integrity.json').write_text(json.dumps(
            dict(before=before, after=after, unchanged=before == after), indent=2),
            encoding='utf-8')
        dump()
        report(before == after)


if __name__ == '__main__':
    main()
