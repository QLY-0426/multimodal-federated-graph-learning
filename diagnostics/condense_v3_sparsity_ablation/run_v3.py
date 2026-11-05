import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
V1 = OUT.parent / 'condense_v1_local_20261102'
SOURCE = Path(r'D:\下载学习软件\韩旭涛_科研项目资料\跑FedMVP实验\10_29跑实验\FedGVD-multimodal_fedgvd\dataset\global\subgraph_fl')
(OUT / 'tmp').mkdir(exist_ok=True)
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
from openfgl.flcore.multimodal_fedgvd.utils import graph_from_payload, graph_to_payload

DEVICE = torch.device('cuda:0')
DATASETS = ('Movies', 'Toys')
SEEDS = (42, 43, 44)
KS = (1, 2, 5, 10)
VARIANTS = tuple(f'topk_sym_k{k}' for k in KS)
ROWS = []
SCHEMAS = {}


def manifest():
    code = ROOT / 'openfgl/flcore/multimodal_fedgvd'
    files = [code / f for f in ('SimGC_transductive.py', 'fedgvd_config.py', 'utils.py')]
    files += sorted(p for p in SOURCE.rglob('*') if p.is_file())
    result = {}
    for path in files:
        with path.open('rb') as stream:
            result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


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
                x_text=text, x_image=image, y=torch.from_numpy(labels.astype(np.int64))[ids],
                modality_mask=torch.stack((text.norm(dim=1)>0, image.norm(dim=1)>0), 1),
                edge_index=edges, edge_weight=torch.ones(edges.size(1)),
                num_global_classes=int(labels.max())+1)
    return real.to(DEVICE), split['train_mask'].to(DEVICE), split['val_mask'].to(DEVICE)


def load_graph(dataset, method, seed):
    path = V1 / 'cache/20261102_224147' / f'{dataset}_{method}_{seed}.pt'
    obj = torch.load(path, map_location='cpu', weights_only=False)
    schema = dict(type=type(obj).__name__, keys=sorted(obj) if isinstance(obj, dict) else None)
    SCHEMAS[path.name] = schema
    payload = obj.get('graph') if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f'Unexpected cache format {path.name}: {schema}')
    schema['payload'] = {key: list(val.shape) if torch.is_tensor(val) else type(val).__name__
                         for key, val in payload.items()}
    required = {'x_text', 'x_image', 'y', 'edge_index', 'edge_weight', 'modality_mask'}
    if set(payload) != required:
        raise ValueError(f'Unexpected graph payload {schema}')
    print('CACHE', path.name, schema, flush=True)
    return graph_from_payload(payload, torch.device('cpu'))


def variant(graph, name):
    result = graph.clone()
    if name == 'orig':
        return result
    sources, targets = graph.edge_index.tolist()
    weights = dict(zip(zip(sources, targets), graph.edge_weight.tolist()))
    loops = {(i, i): w for (i, j), w in weights.items() if i == j}
    if name == 'selfloop_only':
        kept = {(i, i): loops.get((i, i), 1.) for i in range(graph.num_nodes)}
    else:
        k = int(name.split('_k')[-1])
        outgoing = [[] for _ in range(graph.num_nodes)]
        for (i, j), weight in weights.items():
            if i != j:
                outgoing[i].append((j, weight))
        kept = dict(loops)
        for i, edges in enumerate(outgoing):
            for j, weight in sorted(edges, key=lambda item: (-item[1], item[0]))[:k]:
                kept[i, j] = weight
                kept[j, i] = weights.get((j, i), weight)
    result.edge_index = torch.tensor(list(kept), dtype=torch.long).reshape(-1, 2).T.contiguous()
    result.edge_weight = graph.edge_weight.new_tensor(list(kept.values()))
    return result


def classification(pred, labels, classes):
    confusion = torch.bincount(labels.cpu()*classes+pred.cpu(), minlength=classes**2).reshape(classes, classes)
    actual, predicted = confusion.sum(1), confusion.sum(0)
    supported = actual+predicted > 0
    return dict(predicted_counts=predicted.tolist(), actual_counts=actual.tolist(),
                majority_class=int(actual.argmax()), majority_share=float(actual.max()/actual.sum()),
                predicted_majority_share=float(predicted[actual.argmax()]/predicted.sum()),
                macro_f1=float((2*confusion.diag()[supported]/(actual+predicted)[supported]).mean()))


def evaluate(graph, real, val, seed):
    metrics = {}
    original = sim._accuracy

    def capture(model, data, mask):
        score = original(model, data, mask)
        with torch.no_grad():
            mask = mask & data.modality_mask.any(dim=1)
            prediction = model(data)[1][mask].argmax(dim=1)
        metrics.update(classification(prediction, data.y[mask], real.num_global_classes))
        return score

    args = SimpleNamespace(seed=seed, hid_dim=32, num_layers=2, dropout=0.,
                           lr=.01, weight_decay=0., multimodal_fusion_temperature=.5,
                           multimodal_student_epochs=500)
    with patch.object(sim, '_accuracy', capture):
        score = sim._validate_synthetic(graph.to(DEVICE), real, val, args,
                                      (real.x_text.size(1), real.x_image.size(1)), real.num_global_classes)
    return score, metrics



STUDENT_SEEDS = (500, 501, 502)
ALL_VARIANTS = ('orig', 'selfloop_only', *VARIANTS)
ANCHORS = []
CHOSEN = None
STATUS = '执行中'


def save_results():
    dump('results.json', dict(status=STATUS, chosen=CHOSEN, variants=ROWS,
                              anchors=ANCHORS, cache_schema=SCHEMAS))


def run_variant(dataset, method, seed, name, original, real, val):
    graph = variant(original, name)
    key = f'{dataset}_{method}_{seed}_{name}'
    torch.save({'graph': graph_to_payload(graph)}, OUT / 'cache' / f'{key}.pt')
    nonloop = graph.edge_index[0] != graph.edge_index[1]
    diag = dict(nodes=graph.num_nodes, edges=graph.num_edges,
                nonloop_degree=int(nonloop.sum())/graph.num_nodes,
                weighted_indegree=float(graph.edge_weight[nonloop].sum())/graph.num_nodes,
                mean_weight=float(graph.edge_weight.mean()) if graph.num_edges else None)
    for student_seed in STUDENT_SEEDS:
        score, metrics = evaluate(graph.clone(), real, val, student_seed)
        row = dict(dataset=dataset, method=method, seed=seed, variant=name,
                   student_seed=student_seed, acc=score, **diag, **metrics)
        ROWS.append(row)
        print('RESULT', dataset, method, seed, name, student_seed,
              f'acc={score:.6f} F1={metrics["macro_f1"]:.6f}', flush=True)
        save_results()


def rows_for(ds, method, seed, name):
    return [r for r in ROWS if (r['dataset'],r['method'],r['seed'],r['variant']) ==
            (ds,method,seed,name)]


def diff(group, orig, metric):
    baseline = {r['student_seed']: r[metric] for r in orig}
    return [(r[metric]-baseline[r['student_seed']])*100 for r in group]


def mean(group, key):
    return statistics.mean(r[key] for r in group)


def anchor(ds, seed):
    real, train, val = load_client(ds, seed)
    induced = real.subgraph(train)
    x_train, y_train = real.x[train], real.y[train]
    for student_seed in STUDENT_SEEDS:
        gcn, _ = evaluate(induced.clone(), real, val, student_seed)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(student_seed)
            mlp = nn.Sequential(nn.Linear(real.x.size(1),32), nn.ReLU(),
                                nn.Linear(32,real.num_global_classes)).to(DEVICE)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=.01, weight_decay=0.)
            for _ in range(500):
                optimizer.zero_grad()
                F.cross_entropy(mlp(x_train), y_train).backward()
                optimizer.step()
            with torch.no_grad():
                score = float((mlp(real.x[val]).argmax(1)==real.y[val]).float().mean())
        ANCHORS.append(dict(dataset=ds, seed=seed, student_seed=student_seed,
                            real100=gcn, mlp_only=score))
        print('ANCHORS', ANCHORS[-1], flush=True)
        save_results()


def experiment():
    global CHOSEN, STATUS
    for ds in DATASETS:
        real, _, val = load_client(ds,42)
        original = load_graph(ds,'fixed',42)
        for name in ALL_VARIANTS:
            run_variant(ds,'fixed',42,name,original,real,val)
    eligible = [
        name for name in VARIANTS
        if any(mean(rows_for(ds,'fixed',42,name),'macro_f1') >=
               mean(rows_for(ds,'fixed',42,'orig'),'macro_f1') for ds in DATASETS)
    ]
    # A fallback still completes B if both datasets lose macro-F1 for all k.
    CHOSEN = max(eligible or VARIANTS, key=lambda name: statistics.mean(
        mean(rows_for(ds,'fixed',42,name),'acc') for ds in DATASETS))
    print('STAGE B', CHOSEN, 'eligible', eligible, flush=True)
    for ds in DATASETS:
        for seed in SEEDS:
            real, _, val = load_client(ds,seed)
            for method in ('fixed','legacy'):
                if seed == 42 and method == 'fixed':
                    continue
                original = load_graph(ds,method,seed)
                for name in ('orig', CHOSEN):
                    run_variant(ds,method,seed,name,original,real,val)
    for ds in DATASETS:
        for seed in SEEDS:
            anchor(ds,seed)
    STATUS = '阶段 A、阶段 B 与两个锚点均完成'


def fmt(values):
    sd = statistics.stdev(values) if len(values)>1 else 0.
    return f'{statistics.mean(values):.3f} ± {sd:.3f}'


def report(unchanged):
    lines = ['# V3 缓存图稀疏消融', '', STATUS, '',
             '使用 V1 缓存及保存的客户端 ID 和掩码。所有变体沿用 V1 学生超参，CUDA 500 轮，种子 500/501/502。只修改训练图的边，评估图保持真实客户端图。平均加权入度排除自环。标准差为样本标准差。', '',
             '阶段 B 在两数据集上统一使用阶段 A 平均准确率最高且至少一个数据集 macro-F1 未下降的 k。如果没有符合条件的 k，使用平均准确率最高的 k 复核负结果。同分时选较小 k。', '',
             f'阶段 B 选定 **{CHOSEN}**。生产源码和原始数据前后 SHA256 一致性为 **{unchanged}**，完整清单见 integrity.json。', '',
             '## 阶段 A', '',
             '| 数据集 | 变体 | 边数 | 加权入度 | val acc % | 配对 Δ pp | 预测多数类 % | macro-F1 % |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        orig = rows_for(ds,'fixed',42,'orig')
        for name in ALL_VARIANTS:
            group = rows_for(ds,'fixed',42,name)
            if not group:
                continue
            first = group[0]
            lines.append(f'| {ds} | {name} | {first["edges"]} | {first["weighted_indegree"]:.4f} | '
                         f'{fmt([r["acc"]*100 for r in group])} | {fmt(diff(group,orig,"acc"))} | '
                         f'{fmt([r["predicted_majority_share"]*100 for r in group])} | '
                         f'{fmt([r["macro_f1"]*100 for r in group])} |')
    lines += ['', '## 阶段 B', '',
              '| 数据集 | 方法 | 划分 seed | orig % | top-k % | Δ500 pp | Δ501 pp | Δ502 pp | 配对 Δ pp | macro-F1 Δ pp | 加权入度 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    verdicts = []
    for ds, target, random_score in [('Movies',.62,39.3661),('Toys',.30,59.7392)]:
        for method in ('fixed','legacy'):
            pooled, f1s, degrees, top_scores, orig_scores, per_split = [], [], [], [], [], []
            for seed in SEEDS:
                orig = rows_for(ds,method,seed,'orig')
                top = rows_for(ds,method,seed,CHOSEN)
                if len(top)!=3 or len(orig)!=3:
                    continue
                delta, f1 = diff(top,orig,'acc'), diff(top,orig,'macro_f1')
                deg = top[0]['weighted_indegree']
                a = [r['acc']*100 for r in orig]
                b = [r['acc']*100 for r in top]
                vals = ' | '.join(f'{d:+.3f}' for d in delta)
                lines.append(f'| {ds} | {method} | {seed} | {fmt(a)} | {fmt(b)} | {vals} | {fmt(delta)} | {fmt(f1)} | {deg:.4f} |')
                pooled.extend(delta)
                f1s.extend(f1)
                degrees.append(deg)
                top_scores.extend(b)
                orig_scores.extend(a)
                per_split.append(statistics.mean(delta))
            if not pooled:
                continue
            lines.append(f'| {ds} | {method} | 汇总 | {fmt(orig_scores)} | {fmt(top_scores)} | — | — | — | {fmt(pooled)} | {fmt(f1s)} | {statistics.mean(degrees):.4f} |')
            if method == 'fixed' and len(per_split)==3:
                gain, f1_gain = statistics.mean(pooled), statistics.mean(f1s)
                degree_ok = all(target/10 <= d <= target*10 for d in degrees)
                gap_before = max(0,random_score-statistics.mean(orig_scores))
                gap_after = max(0,random_score-statistics.mean(top_scores))
                count = sum(d>0 for d in per_split)
                passed = degree_ok and gain>=2 and count>=2 and f1_gain>0 and gap_after<gap_before
                verdicts.append((ds,passed,gain,f1_gain,count,gap_before,gap_after,degree_ok))
    lines += ['', '## 锚点', '',
              'real100 使用全部真实训练节点及其诱导边训练同款 GCN。mlp_only 使用归一化的双模态拼接特征、32 维隐藏层和交叉熵，不使用图边。新结果按三个划分与三个学生种子共九次聚合，V1 随机基线按三个划分聚合。', '',
              '| 数据集 | real100 val % | mlp_only val % | V1 随机基线 val % |',
              '|---|---:|---:|---:|']
    for ds, random_result in [('Movies','39.3661 ± 1.5741'),('Toys','59.7392 ± 0.6153')]:
        group = [a for a in ANCHORS if a['dataset']==ds]
        if group:
            lines.append(f'| {ds} | {fmt([a["real100"]*100 for a in group])} | '
                         f'{fmt([a["mlp_only"]*100 for a in group])} | {random_result} |')
    lines += ['', '## 判定', '',
              '密度主因门槛采用真实加权入度的 0.1～10 倍、至少两个划分的配对均值为正、整体提升至少 2 pp、macro-F1 回升且与随机基线差距缩小。若变化均在重训标准差内，放弃将密度视为主因。若 top-k 反而变差，记录负结果。固定缓存上的删边只能检验后处理，无法分离凝缩过程中已经发生的特征损失。', '']
    for ds,passed,gain,f1,count,before,after,degree_ok in verdicts:
        lines.append(f'- {ds} 的修复档平均准确率变化 {gain:+.3f} pp，macro-F1 变化 {f1:+.3f} pp，{count}/3 个划分为正。随机基线差距 {before:.3f} → {after:.3f} pp。入度门槛 {degree_ok}。')
    if len(verdicts)==2:
        yes = [ds for ds,passed,*_ in verdicts if passed]
        if len(yes)==2:
            lines += ['', '**一句话判定：按本次门槛，边密度是两组精度损失的主因。** 下一步测试 PGE 的 top-k 稀疏约束或加权度预算。']
        elif yes:
            lines += ['', f'**一句话判定：边密度仅在 {yes[0]} 达到主因门槛，不能作为两个数据集的共同主因。** 对未通过的数据集，下一步隔离合成特征质量与 feat_alpha。']
        else:
            lines += ['', '**一句话判定：固定缓存上的 top-k 删边未达到主因门槛，现阶段不把边密度作为共同主因。** 下一步固定真实诱导边与初始节点，比较合成特征和原始特征，并消融 feat_alpha=0/1/10。']
    else:
        lines += ['', '**一句话判定：缺少完整阶段 B 配对数据。** 最小补齐实验是完成剩余缓存图的 orig 与选定 top-k 三种子重训。']
    lines += ['', '逐次预测类别计数、真实类别计数、多数类占比、macro-F1、边权及准确率见 results.json。预测多数类指真实验证标签的多数类。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    global STATUS
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and '4060' in torch.cuda.get_device_name(0)
    (OUT/'cache').mkdir(exist_ok=True)
    before = manifest()
    dump('integrity_before.json',before)
    try:
        experiment()
        assert len(ROWS)==96 and len(ANCHORS)==18
    except Exception as error:
        STATUS = f'运行中断：{type(error).__name__} {error}'
        raise
    finally:
        after = manifest()
        dump('integrity.json', dict(before=before,after=after,unchanged=before==after))
        save_results()
        report(before==after)


if __name__ == '__main__':
    main()

