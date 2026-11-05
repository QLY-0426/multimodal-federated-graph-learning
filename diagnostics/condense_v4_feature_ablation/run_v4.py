"""Paired feature-only ablation on V1's fixed random induced subgraphs."""
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OUT.parent / 'condense_v3_sparsity_ablation'))
import run_v3 as v3

from run_v3 import F, sim, torch

(OUT / 'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT / 'tmp')
DEVICE = v3.DEVICE
STUDENT_SEEDS = (500, 501, 502)
ALPHAS = (0, 1, 10)
METHODS = ('raw', *(f'alpha{a}' for a in ALPHAS))
ARGS = SimpleNamespace(lr=.01, weight_decay=0., multimodal_teacher_epochs=200,
                       multimodal_validation_interval=50)


def stats(real, train, branch):
    available = real.modality_mask[:, branch]
    feature = (real.x_text, real.x_image)[branch]
    propagated = sim._propagated_features(feature, real.edge_index, real.edge_weight, available, 2)
    result = {}
    for c in real.y[train & available].unique().tolist():
        x = propagated[train & available & (real.y == c)]
        result[c] = (x.mean(0), x.std(0, unbiased=False), len(x))
    return result


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    return float(sum(g.square().sum() for g in grads).sqrt())


def optimize(original, teacher, moments, alpha):
    graph = original.clone().to(DEVICE)
    initial = [F.normalize(graph.x_text, dim=1), F.normalize(graph.x_image, dim=1)]
    parameters = [torch.nn.Parameter(x.clone()) for x in initial]
    graph.x_text, graph.x_image = parameters
    optimizer = torch.optim.Adam(parameters, lr=sim.config['lr_feat'])
    mask = graph.modality_mask.any(1)
    checkpoints = []
    for step in range(1, 301):
        optimizer.zero_grad()
        classification = sim._loss(teacher, graph, mask)
        align = classification.new_zeros(())
        if alpha:
            for i, feature in enumerate(parameters):
                available = graph.modality_mask[:, i]
                propagated = sim._propagated_features(feature, graph.edge_index, graph.edge_weight, available, 2)
                align = align + sim._alignment(moments[i], propagated, graph.y, available)
        loss = classification + alpha * align
        if step in (1, 50, 300):
            checkpoints.append(dict(step=step, cls=float(classification.detach()),
                                    weighted_align=float((alpha*align).detach()),
                                    cls_grad=grad_norm(classification, parameters),
                                    align_grad=grad_norm(alpha*align, parameters) if alpha else 0.))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        shifts = [float((1-F.cosine_similarity(x, start, dim=1))[graph.modality_mask[:, i]].mean())
                  for i, (x, start) in enumerate(zip(parameters, initial))]
        graph.x_text, graph.x_image = (x.detach() for x in parameters)
        graph.x = torch.cat((F.normalize(graph.x_image), F.normalize(graph.x_text)), 1)
        acc = sim._accuracy(teacher, graph, mask)
    return graph.cpu(), dict(cosine_distance=shifts, teacher_on_syn=acc, checkpoints=checkpoints)


def verify_graph(graph, real, train, selected, dataset, seed):
    split = torch.load(v3.V1/f'split_{dataset}_{seed}.pt', weights_only=False)
    index = torch.searchsorted(split['node_ids'], selected)
    assert torch.equal(split['node_ids'][index], selected)
    assert train[index].all()
    for attr in ('x_text','x_image','modality_mask','y'):
        assert torch.equal(graph[attr], real[attr][index].cpu())
    edges, weights = v3.subgraph(index.to(DEVICE), real.edge_index, real.edge_weight,
                                  relabel_nodes=True, num_nodes=real.num_nodes)
    assert torch.equal(graph.edge_index, edges.cpu()) and torch.equal(graph.edge_weight, weights.cpu())


def report(rows, details, unchanged):
    def fmt(values):
        return f'{mean(values):.2f} ± {stdev(values):.2f}'

    lines = ['# V4 固定真实诱导边的特征消融', '',
             f'已完成 {len(rows)}/72 次学生评分。生产源码与原始数据的前后 SHA256 一致为 {unchanged}。', '',
             '复用 V1 的随机子图及同一真实 val，核验节点 ID、真实训练掩码、标签、特征和诱导边。每个划分训练一份冻结教师（200 轮），再固定边和标签优化特征 300 轮。三档只改变对齐项权重，统一关闭 PGE 与平滑项。沿用 V1 特征学习率和 V3 的三个学生种子、500 轮及模型超参。全部按第 300 轮特征评分。', '',
             '| 数据集 | 方法 | val % | macro-F1 % | 同划分同学生种子相对 raw Δ pp |',
             '|---|---|---:|---:|---:|']
    baseline = {(r['dataset'],r['seed'],r['student_seed']):r['acc'] for r in rows if r['method']=='raw'}
    verdict = []
    for ds in v3.DATASETS:
        means = {}
        for method in METHODS:
            group = [r for r in rows if r['dataset']==ds and r['method']==method]
            if len(group)<2:
                continue
            delta = [(r['acc']-baseline[ds,r['seed'],r['student_seed']])*100 for r in group]
            lines.append(f'| {ds} | {method} | {fmt([r["acc"]*100 for r in group])} | {fmt([r["macro_f1"]*100 for r in group])} | {fmt(delta)} |')
            means[method] = mean(delta)
        if len(means)==4:
            best = max(METHODS[1:], key=means.get)
            verdict.append(f'- {ds} 特征优化中最佳为 {best}，相对 raw {means[best]:+.2f} pp。alpha10 相对 alpha0 {means["alpha10"]-means["alpha0"]:+.2f} pp。')
    lines += ['', '## 每个划分的配对准确率差（pp）', '',
              '| 数据集 | 划分 | alpha0 | alpha1 | alpha10 |', '|---|---:|---:|---:|---:|']
    for ds in v3.DATASETS:
        for seed in v3.SEEDS:
            cells = []
            for method in METHODS[1:]:
                values = [(r['acc']-baseline[ds,seed,r['student_seed']])*100 for r in rows
                          if r['dataset']==ds and r['seed']==seed and r['method']==method]
                cells.append(fmt(values) if len(values)>1 else '—')
            lines.append(f'| {ds} | {seed} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 现状分析', '', *verdict, '',
              '教师真实 val、各档 teacher-on-syn、分类与加权对齐损失、梯度范数和归一化特征余弦距离保存在 results.json。V3 的 Toys 删边收益与 V4 的固定边特征收益分属不同干预。', '',
              '## 下一步方案', '',
              '先在相同划分与节点上验证低学习率和余弦锚定，继续保留原始特征对照，按三个划分的同种子准确率差与 macro-F1 判断。Toys 的图结构方向单独补齐 k=1 的 seed43/44，只有通过度预算和配对增益门槛再进入训练期稀疏约束。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and '4060' in torch.cuda.get_device_name(0)
    before = v3.manifest()
    rows, details = [], []
    try:
        for ds in v3.DATASETS:
            for seed in v3.SEEDS:
                real, train, val = v3.load_client(ds,seed)
                cache = torch.load(v3.V1/'cache/20261102_224147'/f'{ds}_random_baseline_{seed}.pt',
                                   weights_only=False,map_location='cpu')
                original = v3.graph_from_payload(cache['graph'],torch.device('cpu'))
                verify_graph(original,real,train,cache['node_ids'],ds,seed)
                dimensions = (real.x_text.size(1),real.x_image.size(1))
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    teacher = sim.MultimodalSGC(dimensions,32,real.num_global_classes,2,0.,.5).to(DEVICE)
                    sim.train_teacher(real,train,val,teacher,ARGS)
                moments = [stats(real,train,i) for i in range(2)]
                with torch.no_grad():
                    pred = teacher(real)[1][val].argmax(1)
                    teacher_val = dict(acc=float((pred==real.y[val]).float().mean()),
                                       **v3.classification(pred,real.y[val],real.num_global_classes))
                for method in METHODS:
                    if method == 'raw':
                        graph = original
                        diag = dict(teacher_on_syn=sim._accuracy(teacher,graph.clone().to(DEVICE),graph.modality_mask.any(1).to(DEVICE)))
                    else:
                        graph,diag = optimize(original,teacher,moments,int(method.removeprefix('alpha')))
                    assert torch.equal(graph.edge_index,original.edge_index) and torch.equal(graph.edge_weight,original.edge_weight)
                    assert torch.equal(graph.y,original.y) and torch.equal(graph.modality_mask,original.modality_mask)
                    details.append(dict(dataset=ds,seed=seed,method=method,teacher_val=teacher_val,**diag))
                    print('DIAG',ds,seed,method,diag,flush=True)
                    for student_seed in STUDENT_SEEDS:
                        score,metrics = v3.evaluate(graph.clone(),real,val,student_seed)
                        rows.append(dict(dataset=ds,seed=seed,method=method,student_seed=student_seed,acc=score,**metrics))
                        print('RESULT',ds,seed,method,student_seed,score,metrics['macro_f1'],flush=True)
                    (OUT/'results.json').write_text(json.dumps(dict(rows=rows,diagnostics=details),indent=2),encoding='utf-8')
        assert len(rows)==72
    finally:
        after = v3.manifest()
        (OUT/'integrity.json').write_text(json.dumps(dict(before=before,after=after,unchanged=before==after),indent=2,ensure_ascii=False),encoding='utf-8')
        report(rows,details,before==after)


if __name__ == '__main__':
    main()
