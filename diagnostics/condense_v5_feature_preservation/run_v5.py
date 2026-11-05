"""Fixed-edge feature preservation and Toys k=1 follow-up."""
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys

OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
V4 = OUT.parent / 'condense_v4_feature_ablation'
V3 = OUT.parent / 'condense_v3_sparsity_ablation'
sys.path.insert(0, str(V4))
import run_v4 as prior
from run_v4 import F, sim, torch, v3

(OUT/'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT/'tmp')
SETTINGS = ((.05,0.), (.05,.1), (.005,0.), (.005,.1))
BASE = json.loads((V4/'results.json').read_text(encoding='utf-8'))
SPARSE = json.loads((V3/'results.json').read_text(encoding='utf-8'))
STUDENTS = (500,501,502)
DEVICE = v3.DEVICE
RESULT = dict(val=[], features=[], test=[], toys_k1=[], winners={})


def save():
    (OUT/'results.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')


def original(ds, seed, real, train):
    cache = torch.load(v3.V1/'cache/20261102_224147'/f'{ds}_random_baseline_{seed}.pt',
                       weights_only=False,map_location='cpu')
    graph = v3.graph_from_payload(cache['graph'],torch.device('cpu'))
    prior.verify_graph(graph,real,train,cache['node_ids'],ds,seed)
    return graph


def optimize(graph, teacher, moments, lr, anchor_weight):
    graph = graph.clone().to(DEVICE)
    initial = [F.normalize(graph.x_text), F.normalize(graph.x_image)]
    params = [torch.nn.Parameter(x.clone()) for x in initial]
    graph.x_text, graph.x_image = params
    optimizer = torch.optim.Adam(params,lr=lr)
    mask = graph.modality_mask.any(1)
    for _ in range(300):
        optimizer.zero_grad()
        cls = sim._loss(teacher,graph,mask)
        align = cls.new_zeros(())
        anchor = cls.new_zeros(())
        for i,x in enumerate(params):
            present = graph.modality_mask[:,i]
            features = sim._propagated_features(x,graph.edge_index,graph.edge_weight,present,2)
            align += sim._alignment(moments[i],features,graph.y,present)
            if anchor_weight:
                anchor += (1-F.cosine_similarity(x[present],initial[i][present])).mean()/2
        loss = cls + 10*align + anchor_weight*anchor
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        distance = [float((1-F.cosine_similarity(x,start))[graph.modality_mask[:,i]].mean())
                    for i,(x,start) in enumerate(zip(params,initial))]
        graph.x_text, graph.x_image = (x.detach() for x in params)
        graph.x = torch.cat((F.normalize(graph.x_image),F.normalize(graph.x_text)),1)
    return graph.cpu(),distance


def key(lr, anchor):
    return f'lr{lr:g}_anchor{anchor:g}'


def score(ds, seed, name, graph, real, mask, dest):
    for student in STUDENTS:
        acc, metrics = v3.evaluate(graph.clone(),real,mask,student)
        dest.append(dict(dataset=ds,seed=seed,method=name,student_seed=student,acc=acc,**metrics))
        print('SCORE',ds,seed,name,student,acc,metrics['macro_f1'],flush=True)
        save()


def summary(ds, name):
    rows = [r for r in RESULT['val'] if r['dataset']==ds and r['method']==name]
    raw = {(r['seed'],r['student_seed']):r for r in BASE['rows']
           if r['dataset']==ds and r['method']=='raw'}
    acc = [(r['acc']-raw[r['seed'],r['student_seed']]['acc'])*100 for r in rows]
    f1 = [(r['macro_f1']-raw[r['seed'],r['student_seed']]['macro_f1'])*100 for r in rows]
    split = {seed:mean(a for a,r in zip(acc,rows) if r['seed']==seed) for seed in v3.SEEDS}
    return dict(acc=mean(r['acc'] for r in rows)*100, f1=mean(r['macro_f1'] for r in rows)*100,
                delta=mean(acc), sd=stdev(acc), delta_f1=mean(f1), split=split,
                passes=mean(acc)>=2 and sum(x>0 for x in split.values())>=2 and mean(f1)>=0)


def feature_matrix():
    for ds in v3.DATASETS:
        for seed in v3.SEEDS:
            real, train, val = v3.load_client(ds,seed)
            raw = original(ds,seed,real,train)
            dims = (real.x_text.size(1),real.x_image.size(1))
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                teacher = sim.MultimodalSGC(dims,32,real.num_global_classes,2,0.,.5).to(DEVICE)
                sim.train_teacher(real,train,val,teacher,prior.ARGS)
            moments = [prior.stats(real,train,i) for i in range(2)]
            for lr,anchor in SETTINGS:
                name = key(lr,anchor)
                graph, distance = optimize(raw,teacher,moments,lr,anchor)
                assert all(torch.equal(graph[a],raw[a]) for a in ('edge_index','edge_weight','y','modality_mask'))
                torch.save(v3.graph_to_payload(graph),OUT/'cache'/f'{ds}_{seed}_{name}.pt')
                RESULT['features'].append(dict(dataset=ds,seed=seed,method=name,cosine_distance=distance))
                score(ds,seed,name,graph,real,val,RESULT['val'])
    RESULT['summary'] = {ds:{key(*s):summary(ds,key(*s)) for s in SETTINGS} for ds in v3.DATASETS}
    for ds, settings in RESULT['summary'].items():
        winners = [name for name,stat in settings.items() if stat['passes']]
        RESULT['winners'][ds] = max(winners,key=lambda name: settings[name]['delta']) if winners else None
    save()


def test_winners():
    for ds,name in RESULT['winners'].items():
        if name is None:
            continue
        for seed in v3.SEEDS:
            real,train,_ = v3.load_client(ds,seed)
            split = torch.load(v3.V1/f'split_{ds}_{seed}.pt',weights_only=False)
            test = split['test_mask'].to(DEVICE)
            raw = original(ds,seed,real,train)
            payload = torch.load(OUT/'cache'/f'{ds}_{seed}_{name}.pt',weights_only=False)
            graph = v3.graph_from_payload(payload,torch.device('cpu'))
            score(ds,seed,'raw',raw,real,test,RESULT['test'])
            score(ds,seed,name,graph,real,test,RESULT['test'])


def toys():
    old = [r for r in SPARSE['variants'] if r['dataset']=='Toys' and r['method']=='fixed']
    for seed in v3.SEEDS:
        real,_,val = v3.load_client('Toys',seed)
        orig = {r['student_seed']:r for r in old if r['seed']==seed and r['variant']=='orig'}
        if seed == 42:
            rows = [r for r in old if r['seed']==seed and r['variant']=='topk_sym_k1']
            degree = rows[0]['weighted_indegree']
        else:
            graph = v3.variant(v3.load_graph('Toys','fixed',seed),'topk_sym_k1')
            degree = float(graph.edge_weight.sum())/graph.num_nodes
            torch.save(v3.graph_to_payload(graph),OUT/'cache'/f'Toys_fixed_{seed}_k1.pt')
            rows = []
            score('Toys',seed,'k1',graph,real,val,rows)
        for r in rows:
            ref = orig[r['student_seed']]
            RESULT['toys_k1'].append(dict(**r,degree=degree,delta=(r['acc']-ref['acc'])*100,
                                          delta_f1=(r['macro_f1']-ref['macro_f1'])*100))
        save()


def report(unchanged):
    lines = ['# 固定边特征保真与 Toys k=1 复核','',
             f'源码和原始数据 SHA256 一致：{unchanged}。共 {len(RESULT["val"])} 条候选 val、{len(RESULT["test"])} 条 test、{len(RESULT["toys_k1"])} 条 k=1 配对。','',
             '沿用 V4 数据、边、300 轮特征优化、alpha=10 和三个学生种子。四档使用同划分同种子，raw 引用 V4 保存的评分。每个划分重新训练一次教师并共享给四档。余弦锚定项按两个模态平均。候选按 val ≥2 pp、至少两划分为正、macro-F1 不下降晋级。先冻结胜者再独立重训学生评分 test。','',
             '| 数据集 | lr / 锚定 | val % | macro-F1 % | 配对 Δ pp ± SD | macro-F1 Δ pp | 正划分 | 晋级 | 平均余弦距离 文本/图像 |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    for ds,stats in RESULT.get('summary',{}).items():
        for name,s in stats.items():
            distances = [r['cosine_distance'] for r in RESULT['features'] if r['dataset']==ds and r['method']==name]
            dist = '/'.join(f'{mean(d):.3f}' for d in zip(*distances))
            lines.append(f'| {ds} | {name} | {s["acc"]:.2f} | {s["f1"]:.2f} | {s["delta"]:+.2f} ± {s["sd"]:.2f} | {s["delta_f1"]:+.2f} | {sum(x>0 for x in s["split"].values())}/3 | {s["passes"]} | {dist} |')
    for ds,name in RESULT['winners'].items():
        lines += ['',f'{ds} 选定配置：{name or "无候选通过，保留原始特征"}。']
        if name:
            raw = {(r['seed'],r['student_seed']):r for r in RESULT['test'] if r['dataset']==ds and r['method']=='raw'}
            trial = [r for r in RESULT['test'] if r['dataset']==ds and r['method']==name]
            if trial:
                delta = [(r['acc']-raw[r['seed'],r['student_seed']]['acc'])*100 for r in trial]
                df1 = [(r['macro_f1']-raw[r['seed'],r['student_seed']]['macro_f1'])*100 for r in trial]
                lines += [f'独立 test 准确率 raw {mean(r["acc"] for r in raw.values())*100:.2f}% → 候选 {mean(r["acc"] for r in trial)*100:.2f}%，配对差 {mean(delta):+.2f} ± {stdev(delta):.2f} pp，macro-F1 差 {mean(df1):+.2f} pp。']
    lines += ['', '## Toys fixed k=1', '',
              'seed42 引用 V3，seed43/44 新跑三个学生种子。对照统一用 V3 对应 orig。', '',
              '| 划分 | 加权入度 | 准确率 Δ pp ± SD | macro-F1 Δ pp |', '|---|---:|---:|---:|']
    group = RESULT['toys_k1']
    for seed in v3.SEEDS:
        rows = [r for r in group if r['seed']==seed]
        if rows:
            delta = [r['delta'] for r in rows]
            lines.append(f'| {seed} | {rows[0]["degree"]:.3f} | {mean(delta):+.2f} ± {stdev(delta):.2f} | {mean(r["delta_f1"] for r in rows):+.2f} |')
    if len(group)==9:
        d = [r['delta'] for r in group]
        f = mean(r['delta_f1'] for r in group)
        positive = sum(mean(r['delta'] for r in group if r['seed']==s)>0 for s in v3.SEEDS)
        passed = all(.03<=r['degree']<=3 for r in group) and positive>=2 and mean(d)>=2 and f>0
        lines += ['', f'k=1 平均提升 {mean(d):+.2f} ± {stdev(d):.2f} pp，macro-F1 {f:+.2f} pp，{positive}/3 个划分为正。满足 Toys 结构门槛：{passed}。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    (OUT/'cache').mkdir(exist_ok=True)
    before = v3.manifest()
    try:
        feature_matrix()
        test_winners()
        toys()
        assert len(RESULT['val'])==72 and len(RESULT['toys_k1'])==9
    finally:
        after = v3.manifest()
        (OUT/'integrity.json').write_text(json.dumps(dict(before=before,after=after,unchanged=before==after),indent=2,ensure_ascii=False),encoding='utf-8')
        save()
        report(before==after)
