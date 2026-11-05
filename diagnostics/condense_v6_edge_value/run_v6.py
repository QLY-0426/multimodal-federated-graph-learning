"""V6 edge-value and class-balanced feature diagnostics."""
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
V5 = OUT.parent/'condense_v5_feature_preservation'
sys.dont_write_bytecode = True
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0,str(V5))
import run_v5 as prior
from run_v5 import F, sim, torch, v3

(OUT/'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT/'tmp')
DEVICE = v3.DEVICE
NAME = 'lr0.005_anchor0'
METHODS = ('B','C1','C2','C3')
STUDENTS = prior.STUDENTS
V5_RESULTS = json.loads((V5/'results.json').read_text(encoding='utf-8'))
R = dict(toys_val=[],toys_test=[],degrees=[],movies_val=[],movies_test=[],summary={})


def save():
    (OUT/'results.json').write_text(json.dumps(R,indent=2),encoding='utf-8')


def graph_v5(ds, seed):
    payload = torch.load(V5/'cache'/f'{ds}_{seed}_{NAME}.pt',weights_only=False,map_location='cpu')
    return v3.graph_from_payload(payload,torch.device('cpu'))


def degree(graph):
    return float(graph.edge_weight.sum())/graph.num_nodes


def metrics(pred, target, classes):
    result = v3.classification(pred,target,classes)
    cm = torch.bincount(target.cpu()*classes+pred.cpu(),minlength=classes**2).reshape(classes,classes)
    support = cm.sum(1)
    result['confusion'] = cm.tolist()
    result['recall'] = [float(cm[i,i]/support[i]) if support[i] else None for i in range(classes)]
    return result


def evaluate(ds, seed, method, graph, real, mask, dest):
    args = SimpleNamespace(hid_dim=32,num_layers=2,dropout=0.,lr=.01,weight_decay=0.,
                           multimodal_fusion_temperature=.5,multimodal_student_epochs=500)
    for student in STUDENTS:
        args.seed = student
        captured = {}
        def capture(model, data, keep):
            model.eval()
            with torch.no_grad():
                keep = keep & data.modality_mask.any(1)
                pred = model(data)[1][keep].argmax(1)
                y = data.y[keep]
                captured.update(metrics(pred,y,real.num_global_classes))
                return float((pred==y).float().mean())
        with patch.object(sim,'_accuracy',capture):
            acc = sim._validate_synthetic(graph.clone().to(DEVICE),real,mask,args,
                                           (real.x_text.size(1),real.x_image.size(1)),real.num_global_classes)
        row = dict(dataset=ds,seed=seed,method=method,student_seed=student,acc=acc,
                   weighted_indegree=degree(graph),**captured)
        R[dest].append(row)
        print('SCORE',dest,ds,seed,method,student,acc,captured['macro_f1'],flush=True)
        save()


def teacher_for(real,train,val,seed):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        model = sim.MultimodalSGC((real.x_text.size(1),real.x_image.size(1)),
                                  32,real.num_global_classes,2,0.,.5).to(DEVICE)
        sim.train_teacher(real,train,val,model,prior.prior.ARGS)
    return model


def balanced_loss(model, graph, mask, class_weight):
    _, fused, branches, _ = model.forward_modalities(graph)
    losses = []
    for i in range(2):
        selected = mask & graph.modality_mask[:,i]
        if selected.any():
            losses.append(F.cross_entropy(branches[selected,i],graph.y[selected],weight=class_weight))
    return (F.cross_entropy(fused[mask],graph.y[mask],weight=class_weight)
            + sim.config['branch_weight']*torch.stack(losses).mean())


def optimize(raw, teacher, moments, method, seed, topology=None, class_weight=None):
    graph = raw.clone().to(DEVICE)
    params = [torch.nn.Parameter(F.normalize(x)) for x in (graph.x_text,graph.x_image)]
    graph.x_text,graph.x_image = params
    feature_opt = torch.optim.Adam(params,lr=.005)
    n = graph.num_nodes
    edge_opt = None
    if method == 'B':
        edges = topology.edge_index.to(DEVICE)
        pairs = torch.minimum(edges[0],edges[1])*n + torch.maximum(edges[0],edges[1])
        _, inverse = pairs.unique(return_inverse=True)
        weights = topology.edge_weight.to(DEVICE)
        sums = weights.new_zeros(int(inverse.max())+1).index_add_(0,inverse,weights)
        counts = torch.bincount(inverse,minlength=len(sums))
        logits = torch.nn.Parameter(torch.logit((sums/counts).clamp(.001,.999)))
        edge_opt = torch.optim.Adam([logits],lr=sim.config['lr_adj'])
    elif method.startswith('C'):
        k = int(method[1:])
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            pge = sim.PGE(sum(x.size(1) for x in params),n,nhid=32,device=DEVICE,
                          args=SimpleNamespace(dataset=['Toys'])).to(DEVICE)
        edge_opt = torch.optim.Adam(pge.parameters(),lr=sim.config['lr_adj'])
        diagonal = torch.eye(n,dtype=torch.bool,device=DEVICE)
    mask = graph.modality_mask.any(1)
    min_degree,max_degree = float('inf'),0.

    def set_edges():
        if method == 'B':
            weight = logits.sigmoid()[inverse]
            graph.edge_index = edges
        else:
            paired = torch.cat([x*graph.modality_mask[:,i:i+1] for i,x in enumerate(params)],1)
            scores = pge(paired)
            target = scores.masked_fill(diagonal,-torch.inf).topk(k,dim=1).indices
            keep = torch.zeros_like(diagonal).scatter_(1,target,True)
            graph.edge_index = (keep | keep.T).nonzero().T.contiguous()
            weight = scores[tuple(graph.edge_index)]
        # Cap graph-wide mean weighted indegree without changing topology.
        mass = weight.sum()/n
        graph.edge_weight = weight*(mass.clamp(.03001,2.99999)/mass.clamp_min(1e-12))

    for step in range(300):
        feature_opt.zero_grad()
        if edge_opt:
            edge_opt.zero_grad()
            set_edges()
            actual = float(graph.edge_weight.detach().sum()/n)
            min_degree,max_degree = min(actual,min_degree),max(actual,max_degree)
        loss = (sim._loss(teacher,graph,mask) if class_weight is None
                else balanced_loss(teacher,graph,mask,class_weight))
        for i,x in enumerate(params):
            present = graph.modality_mask[:,i]
            features = sim._propagated_features(x,graph.edge_index,graph.edge_weight,present,2)
            loss = loss+10*sim._alignment(moments[i],features,graph.y,present)
        assert torch.isfinite(loss)
        loss.backward()
        feature_opt.step()
        if edge_opt:
            edge_opt.step()
        if step in (0,99,199,299):
            print('OPTIMIZE',seed,method,step+1,float(loss.detach()),flush=True)
    with torch.no_grad():
        if edge_opt:
            set_edges()
        graph.x_text,graph.x_image = (x.detach() for x in params)
        graph.x = torch.cat((F.normalize(graph.x_image),F.normalize(graph.x_text)),1)
    graph = graph.cpu()
    actual = degree(graph)
    return graph,dict(final=actual,minimum=min(min_degree,actual),maximum=max(max_degree,actual))


def save_graph(ds,seed,method,graph):
    torch.save(v3.graph_to_payload(graph),OUT/'cache'/f'{ds}_{seed}_{method}.pt')


def test_mask(ds,seed):
    return torch.load(v3.V1/f'split_{ds}_{seed}.pt',weights_only=False)['test_mask'].to(DEVICE)


def compare(rows, baseline):
    refs = {(r['seed'],r['student_seed']):r for r in baseline}
    deltas = [(r['acc']-refs[r['seed'],r['student_seed']]['acc'])*100 for r in rows]
    df1 = [(r['macro_f1']-refs[r['seed'],r['student_seed']]['macro_f1'])*100 for r in rows]
    return dict(acc=mean(r['acc'] for r in rows)*100,f1=mean(r['macro_f1'] for r in rows)*100,
                majority=mean(r['predicted_majority_share'] for r in rows)*100,
                delta=mean(deltas),sd=stdev(deltas),delta_f1=mean(df1),
                splits={s:mean(d for d,r in zip(deltas,rows) if r['seed']==s) for s in v3.SEEDS},
                pairs=[dict(seed=r['seed'],student_seed=r['student_seed'],delta=d,delta_f1=f)
                       for r,d,f in zip(rows,deltas,df1)])


def toys():
    for seed in v3.SEEDS:
        real,train,val = v3.load_client('Toys',seed)
        raw = prior.original('Toys',seed,real,train)
        a = graph_v5('Toys',seed)
        assert torch.equal(a.edge_index,raw.edge_index)
        for source, dest in [('val','toys_val'),('test','toys_test')]:
            R[dest].extend(dict(r,method='A',weighted_indegree=degree(a),recall=None,confusion=None)
                           for r in V5_RESULTS[source] if r['dataset']=='Toys' and r['seed']==seed and r['method']==NAME)
        old = v3.load_graph('Toys','fixed',seed)
        assert torch.equal(old.y,raw.y)
        teacher = teacher_for(real,train,val,seed)
        moments = [prior.prior.stats(real,train,i) for i in range(2)]
        for method in METHODS:
            graph, budget = optimize(raw,teacher,moments,method,seed,topology=old)
            assert torch.equal(graph.y,raw.y) and torch.equal(graph.modality_mask,raw.modality_mask)
            if method=='B':
                assert torch.equal(graph.edge_index,old.edge_index)
            R['degrees'].append(dict(seed=seed,method=method,edges=graph.num_edges,**budget))
            print('BUDGET',R['degrees'][-1],flush=True)
            save_graph('Toys',seed,method,graph)
            evaluate('Toys',seed,method,graph,real,val,'toys_val')
    base = [r for r in R['toys_val'] if r['method']=='A']
    stats = {m:compare([r for r in R['toys_val'] if r['method']==m],base) for m in METHODS}
    for m,s in stats.items():
        s['budget_ok'] = all(.03<=b['minimum'] and b['maximum']<=3 for b in R['degrees'] if b['method']==m)
        s['val_pass'] = s['budget_ok'] and s['delta']>=2 and s['delta_f1']>=0 and sum(d>0 for d in s['splits'].values())>=2
    eligible = [m for m in METHODS if stats[m]['budget_ok']]
    passed = [m for m in eligible if stats[m]['val_pass']]
    winner = max(passed or eligible,key=lambda m:stats[m]['delta']) if eligible else None
    R['summary']['toys_val'] = stats
    R['summary']['toys_winner'] = winner
    print('FROZEN WINNER',winner,flush=True)
    save()
    if winner:
        for seed in v3.SEEDS:
            real,_,_ = v3.load_client('Toys',seed)
            payload = torch.load(OUT/'cache'/f'Toys_{seed}_{winner}.pt',weights_only=False)
            graph = v3.graph_from_payload(payload,torch.device('cpu'))
            evaluate('Toys',seed,winner,graph,real,test_mask('Toys',seed),'toys_test')
        R['summary']['toys_test'] = compare([r for r in R['toys_test'] if r['method']==winner],
                                          [r for r in R['toys_test'] if r['method']=='A'])
        save()


def aggregate_classes(rows):
    cm = torch.tensor([r['confusion'] for r in rows]).sum(0)
    total = cm.sum(1)
    return dict(confusion=cm.tolist(),actual=total.tolist(),prediction=cm.sum(0).tolist(),
                recall=[float(cm[i,i]/total[i]) if total[i] else None for i in range(len(total))])


def movies():
    for seed in v3.SEEDS:
        real,train,val = v3.load_client('Movies',seed)
        raw = prior.original('Movies',seed,real,train)
        evaluate('Movies',seed,'raw',raw,real,val,'movies_val')
        evaluate('Movies',seed,'A',graph_v5('Movies',seed),real,val,'movies_val')
    evidence = {m:aggregate_classes([r for r in R['movies_val'] if r['method']==m]) for m in ('raw','A')}
    (OUT/'movies_confusion_before_balance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('MOVIES CLASS EVIDENCE', {m:e['recall'] for m,e in evidence.items()},flush=True)
    for seed in v3.SEEDS:
        real,train,val = v3.load_client('Movies',seed)
        raw = prior.original('Movies',seed,real,train)
        teacher = teacher_for(real,train,val,seed)
        weights = torch.bincount(real.y[train],minlength=real.num_global_classes).float().clamp_min(1).reciprocal()
        weights = weights/weights.mean()
        graph,_ = optimize(raw,teacher,[prior.prior.stats(real,train,i) for i in range(2)],
                           'balanced',seed,class_weight=weights)
        save_graph('Movies',seed,'balanced',graph)
        evaluate('Movies',seed,'balanced',graph,real,val,'movies_val')
        mask = test_mask('Movies',seed)
        evaluate('Movies',seed,'raw',raw,real,mask,'movies_test')
        evaluate('Movies',seed,'A',graph_v5('Movies',seed),real,mask,'movies_test')
    base = [r for r in R['movies_val'] if r['method']=='raw']
    R['summary']['movies_balance'] = compare([r for r in R['movies_val'] if r['method']=='balanced'],base)
    R['summary']['movies_A_val'] = compare([r for r in R['movies_val'] if r['method']=='A'],base)
    R['summary']['movies_test'] = compare([r for r in R['movies_test'] if r['method']=='A'],
                                        [r for r in R['movies_test'] if r['method']=='raw'])
    save()


def report(ok, status):
    s = R['summary']
    lines = ['# V6 PGE 边价值复核','',status,'',f'生产源码与原始数据的 SHA256 一致：{ok}。见 integrity.json。','',
             'A 引用 V5 的 Toys val/test，未重新训练 A。V5 保存了多数类占比和 macro-F1，未保存混淆矩阵，A 逐类召回记为缺失。B 按相同类标签顺序将旧 PGE 缓存拓扑移植到 V5 节点槽位，冻结边集合，按无向边对直接学习 sigmoid 权重。旧缓存没有原始节点映射。C 使用生产 PGE 评分，在每轮按出边 top-k 取对称并集，保留权重梯度，PGE 与特征同步更新。B/C 每轮将图级平均加权入度投影至 0.03–3.0，包含训练和最终评分。','',
             '沿用 V5 的划分与节点、300 轮特征优化、alpha=10、lr_feat=.005、教师 200 轮、学生 500 轮及 500/501/502。V5 无教师快照，按同种子同协议恢复，每划分 B/C 共用一份教师。A 引用旧运行，GPU 重训存在数值波动。Movies 均衡项对融合及模态分类交叉熵使用真实训练类频率的倒数。','',
             '晋级门槛为预算合规、val 相对 A 至少 +2 pp、至少 2/3 划分为正、macro-F1 不降且 test 准确率差为正。冻结候选时先在 val 达标者中取最高准确率；无达标者则在预算合规档里取最高准确率，仅用于闭卷复核。报告判定中“接近”定义为 val 准确率及 macro-F1 均距 A 不超过 1 pp 的下行差。Movies 均衡档要求 macro-F1 不降且准确率下降不超过 1 pp。Movies test 仅报告。','',
             '## Toys A/B/C 配对对照','',
             '| 档位 | val % | macro-F1 % | 多数类预测 % | Δ acc pp ± SD | Δ macro-F1 pp | seed42/43/44 Δ pp | 最终加权入度 | val 晋级 |',
             '|---|---:|---:|---:|---|---:|---|---|---|']
    base = [r for r in R['toys_val'] if r['method']=='A']
    if base:
        lines.append(f'| A | {mean(r["acc"] for r in base)*100:.2f} | {mean(r["macro_f1"] for r in base)*100:.2f} | {mean(r["predicted_majority_share"] for r in base)*100:.2f} | 0 | 0 | 0/0/0 | '+ '/'.join(f'{r["weighted_indegree"]:.3f}' for r in base[::3])+' | 基线 |')
    for m,p in s.get('toys_val',{}).items():
        deg = '/'.join(f'{b["final"]:.3f}' for b in R['degrees'] if b['method']==m)
        delta = '/'.join(f'{x:+.2f}' for x in p['splits'].values())
        lines.append(f'| {m} | {p["acc"]:.2f} | {p["f1"]:.2f} | {p["majority"]:.2f} | {p["delta"]:+.2f} ± {p["sd"]:.2f} | {p["delta_f1"]:+.2f} | {delta} | {deg} | {p["val_pass"]} |')
    winner = s.get('toys_winner')
    if 'toys_test' in s:
        t=s['toys_test']
        passed=s['toys_val'][winner]['val_pass'] and t['delta']>0
        lines += ['',f'冻结 {winner}。test 相对 A 准确率 {t["delta"]:+.2f} ± {t["sd"]:.2f} pp，macro-F1 {t["delta_f1"]:+.2f} pp。通过全部门槛：{passed}。']
    lines += ['', 'Toys 的逐类召回、混淆矩阵、每个学生配对差及预算范围见 results.json。','',
              '## Movies 混淆矩阵与逐类召回','',
              '完整混淆矩阵逐次保存在 results.json（行是真实类，列是预测类）。均衡优化前的汇总矩阵另见 movies_confusion_before_balance.json。下表汇总九次评分的计数。','',
              '| 类别 | 真实计数 | raw 预测/召回 | A 预测/召回 | balanced 预测/召回 |','|---:|---:|---|---|---|']
    classes={}
    for m in ('raw','A','balanced'):
        rows = [r for r in R['movies_val'] if r['method']==m]
        if rows:
            classes[m]=aggregate_classes(rows)
    if 'raw' in classes:
        for i,total in enumerate(classes['raw']['actual']):
            vals=[]
            for m in ('raw','A','balanced'):
                p=classes.get(m)
                vals.append(f'{p["prediction"][i]} / {p["recall"][i]:.2%}' if p and p['recall'][i] is not None else '—')
            lines.append(f'| {i} | {total} | '+' | '.join(vals)+' |')
    for m,p in classes.items():
        zero=[str(i) for i,x in enumerate(p['recall']) if x==0.]
        lines += ['',f'{m} 零召回类为 {", ".join(zero) or "无"}。']
    if 'movies_balance' in s:
        m=s['movies_balance']
        lines += ['',f'均衡档相对 raw 的 val 准确率 {m["delta"]:+.2f} pp，macro-F1 {m["delta_f1"]:+.2f} pp。通过门槛：{m["delta"]>=-1 and m["delta_f1"]>=0}。']
    lines += ['', '## Movies 固定真实边 test','',
              '| 方法 | test acc % | macro-F1 % | 配对 Δ pp ± SD |','|---|---:|---:|---|']
    if 'movies_test' in s:
        for m in ('raw','A'):
            rows=[r for r in R['movies_test'] if r['method']==m]
            t=s['movies_test']
            delta='0' if m=='raw' else f'{t["delta"]:+.2f} ± {t["sd"]:.2f}'
            lines.append(f'| {m} | {mean(r["acc"] for r in rows)*100:.2f} | {mean(r["macro_f1"] for r in rows)*100:.2f} | {delta} |')
        lines += ['',f'Movies A 的 test macro-F1 配对差为 {s["movies_test"]["delta_f1"]:+.2f} pp。']
    if 'toys_val' in s and 'movies_test' in s:
        c=[m for m in ('C1','C2','C3') if s['toys_val'][m]['budget_ok'] and
           s['toys_val'][m]['delta']>=-1 and s['toys_val'][m]['delta_f1']>=-1]
        b=s['toys_val']['B']
        test_bad = s.get('toys_test',{}).get('delta',0)<-1
        if c and not (winner in c and test_bad):
            lines += ['', '**一句话判定 1：受度预算约束的 PGE 学边接近或超过固定真实边，可继续验证。**',
                      '生产落点为 openfgl/flcore/multimodal_fedgvd/SimGC_transductive.py 的 train_syn 第 180 行和第 216 行，紧接 PGE 打分后共用 top-k 对称并集和度预算投影，分类、传播对齐与验收使用同一规则。能否晋级仍以完整 test 门槛为准。']
        elif b['budget_ok'] and b['delta']>=-1 and b['delta_f1']>=-1 and not (winner=='B' and test_bad):
            lines += ['', '**一句话判定 2：冻结 PGE 边集合后只学权重可行，动态学边不采用。**',
                      '下一轮在 train_syn 初始化时锁定候选集合，训练和验收只更新/读取该集合的边权。']
        else:
            lines += ['', '**一句话判定 3：本设置下 PGE 自动学边无效，采用节点抽样 + 特征优化。**',
                      '此处判定的是自动学边的增量价值，Movies 固定边的 test 收益独立报告。已观察到的收益依赖抽取真实节点及其初始特征，固定边优化和缓存删边不能证明学习结构的图凝缩有效。停止在 Movies/Toys 上继续调边。用 Cora/Citeseer 的公开固定划分或可复现社区客户端划分复验，先量化 real GCN 相对无边 MLP 的结构收益；再固定真实节点配额和初始化，对比随机诱导图、固定边特征优化和预算内 PGE，按同划分同种子配对并在独立 test 核验。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    (OUT/'cache').mkdir(exist_ok=True)
    before = v3.manifest()
    status = '已完成'
    try:
        toys()
        movies()
        assert len(R['toys_val'])==45 and len(R['movies_val'])==27 and len(R['movies_test'])==18
    except Exception as error:
        status=f'运行中断：{type(error).__name__} {error}'
        raise
    finally:
        after=v3.manifest()
        (OUT/'integrity.json').write_text(json.dumps(dict(before=before,after=after,unchanged=before==after),indent=2,ensure_ascii=False),encoding='utf-8')
        save()
        report(before==after,status)
