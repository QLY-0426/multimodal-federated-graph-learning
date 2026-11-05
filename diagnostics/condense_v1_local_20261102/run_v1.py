"""Read-only real-data local comparison; no production source edits."""
import os
from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
(OUT / 'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT / 'tmp')
sys.path.insert(0, str(OUT.parents[1]))
import contextlib
import csv
import hashlib
import json
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from openfgl.flcore.multimodal_fedgvd.SimGC_transductive import data_gc, generate_labels_syn, _validate_synthetic
from openfgl.flcore.multimodal_fedgvd.fedgvd_config import config
from openfgl.flcore.multimodal_fedgvd.utils import graph_to_payload

SOURCE = Path(r'D:\下载学习软件\韩旭涛_科研项目资料\跑FedMVP实验\10_29跑实验\FedGVD-multimodal_fedgvd\dataset\global\subgraph_fl')
BASE_CONFIG = dict(config)
METHODS = ['legacy', 'fixed', 'random_baseline']
RUN_ID = time.strftime('%Y%m%d_%H%M%S')
CACHE = OUT / 'cache' / RUN_ID
CACHE.mkdir(parents=True)
ROWS, META = [], []


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_client(dataset, seed):
    raw = SOURCE / dataset / 'raw'
    labels = np.load(raw / 'labels.npy', allow_pickle=False)
    g = torch.Generator().manual_seed(seed)
    y = torch.from_numpy(labels.astype(np.int64))
    selected = torch.cat([ids[torch.randperm(len(ids), generator=g)[:max(3, int(np.ceil(len(ids)*.1)))]]
                          for c in y.unique(sorted=True) for ids in [torch.where(y == c)[0]]]).sort().values
    x_text = torch.from_numpy(np.array(np.load(raw/'text_features.npy', mmap_mode='r')[selected.numpy()], dtype=np.float32))
    x_image = torch.from_numpy(np.array(np.load(raw/'image_features.npy', mmap_mode='r')[selected.numpy()], dtype=np.float32))
    edges = torch.from_numpy(np.load(raw/'edge_index.npy', allow_pickle=False).astype(np.int64))
    edges, _ = subgraph(selected, edges, relabel_nodes=True, num_nodes=len(y))
    y = y[selected]
    mask = torch.stack((x_text.norm(dim=1) > 0, x_image.norm(dim=1) > 0), dim=1)
    assert torch.isfinite(x_text).all() and torch.isfinite(x_image).all() and mask.any(1).all()
    data = Data(x=torch.cat((F.normalize(x_image,dim=1), F.normalize(x_text,dim=1)),1),
                x_text=x_text, x_image=x_image, y=y, modality_mask=mask, edge_index=edges,
                edge_weight=torch.ones(edges.size(1)), num_global_classes=int(labels.max())+1,
                node_id=selected)
    splits = [torch.zeros(len(y), dtype=torch.bool) for _ in range(3)]
    for c in y.unique(sorted=True):
        ids = torch.where(y == c)[0]
        ids = ids[torch.randperm(len(ids), generator=g)]
        a, b = max(1,int(.6*len(ids))), max(1,int(.2*len(ids)))
        splits[0][ids[:a]] = True
        splits[1][ids[a:a+b]] = True
        splits[2][ids[a+b:]] = True
    p = dict(data=data, **dict(zip(('train_mask','val_mask','test_mask'),splits)))
    torch.save(dict(node_ids=selected, **{k:v for k,v in p.items() if k != 'data'}), OUT/f'split_{dataset}_{seed}.pt')
    META.append(dict(dataset=dataset,seed=seed,global_nodes=len(labels),client_nodes=len(y),
                     edges=edges.size(1),classes=data.num_global_classes,
                     splits=[int(s.sum()) for s in splits], missing_modalities=int((~mask).sum()),
                     synthetic_nodes=len(generate_labels_syn(y[splits[0]],.2))))
    return p


def random_graph(p, seed):
    d = p['data']
    labels = generate_labels_syn(d.y[p['train_mask']], .2)
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([candidates[torch.randperm(len(candidates),generator=g)[:int((labels==c).sum())]]
                     for c in labels.unique(sorted=True)
                     for candidates in [torch.where(p['train_mask'] & (d.y==c))[0]]])
    edges, weights = subgraph(ids, d.edge_index, d.edge_weight, relabel_nodes=True, num_nodes=d.num_nodes)
    graph = Data(x=d.x[ids],x_text=d.x_text[ids],x_image=d.x_image[ids],y=d.y[ids],
                 modality_mask=d.modality_mask[ids],edge_index=edges,edge_weight=weights)
    assert torch.equal(graph.y, labels)
    return graph, d.node_id[ids]


def write_results(final=False):
    (OUT/'results.json').write_text(json.dumps(ROWS, indent=2),encoding='utf-8')
    (OUT/'data_manifest.json').write_text(json.dumps(META, indent=2),encoding='utf-8')
    if ROWS:
        with (OUT/'results.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=ROWS[0].keys()); writer.writeheader(); writer.writerows(ROWS)
    lines = ['# 本地小型 V1 增益对比（2026-11-02）','',
        '状态：' + ('全部 18 个 run 已完成。' if final else f'运行中，已完成 {len(ROWS)}/18。'),'',
        '## 协议','',
        f'- 数据只读来源：`{SOURCE}`。直接加载 raw/*.npy，绕过 FL 划分/pymetis；未实例化会写 processed 的 Dataset。',
        '- 每数据集每 seed 按类比例随机取 ceil(10%) 节点作为 1 个客户端，保留真实有向诱导边，不额外补边/无向化。随后逐类按 60%/20%/余量划分 train/val/test。三个方法共享完全相同的数据与划分，划分保存为 split_*.pt。',
        '- 使用原始文本/图像 CLIP 特征，零向量视为缺失模态，没有人为注入缺失。数据统计见下表。',
        f'- 设备 `{DEVICE}`（{torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "CPU"}），torch={torch.__version__}，Python={sys.version.split()[0]}。',
        f'- teacher={EPOCHS[0]} / condense={EPOCHS[1]} / student={EPOCHS[2]} / validation_interval={EPOCHS[3]}。GPU 按原定轮数；CPU 时全部减半。',
        '- 沿用 V0 学生超参数：hid_dim=32，num_layers=2，dropout=0，lr=.01，weight_decay=0，fusion_temperature=.5；reduction_rate=.2；其他凝缩 config 保持原值（feat_alpha=10）。',
        '- legacy: adj_mode=legacy, alternating_optim=False；fixed: consistent_low + True。只在脚本进程中修改 config，不写回源码。',
        '- random_baseline 按 generate_labels_syn 的 ceil 类配额，从真实训练节点随机抽取并保留真实诱导边/特征/模态掩码；不额外训练教师。',
        '- 三者最终学生均调用同一个 _validate_synthetic，训练固定轮数，固定 seed 初始化，在同一真实客户端 val 上评分。凝缩图先由 data_gc 按六次候选的 val 选最佳，再重训同 seed 学生评分；随机基线仅抽一次。',
        '- 因 teacher/合成图选择使用了 val，本表是开发验证集结果，存在选择偏差，不是独立测试集泛化证据；不使用 test acc 选模型。单客户端随机抽样会使边稀疏，不能代替正式十客户端划分。',
        f'- 缓存目录 cache/{RUN_ID}/，每 dataset×method×seed 独立文件；condense_version=2。日志为 logs/*.log。',
        '- 启动前 nvidia-smi 已确认 RTX 4060；TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1；无安装新依赖。','',
        '## 客户端统计','','| dataset | seed | nodes | edges | train/val/test | syn nodes | missing modalities |','|---|---:|---:|---:|---|---:|---:|']
    for m in META:
        lines.append(f'| {m["dataset"]} | {m["seed"]} | {m["client_nodes"]} | {m["edges"]} | {m["splits"]} | {m["synthetic_nodes"]} | {m["missing_modalities"]} |')
    lines += ['','## 逐次结果','','| dataset | method | seed | real val acc (%) | syn edges |','|---|---|---:|---:|---:|']
    for r in ROWS:
        lines.append(f'| {r["dataset"]} | {r["method"]} | {r["seed"]} | {r["val_acc"]*100:.4f} | {r["syn_edges"]} |')
    lines += ['','## 均值 ± 样本标准差（ddof=1）','','| dataset | method | n | val acc (%) |','|---|---|---:|---|']
    for ds in ['Movies','Toys']:
        for method in METHODS:
            vals = [r['val_acc']*100 for r in ROWS if r['dataset']==ds and r['method']==method]
            if len(vals)>1:
                lines.append(f'| {ds} | {method} | {len(vals)} | {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} |')
    lines += ['','## 凝缩增益（同 seed 配对差，单位百分点）','','| dataset | seed | fixed − random | legacy − random | fixed − legacy |','|---|---|---:|---:|---:|']
    for ds in ['Movies','Toys']:
        deltas = []
        if ds == 'Toys':
            lines += ['', '| dataset | seed | fixed − random | legacy − random | fixed − legacy |', '|---|---|---:|---:|---:|']
        for seed in [42,43,44]:
            values = {r['method']:r['val_acc']*100 for r in ROWS if r['dataset']==ds and r['seed']==seed}
            if len(values)==3:
                d = (values['fixed']-values['random_baseline'],values['legacy']-values['random_baseline'],values['fixed']-values['legacy'])
                deltas.append(d)
                lines.append(f'| {ds} | {seed} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} |')
        if len(deltas)>1:
            stats = [f'{statistics.mean(x):+.4f} ± {statistics.stdev(x):.4f}' for x in zip(*deltas)]
            lines.append(f'| {ds} | mean ± sd | ' + ' | '.join(stats) + ' |')
            if final:
                a,b,c = map(statistics.mean,zip(*deltas))
                lines += ['', f'{ds}：修复档相对随机基线平均 {a:+.4f} 个百分点，相对 legacy {c:+.4f} 个百分点；' + ('本次均值支持正向改善，但仅限当前小型协议。' if a>0 and c>0 else '本次结果未同时证明超过随机基线与 legacy，不能宣称修复有效。'), '']
    if final:
        lines += ['','## 结论边界与下一步','',
                  '三种子不足以得出稳健统计显著性结论，未做真实十客户端联邦聚合。若修复无稳定提升，下一步应固定划分消融 adj_mode × alternating_optim × feat_alpha∈{10,1}，并使用独立 test 评估及正式客户端划分。此轮按要求不改源码或自动追加消融。',
                  '输入数据文件和三个生产源码文件运行前后 SHA256 一致性见 integrity.json；全部执行日志、缓存、随机抽样节点 ID 与分割已保留。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    global DEVICE, EPOCHS
    torch.set_num_threads(2)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = (200,300,500,50) if DEVICE.type=='cuda' else (100,150,250,25)
    sources = [p for ds in ('Movies','Toys') for p in (SOURCE/ds).rglob('*') if p.is_file()]
    sources += [OUT.parents[1]/'openfgl/flcore/multimodal_fedgvd'/name for name in ('SimGC_transductive.py','fedgvd_config.py','utils.py')]
    before = {str(p):sha(p) for p in sources}
    (OUT/'integrity_before.json').write_text(json.dumps(before,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'logs').mkdir(exist_ok=True)
    print(f'DEVICE={DEVICE} EPOCHS={EPOCHS}',flush=True)
    try:
        for dataset in ['Movies','Toys']:
            for seed in [42,43,44]:
                p = load_client(dataset,seed)
                print('DATA',META[-1],flush=True)
                for method in METHODS:
                    start = time.time()
                    config.clear(); config.update(BASE_CONFIG)
                    config.update(adj_mode='consistent_low' if method=='fixed' else 'legacy',alternating_optim=method=='fixed')
                    args = SimpleNamespace(dataset=[dataset],seed=seed,num_clients=10,hid_dim=32,num_layers=2,
                        dropout=0.,lr=.01,weight_decay=0.,multimodal_reduction_rate=.2,
                        multimodal_teacher_epochs=EPOCHS[0],multimodal_condense_epochs=EPOCHS[1],
                        multimodal_student_epochs=EPOCHS[2],multimodal_validation_interval=EPOCHS[3],
                        multimodal_fusion_temperature=.5,
                        multimodal_condensed_pattern=str(CACHE/f'{dataset}_{method}_{seed}.pt'))
                    print(f'START {dataset} {method} {seed}',flush=True)
                    with (OUT/'logs'/f'{dataset}_{method}_{seed}.log').open('w',encoding='utf-8',buffering=1) as log, contextlib.redirect_stdout(log):
                        if method == 'random_baseline':
                            graph, selected = random_graph(p,seed)
                            torch.save({'graph':graph_to_payload(graph),'node_ids':selected}, CACHE/f'{dataset}_{method}_{seed}.pt')
                            graph = graph.to(DEVICE)
                        else:
                            graph = data_gc(p,0,args,DEVICE)
                        real = p['data'].to(DEVICE)
                        score = _validate_synthetic(graph,real,p['val_mask'].to(DEVICE),args,
                            (real.x_text.size(1),real.x_image.size(1)),real.num_global_classes)
                    ROWS.append(dict(dataset=dataset,method=method,seed=seed,val_acc=score,
                        syn_nodes=graph.num_nodes,syn_edges=graph.num_edges,seconds=round(time.time()-start,2)))
                    print('RESULT',ROWS[-1],flush=True)
                    write_results()
                    del graph
                    p = {k:(v.cpu() if hasattr(v,'cpu') else v) for k,v in p.items()}
                    if DEVICE.type=='cuda': torch.cuda.empty_cache()
        after = {str(p):sha(p) for p in sources}
        assert before == after, 'Input or production source changed during experiment'
        (OUT/'integrity.json').write_text(json.dumps({'all_unchanged':True,'before':before,'after':after},indent=2,ensure_ascii=False),encoding='utf-8')
        write_results(final=True)
        print('ALL 18 RUNS COMPLETE; source/data SHA256 unchanged',flush=True)
    finally:
        config.clear(); config.update(BASE_CONFIG)


if __name__ == '__main__':
    main()
