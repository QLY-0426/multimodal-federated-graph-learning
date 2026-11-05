"""Check completed results and append teacher/edge diagnostics to the report."""
import csv
import json
from pathlib import Path
import re

out = Path(__file__).resolve().parent
rows = json.loads((out/'results.json').read_text(encoding='utf-8'))
assert len(rows) == 18 and len({(r['dataset'],r['method'],r['seed']) for r in rows}) == 18
assert json.loads((out/'integrity.json').read_text(encoding='utf-8'))['all_unchanged']
diagnostics = []
for row in rows:
    if row['method'] == 'random_baseline':
        continue
    log = (out/'logs'/f'{row["dataset"]}_{row["method"]}_{row["seed"]}.log').read_text(encoding='utf-8')
    assert 'cache_hit=True' not in log
    epochs = re.findall(r'\[condense\] epoch=(\d+) loss=([\d.]+) validation=([\d.]+)', log)
    assert len(epochs) == 6 and epochs[-1][0] == '300'
    selection_val = max(float(v) for _,_,v in epochs)
    acc = re.findall(r'teacher_on_syn_acc=([\d.]+)',log)
    assert len(acc) == 7
    last = re.findall(r'epoch=(\d+) weight.mean=([\d.]+) fraction_ge_adj_threshold=([\d.]+) retained_edges=(\d+) teacher_on_syn_acc=([\d.]+)',log)[-1]
    shifts = re.search(r'final_text_relative_shift=([\d.]+) final_image_relative_shift=([\d.]+)',log)
    diagnostics.append(dict(dataset=row['dataset'],method=row['method'],seed=row['seed'],
        teacher_initial=float(acc[0]),teacher_final=float(acc[-1]),final_weight_mean=float(last[1]),
        final_fraction_ge_05=float(last[2]),final_edges=int(last[3]),
        text_shift=float(shifts[1]),image_shift=float(shifts[2]),
        selection_val=selection_val,retrained_val=row['val_acc'],
        retrain_delta_pp=(row['val_acc']-selection_val)*100))
with (out/'diagnostics_summary.csv').open('w',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f,fieldnames=diagnostics[0]);writer.writeheader();writer.writerows(diagnostics)
report = (out/'REPORT.md').read_text(encoding='utf-8')
marker = '\n## 凝缩诊断汇总'
report = report.split(marker)[0]
report += marker + '\n\n| dataset | method | seed | teacher syn 初始→最终 (%) | 最终 weight.mean | 最终边数 | 文本位移 | 图像位移 |\n|---|---|---:|---|---:|---:|---:|---:|\n'
for d in diagnostics:
    report += f'| {d["dataset"]} | {d["method"]} | {d["seed"]} | {d["teacher_initial"]*100:.2f} → {d["teacher_final"]*100:.2f} | {d["final_weight_mean"]:.6f} | {d["final_edges"]} | {d["text_shift"]:.3f} | {d["image_shift"]:.3f} |\n'
report += '''
以上是 epoch300 的最终诊断，主结果的合成图按最佳 val 选取，可能来自更早 epoch。检查确认 12 次凝缩均未命中缓存，每次都有六次验收。最终结果统一使用独立重训学生的 val，没有混用选择时的最高 val。

固定 seed 不等于 GPU 位级确定性：本轮未启用严格 deterministic algorithms，部分同 seed 学生重训与候选选择时出现小幅差异（GPU 聚合非确定性是可能原因，尚未单独验证）。差异记录在 diagnostics_summary.csv 的 selection_val/retrained_val/retrain_delta_pp 列，不能声称逐位复现。

本轮输入两种模态全部可用，不能验证缺模态修复效果。各 seed 同时控制客户端抽样、划分及训练，均值±标准差包含这些因素，不能解释为固定客户端上的纯训练方差。随机基线诱导图非常稀疏，而凝缩图更密；这是当前实验协议的重要特征，过平滑是否为主因还需固定拓扑/密度消融。

上述现象只能说明实现可运行与该小型协议下的实际表现。teacher-on-syn 与学生真实 val 可能不一致，不能据前者单独判定凝缩成功。后续先固定划分与学生协议，分别测试低阈值图的密度约束、交替开关及 feat_alpha，再在正式客户端划分和独立测试集上确认。
'''
(out/'REPORT.md').write_text(report,encoding='utf-8')
print('PASS: 18 unique runs; 12 cache misses; six validations each; retraining differences recorded; input/source integrity verified')
