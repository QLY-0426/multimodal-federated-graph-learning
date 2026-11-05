# 凝缩图修复与 V0 验证（2026-11-02）

## 范围与结论

按本轮确认，仅修改修改版中的三个生产文件；未访问或修改原始版，未连接服务器、未启动 V1，也未触碰已有实验。新增文件均为本目录内的诊断脚本、日志和评审材料。MultimodalSGC、client.py、server.py 没有修改。

**V0 三档全部通过，现有凝缩回归测试 4/4 通过。尚未证明真实 Movies/Toys 上的凝缩增益。** 本次人工小图所有学生 val 都为 100%，不能用来区分方法优劣。

## 诊断与逐文件修改

| 文件 | 修改与理由 | diff |
|---|---|---|
| SimGC_transductive.py | 新增共享 `_adjacency` 筛选函数，让 consistent_low 和 soft 的训练/验收采用相同拓扑规则和软权重；分类、对齐、平滑损失统一走训练图。新增可选交替更新、固定种子环境内的 teacher-on-syn 诊断、初始参考值、最终双模态特征相对位移、启动档位与缓存命中档位日志。单节点无 PGE 时仍更新特征。 | [逐文件 diff](SimGC_transductive.py.diff) |
| fedgvd_config.py | 显式增加 `adj_mode="legacy"`、`alternating_optim=False`，按后续确认保持默认旧行为；注释给出推荐修复组合。 | [逐文件 diff](fedgvd_config.py.diff) |
| utils.py（仅 cache_signature） | 在复制后的 settings 加 `condense_version=2`，使旧签名失效，不修改调用方 settings；仍沿用原有图、训练/验证划分和标签的哈希逻辑。 | [逐文件 diff](utils.py.diff) |

完整补丁：[changes.diff](changes.diff)。工作目录没有 Git 元数据，diff 的修改前内容由本次首次读取的源码及本次精确修改反向重建；没有借用原始版作比较。

代码检查确认了 P0（训练全连接、验收 0.5 剪枝）、P1（每轮同时 step）和 P2（诊断缺失）。这些实现差异已被处理；它们对真实数据精度损失的因果贡献仍需 V1/消融确认。P3 未在本轮额外改动：feat_alpha 仍为 10，缺模态 PGE 输入方式仍沿用旧逻辑，以便后续单独消融。

## 档位与向后兼容

| adj_mode | 训练 | 验收 |
|---|---|---|
| legacy | 全部候选边及软权重 | 按 config.adj_threshold（现为 0.5）筛选，保留软权重 |
| consistent_low | 按固定 0.01 筛选，保留软权重 | 同训练规则 |
| soft | 全部候选边及软权重 | 同训练规则 |

0.5 筛选没有进入训练路径。consistent_low 的筛选本身是离散操作，保留下来的边权保留梯度；V0 对 `.005/.02/.7` 三条边检查了筛选结果和梯度 `[0,1,1]`。

交替更新由 `alternating_optim=True` 显式启用：零基 epoch `% 50 < 10` 只 step PGE，其余只 step 特征；False 保持同时 step。缺键时 `adj_mode` 回退 legacy、`alternating_optim` 回退 False。没有新增命令行参数。推荐实验在 config 内设置：

```python
"adj_mode": "consistent_low",
"alternating_optim": True,
```

所有新增打印均为 `[condense-diag]` 且 flush=True。`fraction_ge_adj_threshold` 始终表示候选边权 >= config.adj_threshold 的比例，即使 consistent_low 使用的实际保留阈值是 0.01；它不等于该档位的实际保边比例。没有候选边的单节点场景将边权均值和比例记为 0，避免 NaN。

## V0 环境与方法

- Python：`D:\下载学习软件\Python软件\python.exe`，torch `2.12.0.dev20260408+cu128`，torch_geometric `2.8.1`，设备 cuda:0。
- 运行前设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
- 200 节点，文本 64 维、图像 128 维，4 类；20 个节点各缺一种模态，对应特征置零。
- 类原型加噪声特征、随机同类无向边，用于验证可学习情况下的调用和梯度流程，不模拟真实数据难度。
- train/val/test=120/40/40；按训练集同类配额压缩 20%，即 24 个合成节点，全部非自环候选边为 552 条。
- seed=42，hid_dim=32、num_layers=2、dropout=0、lr=.01、weight_decay=0。
- teacher=50、condense=100、student=100、validation_interval=25。每个候选学生使用相同初始化与训练随机种子。
- 三档各使用新建空缓存目录，通过 `data_gc` 完整执行；没有命中旧缓存，无需降级为函数级调用。模式完成后临时 .pt 目录自动清理。
- legacy 采用 alternating=False，consistent_low/soft 采用 alternating=True。该组合比较没有分离两个改动各自的影响。

## V0 结果

| 档位 | 交替 | teacher-on-syn epoch0 → epoch100 | epoch100 weight.mean | epoch100 >=0.5 比例 | epoch100 保留边 | 最佳图保留边 | 最佳学生真实 val |
|---|---|---|---|---|---|---|---|
| legacy | False | 87.50% → 100% | 0.111979 | 0.105072 | 58 | 90 | 100% |
| consistent_low | True | 41.67% → 100% | 0.192642 | 0.217391 | 346 | 552 | 100% |
| soft | True | 41.67% → 100% | 0.193861 | 0.217391 | 552 | 552 | 100% |

| 档位 | 最终文本 ‖final−init‖/‖init‖ | 最终图像 ‖final−init‖/‖init‖ |
|---|---|---|
| legacy | 2.435049 | 3.574658 |
| consistent_low | 2.162015 | 3.472904 |
| soft | 2.142141 | 3.489191 |

每档四次验收都输出了完整诊断，所有验收保留边数 > 0、权重有限、最终特征位移 > 0，teacher-on-syn 均上升。三档首次验收（epoch25）学生 val 已达 100%，严格 `>` 更新最佳图，因此返回的是 epoch25 图；它的边数与 epoch100 不同属预期。特征位移指最终参数，不是返回的最佳图参数；由于模型会归一化特征，原始参数位移也不能单独证明有效方向变化。

epoch0 的 teacher-on-syn 是初始化 PGE 图上的参考值，**不是 random_baseline 诱导子图结果**。本轮没有训练 random_baseline，也没有报告凝缩增益。

完整输出：[v0_full_output.txt](v0_full_output.txt)；机器可读摘要：[summary.json](summary.json)；可复现脚本：[run_v0.py](run_v0.py)。

## 回归测试

使用项目根目录下专用 `_tmp_test` 作为进程 TEMP/TMP，现有测试文件未修改。4 项全部通过（[完整输出](regression_output.txt)）：

1. 合成节点追加时保持真实评估图与掩码、边偏移正确。
2. 训练划分变化使缓存签名失效。
3. 缓存加载、测试标签不影响缓存、删除缓存后确定性重建。
4. 单节点且图像模态完全缺失时可凝缩。

首次测试命令遇到 Python 自带 test 包解析冲突，改为 unittest discover。随后默认沙箱临时路径中的 `{GUID}` 触发缓存路径 `.format()` KeyError 和清理权限错误；按用户确认，仅更换测试进程临时目录重跑，未为此修改生产代码。最终 `_tmp_test` 已清理，不作为交付物。

复现 V0（项目根目录）：

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\下载学习软件\Python软件\python.exe' -u diagnostics/condense_v0_20261102/run_v0.py
```

回归测试命令（需先创建修改版内专用 TEMP/TMP 目录，测试后删除）：

```powershell
& 'D:\下载学习软件\Python软件\python.exe' -m unittest discover -s test -p test_multimodal_condensation.py -v
```

## V1 / V2 后续

按用户本轮收窄范围，Movies/Toys × legacy/修复/random_baseline × seeds 42/43/44 的对照表及均值±标准差暂不生成；没有服务器实测数据，不能声称达到 +1~2 个点提升。

回实验室后再实现并运行同类配额的随机训练节点真实诱导子图基线，保持学生结构、训练轮数、种子一致。每个方法/种子/客户端用独立缓存、root 和日志；通过 config 显式切换模式并核对启动打印。先按 teacher=200/condense=300/student=500/validation_interval=50、10 客户端跑方向性实验，明确客户端 val 的汇总方式，再比较修复增益与 legacy。无提升时再做 adj_mode × alternating_optim × feat_alpha∈{10,1} 消融，单独定位 P3。
