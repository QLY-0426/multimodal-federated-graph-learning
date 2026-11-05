# 本地小型 V1 增益对比（2026-11-02）

状态：全部 18 个 run 已完成。

**结论：本轮修复未达到超过随机基线的目标。** Movies 修复档相对随机基线平均低 6.1350 个百分点，Toys 低 39.1198 个百分点，六个 dataset×seed 全部为负增益。相对 legacy，Movies 均值提升 2.2495 个百分点，Toys 下降 9.1280 个百分点，缺乏稳定改善。

## 协议

- 数据只读来源：`D:\下载学习软件\韩旭涛_科研项目资料\跑FedMVP实验\10_29跑实验\FedGVD-multimodal_fedgvd\dataset\global\subgraph_fl`。直接加载 raw/*.npy，绕过 FL 划分/pymetis；未实例化会写 processed 的 Dataset。
- 每数据集每 seed 按类比例随机取 ceil(10%) 节点作为 1 个客户端，保留真实有向诱导边，不额外补边/无向化。随后逐类按 60%/20%/余量划分 train/val/test。三个方法共享完全相同的数据与划分，划分保存为 split_*.pt。
- 使用原始文本/图像 CLIP 特征，零向量视为缺失模态，没有人为注入缺失。数据统计见下表。
- 设备 `cuda:0`（NVIDIA GeForce RTX 4060 Laptop GPU），torch=2.12.0.dev20260408+cu128，Python=3.14.3。
- teacher=200 / condense=300 / student=500 / validation_interval=50。GPU 按原定轮数；CPU 时全部减半。
- 沿用 V0 学生超参数：hid_dim=32，num_layers=2，dropout=0，lr=.01，weight_decay=0，fusion_temperature=.5；reduction_rate=.2；其他凝缩 config 保持原值（feat_alpha=10）。
- legacy: adj_mode=legacy, alternating_optim=False；fixed: consistent_low + True。只在脚本进程中修改 config，不写回源码。
- random_baseline 按 generate_labels_syn 的 ceil 类配额，从真实训练节点随机抽取并保留真实诱导边/特征/模态掩码；不额外训练教师。
- 三者最终学生均调用同一个 _validate_synthetic，训练固定轮数，固定 seed 初始化，在同一真实客户端 val 上评分。凝缩图先由 data_gc 按六次候选的 val 选最佳，再重训同 seed 学生评分；随机基线仅抽一次。
- 因 teacher/合成图选择使用了 val，本表是开发验证集结果，存在选择偏差，不是独立测试集泛化证据；不使用 test acc 选模型。单客户端随机抽样会使边稀疏，不能代替正式十客户端划分。
- 缓存目录 cache/20261102_224147/，每 dataset×method×seed 独立文件；condense_version=2。日志为 logs/*.log。
- 启动前 nvidia-smi 已确认 RTX 4060；TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1；无安装新依赖。

## 客户端统计

| dataset | seed | nodes | edges | train/val/test | syn nodes | missing modalities |
|---|---:|---:|---:|---|---:|---:|
| Movies | 42 | 1675 | 1034 | [998, 326, 351] | 209 | 0 |
| Movies | 43 | 1675 | 980 | [998, 326, 351] | 209 | 0 |
| Movies | 44 | 1675 | 1111 | [998, 326, 351] | 209 | 0 |
| Toys | 42 | 2077 | 624 | [1241, 409, 427] | 254 | 0 |
| Toys | 43 | 2077 | 585 | [1241, 409, 427] | 254 | 0 |
| Toys | 44 | 2077 | 694 | [1241, 409, 427] | 254 | 0 |

## 逐次结果

| dataset | method | seed | real val acc (%) | syn edges |
|---|---|---:|---:|---:|
| Movies | legacy | 42 | 37.7301 | 8866 |
| Movies | fixed | 42 | 33.7423 | 17686 |
| Movies | random_baseline | 42 | 38.0368 | 12 |
| Movies | legacy | 43 | 16.8712 | 6676 |
| Movies | fixed | 43 | 34.0491 | 29758 |
| Movies | random_baseline | 43 | 38.9571 | 8 |
| Movies | legacy | 44 | 38.3436 | 840 |
| Movies | fixed | 44 | 31.9018 | 23680 |
| Movies | random_baseline | 44 | 41.1043 | 10 |
| Toys | legacy | 42 | 23.2274 | 15080 |
| Toys | fixed | 42 | 24.2054 | 21730 |
| Toys | random_baseline | 42 | 59.1687 | 11 |
| Toys | legacy | 43 | 20.2934 | 16728 |
| Toys | fixed | 43 | 15.4034 | 20108 |
| Toys | random_baseline | 43 | 60.3912 | 15 |
| Toys | legacy | 44 | 45.7213 | 0 |
| Toys | fixed | 44 | 22.2494 | 21884 |
| Toys | random_baseline | 44 | 59.6577 | 7 |

## 均值 ± 样本标准差（ddof=1）

| dataset | method | n | val acc (%) |
|---|---|---:|---|
| Movies | legacy | 3 | 30.9816 ± 12.2238 |
| Movies | fixed | 3 | 33.2311 ± 1.1613 |
| Movies | random_baseline | 3 | 39.3661 ± 1.5741 |
| Toys | legacy | 3 | 29.7474 ± 13.9114 |
| Toys | fixed | 3 | 20.6194 ± 4.6218 |
| Toys | random_baseline | 3 | 59.7392 ± 0.6153 |

## 凝缩增益（同 seed 配对差，单位百分点）

| dataset | seed | fixed − random | legacy − random | fixed − legacy |
|---|---|---:|---:|---:|
| Movies | 42 | -4.2945 | -0.3067 | -3.9877 |
| Movies | 43 | -4.9080 | -22.0859 | +17.1779 |
| Movies | 44 | -9.2025 | -2.7607 | -6.4417 |
| Movies | mean ± sd | -6.1350 ± 2.6742 | -8.3845 ± 11.9291 | +2.2495 ± 12.9865 |

Movies：修复档相对随机基线平均 -6.1350 个百分点，相对 legacy +2.2495 个百分点；本次结果未同时证明超过随机基线与 legacy，不能宣称修复有效。

| dataset | seed | fixed − random | legacy − random | fixed − legacy |
|---|---|---:|---:|---:|
| Toys | 42 | -34.9633 | -35.9413 | +0.9780 |
| Toys | 43 | -44.9878 | -40.0978 | -4.8900 |
| Toys | 44 | -37.4083 | -13.9364 | -23.4719 |
| Toys | mean ± sd | -39.1198 ± 5.2268 | -29.9919 ± 14.0589 | -9.1280 ± 12.7640 |

Toys：修复档相对随机基线平均 -39.1198 个百分点，相对 legacy -9.1280 个百分点；本次结果未同时证明超过随机基线与 legacy，不能宣称修复有效。


## 结论边界与下一步

三种子不足以得出稳健统计显著性结论，未做真实十客户端联邦聚合。若修复无稳定提升，下一步应固定划分消融 adj_mode × alternating_optim × feat_alpha∈{10,1}，并使用独立 test 评估及正式客户端划分。此轮按要求不改源码或自动追加消融。
输入数据文件和三个生产源码文件运行前后 SHA256 一致性见 integrity.json；全部执行日志、缓存、随机抽样节点 ID 与分割已保留。

## 凝缩诊断汇总

| dataset | method | seed | teacher syn 初始→最终 (%) | 最终 weight.mean | 最终边数 | 文本位移 | 图像位移 |
|---|---|---:|---|---:|---:|---:|---:|
| Movies | legacy | 42 | 38.28 → 98.09 | 0.089067 | 3900 | 11.968 | 14.583 |
| Movies | fixed | 42 | 31.58 → 76.08 | 0.377126 | 19750 | 9.656 | 11.514 |
| Movies | legacy | 43 | 31.58 → 56.46 | 0.029760 | 0 | 9.843 | 12.779 |
| Movies | fixed | 43 | 31.58 → 42.58 | 0.565861 | 25254 | 11.513 | 10.038 |
| Movies | legacy | 44 | 46.41 → 96.65 | 0.016912 | 840 | 14.243 | 17.143 |
| Movies | fixed | 44 | 31.58 → 86.60 | 0.203344 | 18236 | 11.151 | 12.807 |
| Toys | legacy | 42 | 31.89 → 69.69 | 0.234743 | 15080 | 13.522 | 11.856 |
| Toys | fixed | 42 | 10.24 → 59.45 | 0.298355 | 21854 | 10.852 | 10.444 |
| Toys | legacy | 43 | 10.24 → 69.69 | 0.260478 | 16728 | 12.231 | 11.931 |
| Toys | fixed | 43 | 10.24 → 56.30 | 0.289196 | 20108 | 10.768 | 10.849 |
| Toys | legacy | 44 | 10.63 → 100.00 | 0.000013 | 0 | 11.595 | 12.931 |
| Toys | fixed | 44 | 10.24 → 55.91 | 0.315182 | 21884 | 9.651 | 11.290 |

以上是 epoch300 的最终诊断，主结果的合成图按最佳 val 选取，可能来自更早 epoch。检查确认 12 次凝缩均未命中缓存，每次都有六次验收。最终结果统一使用独立重训学生的 val，没有混用选择时的最高 val。

Movies seed43 的 legacy 在最后一轮无非自环边，但返回的最佳候选图有边。Toys seed44 的 legacy 则连返回的最佳图也无非自环边（模型仍有自环），最终 teacher-on-syn=100%、学生 val=45.72%。因此不能把高 teacher-on-syn 或非零特征位移直接等同于健康图结构或有效凝缩。修复档返回图仍有较多边，但此次并未因此获得精度优势。

固定 seed 不等于 GPU 位级确定性：本轮未启用严格 deterministic algorithms，部分同 seed 学生重训与候选选择时出现小幅差异（GPU 聚合非确定性是可能原因，尚未单独验证）。差异记录在 diagnostics_summary.csv 的 selection_val/retrained_val/retrain_delta_pp 列，不能声称逐位复现。

本轮输入两种模态全部可用，不能验证缺模态修复效果。各 seed 同时控制客户端抽样、划分及训练，均值±标准差包含这些因素，不能解释为固定客户端上的纯训练方差。随机基线诱导图非常稀疏，而凝缩图更密；这是当前实验协议的重要特征，过平滑是否为主因还需固定拓扑/密度消融。

上述现象只能说明实现可运行与该小型协议下的实际表现。teacher-on-syn 与学生真实 val 可能不一致，不能据前者单独判定凝缩成功。后续先固定划分与学生协议，分别测试低阈值图的密度约束、交替开关及 feat_alpha，再在正式客户端划分和独立测试集上确认。
