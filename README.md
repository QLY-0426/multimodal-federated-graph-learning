FedGVD: Efficient Federated Graph Learning via Unidirectional Distillation with Dynamic Virtual Nodes

## 凝缩图修改说明（2026-11-05）

本分支只改了凝缩阶段的两个文件，其余生产源码与原始版本一致。

| 文件 | 改动 |
|---|---|
| `openfgl/flcore/multimodal_fedgvd/fedgvd_config.py` | `lr_feat` 0.05 → 0.005；`adj_mode` 默认 `legacy` → `real_induce` |
| `openfgl/flcore/multimodal_fedgvd/SimGC_transductive.py` | 新增 `real_induce`：按类配额选出的合成节点直接复用真实子图在这批节点上的诱导边，不再走 PGE 打分；训练、验收、学生评分共用同一条边 |

### 已验证的配置

压缩率 0.2，`feat_alpha = 10`，`lr_feat = 0.005`，教师 200 轮 / 特征优化 300 轮 / 学生 500 轮。

### 本地 10% 单客户端诊断结果

| 数据集 | 指标 | 原始特征 | real_induce + alpha10 + lr0.005 |
|---|---|---:|---:|
| Toys | val acc | 61.29 | **68.30（+7.01）** |
| Toys | macro-F1 | 54.18 | **56.75（+2.57）** |
| Toys | test acc（冻结选型后独立评分） | 61.67 | **66.30（+4.63）** |
| Movies | val acc | 39.54 | 42.23（+2.69） |
| Movies | macro-F1 | 15.81 | 12.66（−3.15） |

Movies 的准确率提升伴随 macro-F1 下降，按预设门槛未晋级。

### 必须明确的边界

1. 以上结果来自**随机抽取 10% 节点作为单个客户端**的快速诊断协议，不是正式 10 客户端联邦划分，绝对分数不能与正式协议比较。
2. `real_induce` 模式下选中节点之间的真实边很少（Toys 实测 7~15 根），该模式接近“只优化特征、几乎不做跨节点传播”，增益主要来自特征对齐和学习率调整。
3. 正式 10 客户端协议下的效果**尚未验证**。
4. 需要恢复原版 PGE 学边行为时，把 `adj_mode` 改回 `legacy`、`lr_feat` 改回 0.05 即可。

### 运行

默认配置就是上表的 `real_induce` 档，按下方 `main.py` 命令直接运行即可，无需额外参数。

## 新需求
目前打算将论文扩展至多模态，参考F:\ZJUT\活\戴哲豪师兄\MM-OpenFGL

选取 Movies 和 Toys 两个文本+图像节点分类数据集，实际数据目录为服务器 `/ssd-data2/hxt2024/FedGVD/dataset/global/subgraph_fl/`。

师兄给出的建议：
把 FedGVD 扩展到多模态子图联邦学习。可以先参考 MM-OpenFGL，选两个文本加图像的节点分类数据集，先把原版 FedGVD 接入多模态特征，跑通基线。
目前的想的创新点是，原方法在服务器端使用全连接来构建全局代理图，在多模态场景下，文本和图像对节点关系的判断可能不一致，全连接可能引入无关连接，影响全局模型和后续蒸馏。当然这个现象需要先通过实验验证，不能预先当作结论。如果问题确实存在，我们考虑在同一组压缩代理节点上，按文本和图像分别构建两个全局代理图，用模态相似性选择连接，并在服务器端分别训练两路图编码器。两路输出还需要根据模态可用性和预测可靠性进行融合，再把融合后的知识蒸馏给客户端。

目前我已经做的是从 https://github.com/mlfoundations/open_clip 开源地址直接拿来的模型 ViT-L-14

## 代码实现约束
1. 减少不必要的代码可靠性判断
2. 多模态Fedgvd的实现文件和内容结构参考原始的Fedgvd
3. 如果model下新增多模态相关的网络结构，那么直接使用multimodal_XXX，例如multimodal_mlp
4. 用户没有明确说明写代码，就不能修改任何文件
5. 测试代码写在根目录下的test下

## Multimodal FedGVD 实现

已实现 `multimodal_fedgvd`，并新增下文说明的 `multimodal_fedtad` 基线。运行时直接加载已有 CLIP 特征，不读取原始图像或重新提取特征。

### 文件结构

- `openfgl/model/multimodal_{gcn,gat,sgc,mlp,graphsage}.py`：客户端双通道异构模型；每个模型的文本、图像分支参数独立，客户端两路使用相同的本地拓扑。
- `openfgl/model/multimodal_utils.py`：公共双分支结构、可靠性融合和分支监督损失。
- `openfgl/flcore/multimodal_fedgvd/SimGC_transductive.py`：局部 MM-SGC 教师训练、联合图压缩、MM-GCN 学生验证和缓存。
- `openfgl/flcore/multimodal_fedgvd/client.py`：首次上传压缩图，之后本地训练及服务器到客户端的蒸馏。
- `openfgl/flcore/multimodal_fedgvd/server.py`：维护客户端压缩图、双图训练、按客户端回传预测。
- `openfgl/flcore/multimodal_fedgvd/contrastive.py`：压缩子图拼接及旧版余弦直连基线。
- `openfgl/flcore/multimodal_fedgvd/virtual.py`：双模态可学习虚拟节点、动态虚拟节点连接及辅助损失。
- `openfgl/flcore/multimodal_fedgvd/fedgvd_config.py`：蒸馏、压缩等算法默认系数。
- `test/test_multimodal_*.py`：模型、图构建、压缩、消息时序、加载及通信统计测试。

### 训练与数据约定

`x_text`、`x_image` 的行表示同一批节点；`modality_mask[:, 0]` 表示文本可用，第二列表示图像可用。没有掩码时认为两种模态都可用，不将全零特征自动解释为缺失。兼容字段 `x` 按 `[归一化图像, 归一化文本]` 拼接，双通道模型直接读取两路特征。

压缩按训练集各类别样本数乘压缩率向上取整，并从真实训练节点初始化成对特征。两路特征联合优化，使用共享 PGE 学习客户端内部压缩邻接，损失包含教师分类、各模态的类别多跳特征矩对齐和加权特征平滑。平滑项使用归一化特征平方距离；与原版代码中的指数形式有区别。压缩节点的标签和模态行不会被分别排序或独立配对。仅使用训练标签优化；验证标签用于教师和压缩图选择，测试标签不进入压缩流程。

服务器默认使用 `--multimodal_server_mode dual_virtual`：在相同的压缩节点上建立文本、图像两张图，每个客户端新增一个共享编号的虚拟节点，分别保存文本/图像特征。每个模态用独立可学习 query 按余弦相似度做 softmax 加权汇聚，再通过独立 MLP（输入维度→hid_dim→输入维度，GELU）生成虚拟节点特征。保留可用模态的子图内部边，添加所属压缩节点与虚拟节点的双向连接；仅在虚拟节点之间选择每行 top-k 邻居，再取对称并集、去重，因此最终度数可能大于 k。新增边权为 1，避免负余弦值引起 GCN 负度数；原压缩边权保留。每个服务器优化步重新计算虚拟特征和 top-k 关系，特征路径保留梯度，离散选边不求导。虚拟节点及其预测只存在于服务器，下发仍是各原压缩节点融合后的 logits。

`--multimodal_server_mode dual_knn` 保留上一版：按文本/图像余弦相似度直接连接不同客户端的压缩节点。`--multimodal_cross_k`（默认 2）在两种模式中分别控制压缩节点/虚拟节点的邻居选择数量；`--multimodal_similarity_threshold` 只作用于 dual_knn。虚拟模式遵循论文 top-k 选择，不额外使用阈值。单客户端或某模态只有一个有效虚拟节点时，该模态没有跨客户端边，对比损失为零。

虚拟模式服务端目标为 `L_syn + virtual_ce_weight * L_virtual_ce + contrastive_weight * L_contrastive`。`L_syn` 保持原压缩节点的融合 CE 加 0.5 倍平均分支 CE；虚拟 CE 使用相同的融合/分支权重。虚拟节点没有单一真实类别，本文实现明确采用所属压缩节点的类别分布作为软目标：融合目标统计至少一个模态可用的节点，各分支只统计该模态可用节点。query 和 MLP 在服务端优化器中更新，客户端新增时保留已有参数及优化器状态。

论文未定义多模态虚拟节点、虚拟节点软标签和全局表示的具体生成方式；以上是扩展/补充定义，不能称作原论文逐式无歧义复现。对比损失按模态计算并平均，使用当前虚拟特征的均值作为全局 anchor；默认 `--multimodal_contrastive_mode paper` 忠实使用式（13）的“其他虚拟节点”分母，通过 logsumexp 稳定计算，值允许为负。可显式切换 `infonce`，将正样本项加入分母；这属于替代目标。默认温度 0.5，两个辅助损失权重均为 1.0，属于待实验验证的初始配置。未添加式（12）中没有明确给出的重构损失。

三组服务端对照参数（其余模型、数据、缓存和训练预算相同）：

```text
A  --multimodal_server_mode dual_knn
B  --multimodal_server_mode dual_virtual --multimodal_virtual_ce_weight 0 --multimodal_contrastive_weight 0
C  --multimodal_server_mode dual_virtual --multimodal_virtual_ce_weight 1 --multimodal_contrastive_weight 1 --multimodal_contrastive_mode paper --multimodal_contrastive_temperature 0.5
```

`test/test_multimodal_virtual.py` 检查虚拟节点数和对齐、双向连接、无压缩节点跨客户端直连、query/MLP 分类梯度、动态选边、缺失模态和单客户端、软标签监督、两种对比损失的解析值、晚加入客户端、融合预测切片及通信隔离。服务器 `diagnostics` 逐轮保存损失分量、query 梯度范数、轮内拓扑变化次数和节点/边数。

两路预测使用归一化预测熵计算权重：可用模态的权重正比于 `exp(-entropy / fusion_temperature)`，缺失模态权重为零。权重停止梯度，融合 logits 为两路 logits 的加权和。两路同时缺失时返回零 logits，监督损失忽略该节点。熵是可靠性的近似指标，并不保证置信度经过校准。

每个客户端首次参与时压缩并上传一次；第二次参与时追加服务器已收集的其他客户端压缩子图（排除自身），训练本地真实节点和其他客户端压缩节点的监督目标；第三次起增加对应压缩节点的上一轮服务器预测 MSE 蒸馏。保持原有损失系数及蒸馏开始时序。没有其他客户端子图时只训练本地真实节点。服务器不聚合客户端模型参数。原始验证/测试图保持独立，只统计真实节点；压缩节点及 logits 都按来源 client_id 升序拼接，保持节点顺序一致。部分客户端参与时，服务器缓存已上传图，在新客户端加入时重建双图，并向参与客户端补发尚未确认接收的其他客户端子图；客户端从原始训练图重建增广图，避免重复追加。常规轮次只下发 logits，通信量统计包含其他客户端 logits 和新增子图。

压缩缓存包含图特征、拓扑、训练/验证划分和压缩参数的指纹，不匹配时重新生成。缓存默认位于算法目录的 `saved_ours/`；测试缓存建议单独指定路径。当前支持 `subgraph_fl`、`node_cls`、`processing=raw`、`dp_mech=no_dp`。

### Multimodal FedTAD 基线

算法目录为 `openfgl/flcore/multimodal_fedtad/`，统一以 [OpenFGL 的 FedTAD](https://github.com/xkLi-Allen/OpenFGL/tree/main/openfgl/flcore/fedtad) 为参照。已逐文件校验该目录与本仓库 `fedtad/` 一致（仅 CRLF/LF 换行差异）。多模态服务器代码与原版仅类名、配置导入不同；保留按节点数聚合参数、条件生成器、语义/多样性/对抗差异损失、CKR 加权蒸馏、梯度与 detach 位置、循环次序和模型参数下发。客户端继承原版收发及本地交叉熵训练。原版 `fedtad/` 源文件不修改。

这是同构基线：默认所有客户端和服务器使用 `multimodal_gcn`，也可用一个 `--model multimodal_gat` 等替换；不支持同时指定多个模型。两路编码器参数独立、共享图拓扑，logits 在可用模态间等权平均，不使用 FedGVD 的熵可靠性融合或双全局图，不额外增加分支监督。

模型及训练默认值与本仓库 OpenFGL 设置对齐：每个分支两层 GCN，隐藏维度 128、Dropout 0.5；显式 `--hid_dim` 优先。`main.py` 对原版 FedTAD 和多模态版本均调用相同的随机种子设置，不再混用作者独立 FedTAD 仓库的隐藏维度 64。

输入统一采用现有数据加载器的 `x=[image, text]`，原版单分支读取整个 `x`，多模态模型按维度切开后送入两路，不在分支中额外归一化。真实 Movies/Toys 的 `x` 已由公共加载器分别归一化两种模态；生成器输出保持原始数值尺度。服务器直接复用原版生成器和建图函数：仅对完整生成向量整体归一化以计算余弦相似度，构建一张 top-k 伪图，保留原版自环约定；模型输入仍是未归一化的生成向量。类别可靠性加权的是不同客户端教师的损失，与模态融合权重不同。

CKR 完整保留原版计算、归一化及缓存流程：使用相同的 `data.x`、拓扑嵌入、训练节点与邻居循环。缓存目录为 `<客户端数据目录>/ckr_multimodal_fedtad/client_<id>.pt`，与原版 `ckr/` 隔离。已有缓存直接读取；没有缓存则计算；`multimodal_fedtad/fedtad_config.py` 中 `save_ckr=True` 时才保存。默认 `False` 与原版一致，但不阻止读取已有缓存。缓存不包含数据指纹（同原版），更换特征或训练划分后需清理对应缓存。CKR 缓存只是类别可靠性向量，不是模型权重或压缩图。

PyCharm 的 **Script parameters** 填入：

```text
--fl_algorithm multimodal_fedtad --dataset Movies
```

Toys 改成 `--fl_algorithm multimodal_fedtad --dataset Toys`。不传算法仍默认运行 `multimodal_fedgvd`；GPU 继续由运行环境的 `CUDA_VISIBLE_DEVICES` 指定。两种算法复用相同的数据划分和默认本地评估方式。若比较算法本身，FedGVD 也应使用 `--model multimodal_gcn` 统一模型条件，避免将异构模型影响混入对比。

默认 100 轮、每轮客户端训练 3 epoch；服务器每轮 5 次生成/蒸馏循环，每次生成器 1 步、全局模型 5 步，生成 100 个节点。算法配置与原版逐项一致，位于 `multimodal_fedtad/fedtad_config.py`。保留原版 `raw_distill`/`rep_distill` 代码分支，默认并验证的是 `raw_distill`；原版 `rep_distill` 将隐藏维度生成特征送入输入编码器，存在输入维度兼容限制，Movies/Toys 当前配置不可直接切换，未另行改造其算法。没有额外增加每类至少一个生成节点或截断 top-k 的行为。

不读取 FedGVD 压缩缓存，也不产生压缩图缓存或最佳模型权重文件。`test/test_multimodal_fedtad.py` 验证模态融合、原始特征尺度、生成梯度、CKR 缓存及数值一致性，并将相同双分支模型接入原版和多模态服务器：相同初值、随机种子下，含 Dropout 的两轮默认模式更新后，服务器与生成器参数逐项完全一致。该检查验证训练流程对齐，不表示单分支与双分支模型的实验精度相同。

### Multimodal FED-PUB 基线

算法目录为 `openfgl/flcore/multimodal_fedpub/`，通过 `--fl_algorithm multimodal_fedpub` 选择。文本和图像各使用独立的两层 MaskedGCN 加掩码分类器，直接复用单模态 `fedpub/maskedgcn.py` 的掩码卷积、分类器及推理剪枝；隐藏维度由 `--hid_dim` 指定，两种模态输入维度可以不同。参数聚合要求所有客户端模型同构，因此只接受单个 `--model multimodal_gcn`（或 `gcn`）和 `--num_layers 2`。

沿用 FedTAD 基线的数据约定，从 `data.x=[image, text]` 切分输入，保留加载器给出的数值尺度；公共代理图的高斯特征不额外归一化。两路 logits 在可用模态间等权平均，隐藏表示按 `[text, image]` 拼接。支持 `modality_mask[:, 0/1]` 指示文本/图像是否可用，缺失分支节点不参与消息传播，两种模态同时缺失时输出零。监督项保持原版的融合 logits 交叉熵，不添加分支监督、蒸馏或可靠性融合。

保留 FED-PUB 的三个机制：① 客户端保留本地可学习掩码，并使用掩码 L1 正则、相对本轮下载权重的 L2 范数正则（第 0 轮不加 L2）；② 服务器固定生成公共随机分块代理图，默认 5 个各含 100 节点的分块、块内概率 0.1、块间无边、标准高斯节点特征，客户端上传代理图上隐藏表示的节点均值；③ 服务器用功能嵌入余弦相似度，经 `exp(10 * similarity)` 行归一化后产生各客户端个性化参数，同时保留按节点数聚合的全局参数，供首次参与客户端使用。算法系数位于 `multimodal_fedpub/fedpub_config.py`，默认复制单模态配置。

多模态版本修正了原本地实现中影响机制生效的细节：正则项使用 `named_parameters()` 保留梯度；下载、上传和正则锚点使用独立快照；个性化权重按客户端 ID 缓存，支持部分参与及重新加入；零功能嵌入的相似度按零处理，无法归一化的行回退为等权聚合。默认指数归一化采用等价的行最大值平移防溢出。单模态 `fedpub/` 不修改。常规非零嵌入下，聚合结果与原版在数值容差内一致。

在服务器项目目录、已有 `fedgvd` 虚拟环境中运行：

```bash
python main.py --fl_algorithm multimodal_fedpub --dataset Movies
python main.py --fl_algorithm multimodal_fedpub --dataset Toys
```

默认 10 个客户端、100 轮、每轮本地 3 epoch，隐藏维度 128、Dropout 0.5；GPU 由 `CUDA_VISIBLE_DEVICES` 和逻辑编号 `--gpuid` 控制。短轮检查可以加 `--num_rounds 3 --num_epochs 1 --hid_dim 16 --client_frac 0.5`。评估使用 `local_model_on_local_data`（默认）或 `local_model_on_global_data`，不支持全局模型评估模式。该基线不读取或产生 FedGVD 压缩缓存。

`test/test_multimodal_fedpub.py` 检查双分支与原始 MaskedGCN 的数值对齐、随机代理图、相似度聚合、正则梯度、掩码保留和剪枝、模态缺失、消息快照及部分客户端参与。

### 服务器运行

默认算法已设为 `multimodal_fedgvd`。在项目根目录、使用服务器虚拟环境直接运行 `python main.py`，即可训练 Movies；默认数据根目录为 `dataset`，使用进程可见的第 0 张 GPU。训练 Toys 使用 `python main.py --dataset Toys`。运行原版可显式指定 `--fl_algorithm fedgvd --dataset Computers`。

GPU 由运行环境控制，代码不覆盖 `CUDA_VISIBLE_DEVICES` 或 `CUDA_DEVICE_ORDER`。在 PyCharm 的环境变量中设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=7`，并保持 `--gpuid` 默认值 0，即使用物理 GPU 7。`--gpuid` 是可见设备列表中的逻辑编号；只暴露一张卡时应使用 0。切换物理 GPU 只需修改环境变量。

在 `/ssd-data2/hxt2024/FedGVD/` 下使用已有虚拟环境，分别运行两个数据集：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 /ssd-data2/hxt2024/envs/fedgvd/bin/python main.py \
  --fl_algorithm multimodal_fedgvd --dataset Movies \
  --root /ssd-data2/hxt2024/FedGVD/dataset --gpuid 0 \
  --debug True --comm_cost True

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 /ssd-data2/hxt2024/envs/fedgvd/bin/python main.py \
  --fl_algorithm multimodal_fedgvd --dataset Toys \
  --root /ssd-data2/hxt2024/FedGVD/dataset --gpuid 0 \
  --debug True --comm_cost True
```

GPU 编号应按服务器当前空闲情况选择。默认 10 个客户端、100 轮，客户端和服务器均使用 `multimodal_gcn`。可重复传入 `--model multimodal_gat --model multimodal_graphsage` 指定客户端模型组合。每次只运行一个数据集。

跨客户端子图修复的基线为 `multimodal-fedgvd` 分支提交 `3274f73`，修复分支为 `fix/multimodal-fedgvd-cross-client`。PyCharm 使用原 SSH 解释器 `/ssd-data2/hxt2024/envs/fedgvd/bin/python`，脚本为项目的 `main.py`，远程工作目录为 `/ssd-data2/hxt2024/FedGVD`。Script parameters 可填写：

```text
--fl_algorithm multimodal_fedgvd --dataset Movies --model multimodal_gcn --num_layers 2 --hid_dim 128 --gpuid 0 --train_val_test 0.6-0.2-0.2 --num_clients 10 --client_frac 1.0 --num_rounds 100 --num_epochs 3 --lr 0.01 --weight_decay 0.0005 --dropout 0.5 --seed 42 --multimodal_reduction_rate 0.2 --multimodal_server_model multimodal_gcn --multimodal_server_epochs 10 --evaluation_mode local_model_on_local_data --debug True --comm_cost True --multimodal_server_mode dual_knn --log_root logs/multimodal_fedgvd_cross_client --log_name Movies_gcn_seed42
```

Environment variables 沿用截图：`PYTHONUNBUFFERED=1;CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=6`，对应参数仍为逻辑设备 `--gpuid 0`。Toys 改为 `--dataset Toys --log_name Toys_gcn_seed42`。不同种子同时修改 `--seed` 与 `--log_name`，避免覆盖日志。正式运行前通过原 PyCharm 映射同步修复分支代码。压缩过程未变，指纹一致的原凝缩缓存可以复用；10 个客户端全参与后，每个客户端训练本地真实图加另外 9 个客户端的凝缩子图，显存需求会高于旧版。

主要参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--multimodal_reduction_rate` | 0.2 | 相对各客户端训练节点的压缩率 |
| `--multimodal_teacher_epochs` | 1000 | 压缩教师训练轮数 |
| `--multimodal_condense_epochs` | 1500 | 压缩优化轮数 |
| `--multimodal_student_epochs` | 3000 | 每次压缩验证的学生训练轮数 |
| `--multimodal_validation_interval` | 100 | 教师评估与压缩图验证间隔 |
| `--multimodal_server_epochs` | 10 | 每轮服务器训练轮数 |
| `--multimodal_cross_k` | 2 | 每节点每模态选择的跨客户端近邻数，对称化后度数可能更高 |
| `--multimodal_similarity_threshold` | 0.0 | 跨客户端余弦相似度下限 |
| `--multimodal_fusion_temperature` | 0.5 | 熵权重温度，必须大于零 |

默认压缩验证开销较大。快速检查可在上述命令后添加以下参数；三轮才能覆盖首次蒸馏：

```bash
--num_rounds 3 --num_epochs 1 --hid_dim 16 \
--multimodal_reduction_rate 0.02 \
--multimodal_teacher_epochs 2 --multimodal_condense_epochs 2 \
--multimodal_student_epochs 2 --multimodal_validation_interval 2 \
--multimodal_server_epochs 2 \
--multimodal_condensed_pattern '/tmp/fedgvd-smoke/{dataset}_{seed}_{client_id}.pt'
```

测试命令（无需 pytest）：

```bash
/ssd-data2/hxt2024/envs/fedgvd/bin/python -B -m unittest discover \
  -s test -p 'test_multimodal_*.py' -v
```

本地维护源代码，服务器运行；同步时仅传输本次修改的源码、测试和本 README，并校验文件内容，不覆盖数据集、实验结果或虚拟环境。

### FedGVD+ 下发错配与原型筛选实验

`multimodal_fedgvd` 分支默认采用完整第三种方案：随机置换＋原型筛选＋标准化KL蒸馏。
实现论文第6–7页式(14)–(17)的机制：服务器对融合后的 logits 逐行以
`beta=(exp(epsilon)-1)/(exp(epsilon)-1+C)` 保留，否则均匀随机置换类别位置（包括恒等置换）。
同一轮同一来源的扰动结果广播给各客户端；服务器内部原始 logits 不变。使用独立的可复现随机流，
不会消耗训练的全局 RNG；这不是密码学随机数实现。

**该功能是论文机制复现，不是经过证明的完整向量差分隐私机制。** 置换保留全部数值和排序后的向量，
因此 `multimodal_mismatch_epsilon` 仅是论文的保留率参数，不报告它为已验证的隐私预算，也不套用多轮 DP 组合公式。
初次上传的凝缩图没有额外隐私处理；仍使用 `--dp_mech no_dp`，不能用此实验声称保护完整原始图。

以下参数已是默认值，也可显式写入运行配置（旧配置中的 `none`、`mse` 等覆盖项需要移除）：

```bash
--multimodal_logit_protection paper_mismatch --multimodal_mismatch_epsilon 1 \
--multimodal_prototype_filter --multimodal_prototype_quantile 0.95 \
--multimodal_distill_loss normalized_kl --multimodal_distill_temperature 1
```

原型由客户端自己的真实训练节点嵌入求各类均值；多模态嵌入采用模型返回的文本/图像隐藏表示拼接。
每次本地参与训练前以 eval 模式计算原型，阈值采用有效训练节点类内余弦距离的95%分位数。
**分位数阈值是明确的实现选择，论文只说根据类内统计动态调整，未给出唯一公式。**
目标类别无有效原型、嵌入为零或距离超过阈值时，拒绝该节点的蒸馏监督。
筛选不影响本地真实节点及其他客户端凝缩节点的分类损失；全部拒绝时蒸馏项为可微的零。
原型、筛选掩码不上传。

`normalized_kl` 对每行 logits 按类别维度标准化（总体标准差下限1e-6），计算
`KL(softmax(teacher/T) || softmax(student/T))` 并在接受的节点上取平均，不额外乘T平方。
客户端蒸馏权重仍为1e-4，服务器虚拟节点分类/对比损失保持原配置。
默认保护模式为 `paper_mismatch`、筛选开启、蒸馏为 `normalized_kl`。
仅保留 `main` 与 `multimodal_fedgvd` 两个分支，历史对照实验仍保存在实验归档中。
为复查历史实验保留显式消融参数：关闭筛选需使用 `--no_multimodal_prototype_filter`；
无扰动KL对照再加 `--multimodal_logit_protection none`；原始MSE对照再加 `--multimodal_distill_loss mse`。
