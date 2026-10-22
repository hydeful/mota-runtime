# 成熟系统：训练组织、硬件与加速实现

这是供独立审查使用的机制地图，不是“照抄即得某倍数”的承诺。链接优先官方论文/官方仓库；源码链接固定提交。外部AI应继续读相关实现，判断在魔塔的小网络、大状态、确定性单人任务中是否成立。

## 对照总表

|体系|学习/搜索体系|硬件证据及口径|值得检查的工程机制|与魔塔的差别|
|---|---|---|---|---|
|AlphaGo Zero|P/V残差网、MCTS、自生成数据、训练、候选模型门控|官方博客的4TPU描述的是下棋配置，不能写成完整训练只用4TPU|统一P/V减少重复评估；采集、训练、评估角色分离|双人围棋；模型、吞吐和机器规模不同|
|AlphaZero 2018|持续更新最新网供采集、P/V+MCTS、独立训练|5000 TPUv1自弈，16 TPUv2训练；比赛另用4TPU|独立大规模actors与learner、训练/推理资源分工|不是单卡100倍证明；2017预印本64TPUv2不能混入2018口径|
|Lc0|C++搜索/后端；Go采集客户端和数据服务；Python训练|志愿者异构设备；具体总训练算力随run变化；比赛2A100是推理配置|NN缓存、并发搜集、batch、GPU后端、网络/数据分发|比赛强度/节点吞吐不是闭环训练吞吐|
|Lc3|Lc0新搜索设计，分阶段流水线|本次没有可归因的硬件/端到端倍率证据|Gather/Eval/Backprop、事件队列、inflight处理、分片节点表|文档中的计划功能不能当全部已生产落地|
|KataGo|C++采集、数据shuffle、训练、导出、可选门控|2019主run最大28V100，平均26–27，19天；分工见下|持续异步阶段、批推理、训练与采集算力配比|论文约50倍含学习效率改动，不是相同工作量纯工程倍率|
|Stockfish/NNUE|CPU alpha-beta+增量神经评估；离线训练|ISA/核数和网络决定效率；本次无统一训练机器配置可归纳|make/unmake、增量hash、稀疏特征、accumulator、量化/SIMD|搜索算法不同，不能因此要求本项目改成alpha-beta|
|Pikafish|中国象棋、Stockfish衍生CPU搜索与NNUE|机器与网络因发布/比赛变化，本包未验证统一配置|accumulator stack、push/pop、局部状态更新|同样不等于AZ自生成训练循环|

## AlphaGo Zero / AlphaZero

[AlphaGo Zero 官方介绍](https://deepmind.google/blog/alphago-zero-starting-from-scratch/)说明单个网络联合策略/价值，去掉快速随机rollout。工程启发是减少重复计算、明确组件分工；不是建议改变当前已选路线。博客的4TPU图不能拿来和本项目整套训练机器直接比。Nature正文在本次抓取遇到访问跳转，完整训练集群硬件未在此重新核实，留作来源缺口。

[AlphaZero 2018 官方论文](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)：训练70万步、batch4096；采集5000TPUv1、训练16TPUv2；国际象棋9小时、将棋12小时、围棋13天。区别于2017预印本的硬件口径。独立采集与learner组织证明其执行系统不必等同本项目的四局同步Python链；大集群数字本身不能预测单机收益。

需要外部AI回答：在20核+4090和小模型下，是否应采用独立常驻actors、跨局批推理、独立learner？如何界定模型滞后和训练数据比例？哪些可以在严格模型屏障下先做，哪些必须列为行为变化？

## Lc0：完整训练体系与推理实现

官方源码/说明快照 `LeelaChessZero/lczero.org@a99d59c497bf6a81151d0d97b06a67d7f6c5722c`：

- [系统组件](https://github.com/LeelaChessZero/lczero.org/blob/a99d59c497bf6a81151d0d97b06a67d7f6c5722c/content/dev/overview.md)：C++引擎，Go客户端获取网络/执行采集/上传数据，Go数据服务收集与分发，Python训练读取数据产出网络。
- [引擎配置](https://github.com/LeelaChessZero/lczero.org/blob/a99d59c497bf6a81151d0d97b06a67d7f6c5722c/content/play/configuration/flags/index.md)：线程、MinibatchSize、NNCacheSize、OutOfOrderEval、MaxConcurrentSearchers 等控制搜索/推理供给；它们涉及吞吐、碰撞、延迟与搜索行为的折衷，不是batch越大越好。
- [官方 CCC 配置记录](https://lczero.org/dev/wiki/cccc/)：CCC17 Blitz使用2×A100、2×EPYC7H12（128物理核/256线程）、cuda-fp16 demux、batch384、NNCacheSize20000000。这是比赛推理实例，不是训练服务器清单，也不能当当前默认参数。

后端与搜索分离，支持不同GPU/CPU执行方案；不要将旧文档里的训练计划当已完成系统。魔塔重点可比的是“如何让独立搜索工作产生足够批量、哪些数据要传、缓存绑定什么身份、何时回传”，而非照抄batch384。

## Lc3：重构搜索执行组织

[固定版本设计文档](https://github.com/LeelaChessZero/lczero.org/blob/a99d59c497bf6a81151d0d97b06a67d7f6c5722c/content/dev/lc0/search/lc3/overview.md)。以Gather/Eval/Backprop阶段和NodeEvent队列组织搜索；节点表分片，记录inflight工作；缓存/终局可走快速路径，碰撞需要撤销/修正。它提示加速可能来自改变工作流与数据结构，不只是换forward设备。

本次未取得可对齐魔塔工作量的Lc3端到端基准，不能给它分配“已验证N倍”。哪些设计已实现、哪些仍planned，以固定版本正文为准。

## KataGo：采集—数据—训练五阶段

[固定版本 SelfplayTraining](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/SelfplayTraining.md)：C++selfplay持续写数据，Python shuffle生成训练块，Python train保存模型，export转C++推理格式，gatekeeper可选。正常多机训练五者同时运行；单机同步脚本只是便利起步，文档明确其默认值并非主run所用成熟配置。建议采集GPU算力为训练4～40倍，是经验配置指导，不是魔塔应该硬设的比例。

[训练配置示例](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/configs/training/README.md)覆盖1/2/8GPU、不同网络/访问预算。应按网络规模与批量供给配资源，不能把8GPU配置直接放一张4090。[nneval.cpp](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/neuralnet/nneval.cpp)可继续核查常驻评估服务、队列、buffer、batch统计。

[2019研究，v5论文](https://arxiv.org/pdf/1902.10565)报告19天，前两天16张V100采集、后来24张；2张门控、1张训练，额外1张并行训练下一尺寸。最大28张、平均26–27；约420万局/2.41亿样本。其约50倍计算节省包含playout cap randomization、策略目标修剪、网络与辅助目标等，属于学习效率和工程混合贡献，不能归入同预算工程提速。

魔塔审查可借鉴常驻异步组件、数据shuffle与训练分离、推理模型导出和资源配比。若借鉴搜索预算随机化/目标修剪，须列为学习协议变化，不能悄悄加入严格工程基准。

## Stockfish / NNUE：高频小步更新的数据布局

[position.cpp](https://github.com/official-stockfish/Stockfish/blob/effdc3fd3f6de6e75fe1374a51a7a7eed66f880b/src/position.cpp)展示make/unmake与StateInfo、增量身份维护。[NNUE工程说明](https://github.com/official-stockfish/nnue-pytorch/blob/ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8/docs/nnue.md)解释稀疏变化、累加器更新与必要刷新、量化计算；[训练仓库](https://github.com/official-stockfish/nnue-pytorch/blob/ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8/README.md)与运行引擎分开。

魔塔每交互往往只改变少数资源和对象，可检验增量状态、特征、hash或编译布局是否减少反复遍历全字典/全张量。但是“原CNN做精确增量计算”和“改成NNUE新模型”是两件事；后者改变模型函数，需独立验证能力。维护撤销栈也必须处理门、道具、任务标记、自动结算等多处修改，不能只逆转HP。

## Pikafish：相似工程机制在另一种棋上的适配

[nnue_accumulator.h](https://github.com/official-pikafish/Pikafish/blob/545c4e3b251afbb94f493c6c5f32549f900e76fd/src/nnue/nnue_accumulator.h)定义AccumulatorState/Stack及push/pop。它说明应围绕状态变化组织评估，而非每节点重新物化一份完整网络输入；但本文件不提供端到端训练收益或固定机器对照，不能编造倍数。

外部AI应比较“不可变静态图+紧凑动态状态/事务差分”和现有prepared/COW的真实差别，不能只给已有机制换名。

## 从案例到实验，而不是从案例到结论

请为每项迁移机制回答：消除了什么工作？新增什么队列/同步/复制/缓存成本？需要多大并发才饱和？是否改变搜索/样本顺序？什么小实验可以证伪？没有相同任务和硬件的证据，不承诺倍数。

这些是参考起点，不是封闭候选清单。整体工程体系可以重建；保留AlphaZero式学习原则，并不要求保留当前Python事务、序列化和进程结构。

