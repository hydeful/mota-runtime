# P49对象拓扑学习版本

[总对照入口](../COMPARISON.md) · [完整报告](results/seed49001/REPORT.md) · [适配与认证](ADAPTATION.md) · [冻结协议](EXPERIMENT_PLAN.md) · [首次失败](FIRST_FAILURES.md)

网络直接消费21个对象的节点特征、关系与勇士状态，无二维网格或坐标。运行时决策不回格点寻路。原战斗/事件规则、mean-Q PUCT与采集/SGD/发布流程复用。认证限冻结P49，详见边界说明。

本次真实完成N0→N5及重载N5的4局采集；直接策略188.25→184，MCTS100 324→174。最小闭环已跑通，学习收益尚未得到验证。

已安装PyTorch与NumPy的仓库根目录，可复制单行运行（新目录，禁止覆盖已有结果；这里仅作复现入口，本轮不再执行）：

`OMP_NUM_THREADS=1 python -u -m tasks.tower10.p49.topology.experiment run --root /workspace/p49-topology-reproduction`

状态：`python -m tasks.tower10.p49.topology.experiment status --root /workspace/p49-object-topology-run-49001`

离线恢复全部权重/样本/精确执行源码与核验：`python tasks/tower10/p49/topology/results/seed49001/restore.py /workspace/p49-topology-restored`

默认轻量逐局数据保留评价seed、模型hash、决策/模拟/前向计数和终局。完整逐决策数据、checkpoint、N0—N5权重与账本另列artifacts/ARCHIVE.json。
