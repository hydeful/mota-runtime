# P49 原CNN不变的地图／对象拓扑序列化受控对照

[结论](REPORT.md) · [完整数字表与曲线](results/three_pairs/REPORT.md) · [冻结协议](CONTROLLED_PLAN.md) · [总入口](../CONTROLLED.md)

全部两阶段三对完成。B可以拟合这批真实经验；A/B在线直接采样三对均有N5收益，MCTS100 A三对提高、B两对提高一对退化。不能外推普遍表示或图网络优势。

原CNN123994参数不变。B的34×5×9仅是1456个拓扑数值加74个零的序列化容器，不是物理地图。

完整证据离线恢复（新目录）：`python tasks/tower10/p49/controlled/results/three_pairs/restore.py /workspace/p49-controlled-restored`。
复现单行（新目录，不覆盖现有结果；本轮不再执行）：`OMP_NUM_THREADS=1 python -u -m tasks.tower10.p49.controlled.experiment run --root /workspace/p49-controlled-reproduction --data /workspace/p49-controlled-restored/data --execution-commit 8073025dc9ce94727baf3d12ca5c8df3bcd30ac8`。
需torch2.10.0+cpu、numpy2.3.5、matplotlib及历史两个run目录的sources.json供源码身份收集（不读历史权重）；可从旧两版证据恢复这些文件。
状态：`python -m tasks.tower10.p49.controlled.experiment status --root /workspace/p49-controlled-run`。

首次失败、配置白名单、B无格点证据及全部原始记录见报告和完整归档。
