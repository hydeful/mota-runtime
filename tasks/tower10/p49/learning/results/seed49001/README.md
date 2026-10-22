# seed49001：结果与证据

正式名称：P49地图输入网络＋对象交互搜索的5轮学习实验。

- [REPORT.md](REPORT.md)：三项独立结论、完整六轮结果、耗时和拓扑架构边界。
- [evaluation_curves.svg](evaluation_curves.svg)：两条件均值曲线及逐局范围。
- [evaluation_summary.csv](evaluation_summary.csv)：12组统计，含结局数、模拟数、真实前向数和耗时。
- [evaluation_games.jsonl](evaluation_games.jsonl)：192局逐局原始元数据，无筛选。
- [training/](training/)：五轮完整训练账本、模型父链、首次正式SGD检查。
- [interface.json](interface.json)、[postrun_validation.json](postrun_validation.json)：接口核对及轻量账本复核。
- [sources.json](sources.json)、[ARTIFACTS.json](ARTIFACTS.json)：实际运行源码及全部原始文件身份。

完整证据含N0–N5权重、checkpoint/优化器/RNG、28局实际采集、192局全部决策记录、完整正式日志及实际执行源码。
以可校验的Base64分片保存于本私有分支的artifacts/，没有自动公开。分片只是二进制交付方式，不是训练数据变换。

在仓库根目录运行一条命令恢复并校验完整证据（目标目录须不存在）：

```bash
python tasks/tower10/p49/learning/results/seed49001/restore.py /workspace/p49-restored-evidence
```

完整ZIP约5.14MB；默认摘要ZIP约0.812MB，权重不放入摘要。ARCHIVE.json列出分片、归档和摘要SHA。
完整归档已实际解包并逐个核对109个清单条目；旧版归档清单也保留，用于记录首次progress身份失配。

本次正式运行源码固定提交：9fe7b7fb28a08d69ed6f8e100e4ff3a8d35a8b25；实际字节亦在完整包executed_source/。
最终交付代码只追加归档末尾身份刷新修复和报告措辞/拓扑架构边界；没有重训、重评或改变冻结协议。

复现新实验：`python -u -m tasks.tower10.p49.learning.experiment run --root /workspace/p49-new-run`。
查看状态：`python -m tasks.tower10.p49.learning.experiment status --root /workspace/p49-learning-run-49001`。
run拒绝覆盖已有目录，不自动续跑或重置预算。
