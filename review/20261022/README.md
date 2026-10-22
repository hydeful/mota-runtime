# 外部工程审查入口

日期：2026-10-22。任务：魔塔 AI 端到端约 100 倍工程加速。AlphaZero 式“网络引导搜索→真实交互采样→经验回放→SGD→新模型用于后续采样”是当前选定路线。

## 这份资料希望得到什么

请站在独立工程架构师视角，评估整个训练执行体系应如何组织，而不是只修补现有 Python 函数。允许替换运行时、状态布局、搜索执行器、数据平面、进程模型、推理后端、训练调度和持久化体系。目标是把真实闭环的单位墙钟产出提高两个数量级，并给出可验证的分阶段路径。

100 倍是设计目标，尚未实现。现有约 2.1 倍收益已实测，但不是性能上限。也不能从棋类论文的倍数直接推导本项目的倍数。

## 推荐阅读顺序

1. [CONTEXT.md](CONTEXT.md)：游戏、目标、学习规则与代码调用关系。
2. [BASELINE.md](BASELINE.md)：机器、工作量、原始计时和测量限制。
3. [EXPERIMENTS.md](EXPERIMENTS.md)：已做尝试、收益、负结果及尚未执行项。
4. [REFERENCE_SYSTEMS.md](REFERENCE_SYSTEMS.md)：AlphaGo Zero、AlphaZero、Lc0/Lc3、KataGo、Stockfish/NNUE、Pikafish 的硬件、工程与训练组织。
5. [REVIEW_CONTRACT.md](REVIEW_CONTRACT.md)：100 倍的分母、正确性要求和期望交付。
6. [REPRODUCE.md](REPRODUCE.md)：公开代码和证据如何核验、哪些内容缺失。
7. 阅读实际源代码和原始 evidence，再形成独立意见。

## 最重要的证据入口

|问题|入口|
|P48 完整复盘与原始计时|[REVIEW](evidence/p48/REVIEW.md)、[state.json](evidence/p48/state.json)、[MEASUREMENTS](evidence/p48/MEASUREMENTS.json)|
|P49 两个历史版本|[COMPARISON](../../tasks/tower10/p49/COMPARISON.md)|
|P49 受控三对 CNN 对照|[CONTROLLED](../../tasks/tower10/p49/controlled/REPORT.md)、[完整原始评价](../../tasks/tower10/p49/controlled/results/three_pairs/evaluation_games.jsonl)|
|实际执行代码|[P45 Runner](../../tasks/tower10/p45/runtime.py)、[PUCT](../../tasks/tower10/p45/search.py)、[P48 perf](../../perf/README.md)、[P49](../../tasks/tower10/p49)|

P49 历史比较正确链接：[COMPARISON.md](../../tasks/tower10/p49/COMPARISON.md)。历史文件中指向私有仓库或未公开大档案的链接不能作为外部 AI 的必读依赖；本目录说明取代这些访问前提，历史原文保留供辨认。

## 事实等级

- **已验证记录**：固定提交的代码、归档的原始数字、已经执行的核验。
- **解释/判断**：对记录的解释，可以被外部审查推翻。
- **方案**：待测的工程路径，不能写成已有收益。
- **缺失**：原归档或本公开包没有的证据，明确列出，不用估计补成事实。

本包对旧团队的优先级建议不背书为外部任务限制。尤其是 P48 REVIEW 中 P0/P1/P2 排序属于历史建议，应由审查者重新判断。

