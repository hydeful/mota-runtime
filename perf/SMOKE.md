> 本轮为验收分类修复。已有smoke2无需重新运行；只读重分类入口见下文。保留下方DDC首次诊断命令作为历史操作说明，本轮不要据此重启实验。

# P48：600秒流程诊断

沿用状态比较器、独立`mota-smoke`入口及原P45；不改冻结算法文件。本模式用于排错，**不是600秒性能对照，不证明提速，不自动进入一小时实验**。安装另计；从start开始，子进程导入、预热、失败、检查和打包均共用不可重置的最多600秒绝对预算，可提前结束。关闭终端继续，不自动重跑或追加预算。

首次安装与后台启动的固定提交命令见本页末尾的发布命令。新版使用新名称`smoke2`，只写入`/root/autodl-tmp/p48-perf/smoke-tests/smoke2/`，不读取或修改`cloud600`、`probe1`或其预算。

## 实际阶段及依赖

| 阶段 | 上限秒 | 正常输入/依赖 | 上游失败时 |
|---|---:|---|---|
| reserve_inputs | 30 | 独立、新鲜随机网络及一局实际搜索 | 失败照实记录；其他独立路径继续 |
| fixture | 50 | 独立、原版四局及7次模拟的未完成树 | 不伪造成功 |
| serialization | 30 | fixture完整断点、树、RNG及样本 | 已验证备用完整输入 |
| micro | 25 | fixture模型、样本、状态 | 备用输入；故意packed检查失败不阻断CPU计算 |
| cpu_parallel | 45 | 独立、两个CPU采集进程/一个学习器 | 不依赖fixture |
| service | 20 | 独立CPU批量服务、初始编码 | 不依赖fixture/micro |
| gpu_micro | 40 | 同模型/样本及匹配CPU金标准 | 整组替换成匹配备用输入+参考值，不混用 |
| gpu_service | 25 | 独立CUDA前向、真实批量内核及SGD | 不依赖CPU金标准；无可用环境明确报告 |
| learning | 80 | 有效样本账本和断点 | 备用有效测试账本 |
| orphan_parent | 6 | 受控异常父退出，子进程另建会话 | 独立验证任务归属清理 |
| shutdown_timeout | 12 | 真实服务故意不回复shutdown | 保留原始错误，强制回收 |
| worker_failure | 15 | 正式并行job故意缺失snapshot、真实CPU服务 | 验证原始异常/最终保存/服务关闭 |
| controlled_failure | 8 | 受控单项异常 | 后续真实路径继续 |
| controlled_timeout | 5 | 留下已进入超时工作的标记，再等待 | 到点清理，后续真实路径继续 |
| fallback_serialization | 20 | 故意令该项看不到主上游，使用备用材料 | 自身和失败衔接分别记录 |
| real_candidate_worker | 25 | 实际micro验证过的候选配置 | 备用micro；都无效时只尝试冻结参考配置，候选衔接未验证 |
| scheduling_logic | 余量 | 正式选择/调度函数，print占位子进程 | **仅调度逻辑检查** |

主体上限共436秒；后项保留启动机会，最后35秒用于收尾和小包。临近截止缩短单项上限或记录未执行，不拼接中断结果。正常情况下优先使用实际上游输出，所有材料先校验清单SHA；无有效主/备输入时明确列出缺失条件，继续独立路径。

## 备用材料及衔接结论

备用材料在主fixture之前一次性新建：新鲜随机网络、真实原版一局终局样本，参考规则重放，未完成100模拟决策的树和两种actor RNG，实际模型/优化器/全局RNG。一个终局被放入明确标记的**诊断单局账本**，不是正式四局提交。所有必需文件封存SHA清单；CPU金标准绑定同一输入basis_id和参考文件SHA。备用材料不是历史权重或数据，不使用损坏断点训练。

`status`表示综合验收；`acceptance.functional_status`保留阶段功能结果，`unexpected_live_process_status`、`reaping_status`、`expected_fault_status`分别报告非预期活进程、回收和受控故障；`link_status`表示直接上下游衔接；`chain_status`继承整个实际依赖链的失败/未验证。备用输入自身通过，原衔接仍失败或未验证，后面阶段通过不会洗掉它。设备不可用时不声称其消费路径已验证。文件完整性在控制器及消费入口都校验，包内记录输入来源与basis_id。

为了覆盖训练发布路径，测试仍明确注入`new_positions=512`，不改正式512门槛、网络、100模拟、批量64、优化器/学习率或曝光公式。构造阶段执行63次真实SGD（仅跳过构造过程的中间写盘），两种保存各恢复后完成最后一次真实SGD及N1后续搜索。实际位置和合成计数分开，**不计正式产能或学习收益**。

新比较器逐字段、精确比较模型/优化器/进度、树/子节点、数组dtype/shape/data/writeable、Python/NumPy/Torch随机状态及Generator/BitGenerator内部状态。未知类型明确失败；不推进随机状态、不修改对象、不依赖地址repr。人工只读数组负例独立记录原版Torch的属性损失，不静默忽略。

## 实际worker与调度逻辑分开

`real_candidate_worker`把实际通过micro门的`direct_hash`配置写入options.json，经正式`perf.worker`命令行入口执行至少一个真实100模拟决策，规则审计后解析result.json，输出`real-worker-summary.json`。没有合格优化门时只测试原版参考入口。它是参数/路径/入口/结果解析的实际衔接，**不是完整学习闭环通过**，不会把零训练位置包装为产能。

`scheduling_logic`保留原有正式函数+print子进程测试，仅写逻辑检查。两者在覆盖表和结果包中分别列项。

## 异常进程清理

每项获得随机任务归属标记。控制器在直接子进程退出后继续扫描和清理本项后代，包括继承标记但另建会话/已被重挂的进程；记录PID+创建时间、清理前/后状态及可观测范围。Linux子收割器可用时回收本项被收养的后代。绝不按进程名称批量结束进程，也不杀其他任务。不可读取的进程数量和扫描限制保留；AutoDL实际完整树仍待云端证据。

Service创建失败也会回收已创建的服务。close的shutdown超时/异常仍关闭连接、join/terminate/kill；清理错误单独保留，不覆盖原始worker/调用错误，不阻止最终断点/证据保存。包内保留异常路径和退出码。停止诊断可能中断当前case，明确标记未完成；真正安全保存/恢复另由学习case的真实Runner STOP验证，不假称强制中断已保存。

## 操作与结果

```bash
/root/autodl-tmp/p48-perf/bin/mota-smoke status smoke2
/root/autodl-tmp/p48-perf/bin/mota-smoke stop smoke2
/root/autodl-tmp/p48-perf/bin/mota-smoke resume smoke2
/root/autodl-tmp/p48-perf/bin/mota-smoke export smoke2
```

已记录项不自动重跑，resume保留原截止。完成后拒绝自动重复恢复。状态分别显示固定的实际执行耗时、结束时间和随当前时间变化的剩余绝对预算；再次查看或导出不增长执行耗时。

自动小诊断包：`/root/autodl-tmp/p48-perf/smoke-tests/smoke2/results.zip`，上限5MB，默认只收覆盖表、环境/配置、每项耗时/输入来源/退出码/清理及错误路径、8KB日志和资源记录尾部、真实worker摘要及预算/导出检查。不包含终局样本、权重或重复工件。

`export-verification.json`确实在包内：先校验payload CRC和清单SHA，再加入验证记录，完成后再校验整个ZIP的CRC。清楚区分验证范围；最终小标记和ZIP封口耗时另记在包旁export.json，实际执行结束计时保持固定。手动导出时间与执行时间分开。

完整资料保留在同一目录，必要时显式另导出（可能包含测试断点和权重；默认不需要）：

```bash
/root/autodl-tmp/p48-perf/bin/mota-smoke export smoke2 --full
```

输出`full-results.zip`，不替换小包。不要默认上传全量资料。

安装继续功能核查后复用CPU/CUDA环境，代码更新不触发整套CUDA重装。首次fetch失败留下的空.git可重试；核对完整提交对象和工作区，有用户文件/修改则拒绝覆盖。必须安装CUDA时显示实时输出及持久日志位置，实验计时尚未启动。

本地CPU真实计算、受控输入、逻辑模拟和设备不可用分别记录。没有AutoDL/CUDA实证，不能宣布云端诊断通过；也不能据此宣布性能改善或自动安排一小时实验。

## 本次固定发布命令

固定实验代码：`ddc29d4d12a4eccf711a012a2017f75d0eca7996`。复制这一条，安装成功后后台启动一次新的`smoke2`：

```bash
curl -fsSL https://raw.githubusercontent.com/hydeful/mota-runtime/ddc29d4d12a4eccf711a012a2017f75d0eca7996/perf/install.sh -o /root/autodl-tmp/p48-smoke2-install.sh && bash /root/autodl-tmp/p48-smoke2-install.sh ddc29d4d12a4eccf711a012a2017f75d0eca7996 auto && /root/autodl-tmp/p48-perf/bin/mota-smoke start smoke2
```

本发布已做匿名固定源码下载、文件核对、CPU环境复用安装、自检和本地实际CPU路径诊断。GPU和AutoDL实际诊断仍待用户结果包；安装成功、云端诊断通过和性能改善是三个不同结论。


## 僵尸退出记录的验收分类修复

`cleanup.before_cleanup`非空本身不是失败。case.json功能结果与进程验收分开：已退出的zombie/dead记录在成功回收、remaining为空、无清理错误且观察范围可确认时通过；发现仍运行的非预期后代，即使随后被清理也保留泄漏问题；剩余退出记录、回收错误或无法确认的情况保留失败/unknown。不会为了消除FAIL而删除清理记录。

受控孤儿故障只能豁免该case明确记录的child_pid，不能豁免额外活进程。受控shutdown超时/worker异常单独验证，原始功能结果和故障仍保留。coverage.status是综合验收，不再标为“自身功能”；四项明细在acceptance字段和CSV中。

本修复不改变任何信号、等待、回收、服务关闭、超时、训练或预算行为。不需要因此重跑云端进程测试；已有smoke2证据可以离线重分类。完整原文件未提供时，不从摘要推断未报告的errors/观察范围/其余验收项已通过。

新版本入口可执行`mota-smoke reclassify smoke2`，只读取原state/cases/coverage，生成独立`acceptance-reclassified-*.json`侧表。原case、cleanup、coverage、results.zip、结束时间及旧预算均不改。也可在独立源码目录用系统Python（不需要Torch/CUDA或安装依赖）运行：

```bash
python3 -B -m perf.reclassify --root /root/autodl-tmp/p48-perf/smoke-tests/smoke2 --output /root/autodl-tmp/p48-perf/smoke-tests/smoke2/acceptance-reclassified.json
```

输出已存在则拒绝覆盖。该命令不调用worker、GPU、搜索、训练或任何实验控制程序。它只复核任务分类，不会补造缺失的功能证据或全链路通过，不授权一小时实验。

分类自检（零训练）：`python3 -B -m perf.acceptance_test`。

分类补丁固定代码：`d8da913bc466db6276ef3222abf0e2bd063820d6`。可在独立源码目录检出此提交，用系统Python运行上述`perf.reclassify`；不需要更新原run绑定版本、重装CUDA或重跑诊断。此补丁不启动任何云端实验。
