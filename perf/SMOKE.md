# P48 600秒流程冒烟

本模式检查执行路径，不提供加速、正式产能或学习增强结论。旧CPU入口、`cloud600`、`experiments/probe1`和旧预算均不修改、不读取。所有输出在独立的`smoke-tests/名称/`。

固定实验代码：`7daf30117f3c8228b409a42971d67ba684a4af69`。先复制这一条，安装成功后仅启动一次独立冒烟：

```bash
curl -fsSL https://raw.githubusercontent.com/hydeful/mota-runtime/7daf30117f3c8228b409a42971d67ba684a4af69/perf/install.sh -o /root/autodl-tmp/p48-smoke-install.sh && bash /root/autodl-tmp/p48-smoke-install.sh 7daf30117f3c8228b409a42971d67ba684a4af69 auto && /root/autodl-tmp/p48-perf/bin/mota-smoke start smoke1
```

安装优先检查并复用`installation.json`中原CPU/CUDA解释器：检查固定软件版本、NumPy转换、CUDA可用性及真实小张量反向计算，代码更新本身不会触发整套CUDA下载。验证失败才尝试独立环境安装，不修改原CPU入口。无可用CUDA环境时明确记设备不可用，不声称GPU覆盖通过。

安装后运行`/root/autodl-tmp/p48-perf/bin/mota-smoke start smoke1`。安装另计，从start命令创建控制状态起，环境检查、预热、失败、调度和打包共享最多600秒绝对截止；可以提前结束。关闭终端继续。不会自动重跑、自动追加预算或启动一小时实验。名称已存在则拒绝覆盖。

日常单行命令：

```bash
/root/autodl-tmp/p48-perf/bin/mota-smoke status smoke1
/root/autodl-tmp/p48-perf/bin/mota-smoke stop smoke1
/root/autodl-tmp/p48-perf/bin/mota-smoke resume smoke1
/root/autodl-tmp/p48-perf/bin/mota-smoke export smoke1
```

resume保留原截止时间，已记录项不重跑；结束后拒绝再次恢复。停止会终止当前诊断子进程，而安全保存/恢复由独立的真实Runner STOP检查覆盖；诊断被强制中断时标记未完成，绝不宣称已保存。

## 覆盖和上限

| 项目 | 子进程上限 | 验证 |
|---|---:|---|
| 新鲜断点 | 65秒 | 原版串行、四局真实结束和样本、100模拟规则审计、保留7次模拟的未完成树 |
| 保存恢复 | 65秒 | 原版和packed完整语义比较，负例，恢复后的下一真实决策 |
| CPU局部检查 | 35秒 | 编码、哈希、神经网络和真实SGD；故意使packed保存检查失败，其他子项继续 |
| 两个CPU进程 | 65秒 | 四局合并至一个协调器，单一共享训练循环执行真实SGD |
| 共享批量服务 | 25秒 | CPU批量内核、缓存命中、模型更新、失效和旧版本拒绝 |
| CUDA | 80秒 | 独立已验证环境，CPU金标准数值检查、真实CUDA前向/批量/SGD；无设备明确报告 |
| 学习和恢复 | 125秒 | 接近完成的测试断点、两种保存恢复后下一SGD一致、N1继续100模拟搜索、STOP/恢复/预算/互斥 |
| 受控失败/超时 | 3秒/2秒 | 故意失败后继续超时检查、真实调度和导出 |

合计525秒子项上限；终止宽限、调度、监测和打包使用余量。前项到上限即终止进程组，不吃掉后项配额；剩余时间不足单项上限加42秒时标记未执行。最后保留打包时间，不启动新计算。硬中断的子项保留日志，不拼接成完整结果。

`candidate_selection_schedule`使用正式代码的依赖筛选、选择和后续对照调度函数，派发缩小工作量的真实子进程；600/300标签只表示所覆盖的正式分支，绝不是600/300秒性能对照。完整搜索和学习正确性另由上面的真实Runner检查，不靠调度占位输出证明。

## 测试断点来源

只使用本次新鲜随机网络及其四局实际样本。为了在时间内覆盖训练发布路径，明确注入`new_positions=512`测试计数，正式配置的门槛512、网络、搜索100、批量64、SGD/学习率/曝光公式均不修改。构造阶段执行63次真实SGD，仅跳过构造阶段中间写盘；随后保存真实的63/64训练断点，两种保存各恢复后执行最后一次真实SGD及后续搜索。测试计数与真实合格位置分别记录，**这不是合格的正式训练账本，不能计入产能**。恢复前后必须保持同一绝对预算。

## 修复和失败隔离

冻结的P45文件不变。新`semantic.compare`逐字段比较Node树、数组dtype/shape/data/writeable、模型、优化器、进度、Python/NumPy/Torch随机状态以及Generator和BitGenerator的类型/内部状态。比较不抽样、不前进RNG、不修改对象。未知对象明确报类型和路径，禁止repr地址作为等价依据。精确比较，不放宽浮点容差。

原版Torch对人工只读数组会丢失只读标志；这个负例必须被检测，并与真实断点的结果分开。优化保存必须保留它，不能静默修正比较对象。

micro保存、哈希、JSON、各CPU/CUDA配置分别落盘并带traceback。packed失败只排除依赖packed的候选；原版保存或共同CPU基础失败会阻断正式完整闭环，先保留独立GPU诊断。GPU依赖CPU金标准文件；该文件缺失时明确阻断数值门，不声称验证完成。冒烟额外注入一次保存失败，验证CPU局部计算和GPU派发仍继续。

## 结果

自动结果包：`/root/autodl-tmp/p48-perf/smoke-tests/smoke1/results.zip`。

包内`COVERAGE.txt`、`coverage.csv/json`为覆盖表，状态为通过/失败/超时/未执行/设备不可用。`controlled_failure`和`controlled_timeout`必须保留真实失败/超时状态，并由`failure_isolation`说明是否符合预期；不能把它们改为候选通过。包含环境、逐项日志、资源采样、错误路径及最终预算标记，排除权重和全量断点。必要完整断点留在本机测试目录以便排错。

`export.json`及`export-verification.json`在结果包旁，记录含打包总耗时/完整性；后者在首次打包完成后写入，手动export可将其收入结果包。

当前发布只证明本地CPU准备与检查。实际AutoDL和CUDA覆盖必须以用户返回结果包判断，不能根据设备型号预先标通过。流程通过后再单独安排正式一小时实验。
