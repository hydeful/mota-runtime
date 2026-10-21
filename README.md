# mota-runtime：独立运行代码

> 首次发布为已验证的 P45 CPU 运行版；部署准备完成后，AutoDL 实际验收仍需用户返回输出。

这是长期复用的公开运行代码仓库。后续版本采用独立固定提交；已有运行继续绑定原提交，不自动跟随 main 更新。首次版本沿用已验证的 `p45` 命令和 `/root/autodl-tmp/p45` 目录。

基于私有工程固定提交 `63bdd80216edf825d887bc5316eb0ad78e7250ad`。
P45 原始源码、资产、算法配置逐字节保留；独立 Git 历史只包含运行所需内容。
来源和许可边界见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)、[NOTICE.md](NOTICE.md)。

P45 默认串行，工程 E1.3 / 算法 A1.1 / 拓扑 T1.0。搜索、网络推理、SGD
均为 **CPU**，1 intra-op / 1 inter-op，确定性算法。RTX 4090 不参与本程序计算。
网络 2×32、每决策新增100搜索、四局 cohort、512新位置触发、batch64、8曝光、
联合 SGD、回报和终止规则保持原样。新建从随机网络开始。

## 首次安装

需要 Linux、Python3.12、git、可访问 GitHub 和 PyTorch 官方 CPU wheel 源。
固定公开提交由交付消息给出，必须使用完整40位值替换下列 COMMIT。
先确认 `/root/autodl-tmp` 为数据盘；安装会显示其实际挂载点和可用空间。
仅把代码/环境/运行写入该数据目录，不修改系统预装环境。

```bash
git clone --no-checkout https://github.com/hydeful/mota-runtime.git /root/autodl-tmp/mota-runtime-bootstrap
```

```bash
git -C /root/autodl-tmp/mota-runtime-bootstrap checkout --detach COMMIT && bash /root/autodl-tmp/mota-runtime-bootstrap/scripts/bootstrap.sh COMMIT
```

bootstrap 不读取私有仓库凭据，抓取并核对固定公开提交。
若默认数据盘不存在则立即失败；其它机器可传第二个参数指定已核实数据目录。
安装检查 Python、源码/地图/规则 hash、实际网络前向、PyTorch/NumPy、CPU型号、
亲和性、配额、内存、GPU、磁盘和保存路径。输出 `INSTALL_OK ... device=cpu` 才成功。
精确满足 Torch2.14.1 / NumPy2.3.5 的预装依赖会被复用；不满足时只在
独立 venv 安装已验证版本。CUDA原环境保留。完整检查记录：`p45/install-check.json`。
重复安装保留已有运行。环境绑定提交；恢复同一运行须使用其原安装。

## AutoDL 第一轮（总预算最多720秒）

以下命令都为单行。第一次先新建和后台启动；关闭终端后重新打开再查看。
预算按新建时的绝对墙钟登记，暂停/关闭终端/等待不延长截止时间。
因此新建后应立即启动。关闭浏览器不会停止后台任务；机器关机仍会中断。

```bash
/root/autodl-tmp/p45/bin/p45 new cloud600 --seed 4801 --budget 720 && /root/autodl-tmp/p45/bin/p45 start cloud600 --seconds 600
```

```bash
/root/autodl-tmp/p45/bin/p45 status cloud600
```

600秒切片结束显示 `slice_finished`（或全局预算耗尽 `budget_finished`）。
先返回诊断文件；不要自行延长预算或重复新建来替代失败结果。
额外停止/恢复验收最多120秒，原截止时间不改变：

```bash
/root/autodl-tmp/p45/bin/p45 resume cloud600 --seconds 120
```

```bash
/root/autodl-tmp/p45/bin/p45 stop cloud600
```

`stop` 只请求安全停止；再次 status 确认 `stopped`。在搜索/SGD边界保存，
不强杀、不重新初始化。`resume` 保留同一模型、优化器、RNG、样本、树和预算。
截止后恢复明确失败；开始新的独立任务须用新名称和明确授权的新预算。

## 日常固定操作

将 `alias p45=/root/autodl-tmp/p45/bin/p45` 加入终端配置后，可用：

|操作|短命令|
|---|---|
|环境检查|`p45 check`|
|随机新建|`p45 new NAME --seed 4801 --budget 720`|
|后台启动|`p45 start NAME --seconds 600`|
|查看|`p45 status NAME`|
|安全停止|`p45 stop NAME`|
|同一运行恢复|`p45 resume NAME --seconds 120`|
|汇总诊断|`p45 diagnose NAME`|

NAME 只能用字母、数字、下划线、短横线。已有目录明确拒绝覆盖。
同一目录由 flock 保护，主进程和核心子进程共同持锁，重复启动明确失败。
后台使用独立 session、关闭 stdin、文件日志；不依赖当前终端或 Codex 会话。
`running / stopped / slice_finished / budget_finished / abnormal_exit` 分开显示。
正常结束及异常退出保留退出码、实际耗时和超出本切片/全局截止后的收尾秒数。

查看字段包括 `completed_episodes`（四局提交后的完整局数）、
`qualified_training_positions`、`model_updates`、`real_sgd_steps`、
`elapsed_core_seconds`、`remaining_wall_budget_seconds`、`source_commit`、
`model_id/model_hash`、`actual_compute_device`、更新后模型搜索消费记录。
这些计数来自真实持久状态；不承诺在任何机器600秒内必然达到训练阈值。
完整原始状态也包含在输出中。预算到期后保存不回填预算内成绩。

## 文件布局与故障

`/root/autodl-tmp/p45/` 下：`code/<commit>` 源码，`envs/<commit>` 环境，
`runs/NAME/logs` 日志，`models` 发布模型，`data` 完整局/训练记录，
`state` 原子断点、优化器、RNG和账本。保留 core 原路径的相对软链接；
搬迁/备份必须复制完整 `runs/NAME`（不能只复制 checkpoint）。
未发布训练中的模型及必要 pre-SGD 见证仍属于 state 断点。
不自动上传模型或完整训练数据到 GitHub。

故障先执行：

```bash
/root/autodl-tmp/p45/bin/p45 diagnose cloud600
```

把打印的 `/root/autodl-tmp/p45/cloud600-diagnostics.zip` 返回。
包只收集软件/资源检查、预算、状态、模型账本和末尾2MB程序日志，不收集环境变量、
认证信息、系统密钥、模型或完整样本。若安装失败尚无运行，返回完整安装报错及
存在时的 `install-check.json`。不要发密码、令牌或私钥。

## 验证边界

验证记录发布于仓库 `validation/`，注明被测固定运行提交。
本地独立验证与用户AutoDL实际验收分开；没有AutoDL实际输出时，结论只能是
“部署准备完成，云端待验证”。部署成功不代表学习增强。短测吞吐不冒充一小时实测。

本次公开发布仅调整仓库地址、发行元数据和说明；沿用已记录的600秒实测，未重复该试跑。
公开下载与固定提交内容核对在发布后记录，不会被当作AutoDL采集验收。
