# 可复核范围、来源与复现入口

## 固定来源

- 公开运行基底：`hydeful/mota-runtime@49ef801d2846e45edb9ec196b5d7f6744a340ff7`。
- P48被测版本：`hydeful/mota-runtime@d8da913bc466db6276ef3222abf0e2bd063820d6`。与上述公开基底逐文件树对照，差异仅`perf/SMOKE.md`，执行代码相同。
- P49发布快照：用户授权从私有`hydeful/mota-project@4407df153d5419d4b9deeba02254d553c3e62528`导出；历史地图运行提交61899150c73d401d73ff0ac45626d921e98e5d5c，对象关系运行提交5da003c726eeb8d5939b62e6c8700612652e5a14。
- P48原始摘要：adf01c88b54891ba90aad0da8f7641f8efda749b；深入复盘/重算：0c05a46c095b2b9d2a78d73df27251ccf05ae966；随机铁剑探测：94485bf9e4d397f00778ebdce0a4b77c11e9937e。
- 原交接：0f732f99d9c0db093258fa807a1dbd4e879fc111。

所有必要阅读内容已公开复制在本分支，外部AI不需要私有仓库权限。原文内部的私有URL与历史绝对路径为来源标记，不是可复现的当前路径；请从本入口查找公开对应文件。

`PUBLIC_MANIFEST.json`记录公开新增文件的SHA256、Git blob身份和来源分类；源文件保持原字节。原有根`SOURCE_MANIFEST.json`只覆盖原部署快照，不是本次全部文件的清单。

## 已包含与未包含

已包含P45/P48执行代码、规则/状态JSON、P49环境/三种实现/配置/认证文本/全部评价jsonl与成绩/逐代训练统计、P48 state与计时重算、部分更早报告、原始episode样例。

未包含：P49完整权重、所有压缩恢复档案与数据张量大包，P48原归档就缺失的云端完整episode/模型/子项微测文件，以及所有早期阶段原始日志。历史ARTIFACTS清单与restore.py按原文保留，但本公开包不提供它们要求的所有分片；直接运行恢复脚本会缺文件。这不影响读代码和原始成绩，不代表能只凭本包复演原训练。

外部系统仅提供原始链接和本项目撰写的摘要，没有整体复制第三方项目。原NOTICE继续适用；没有添加覆盖所有内容的开源许可。

## 不训练的本地检查

在仓库根目录：

```bash
python review/20261022/verify.py
python -m unittest tasks.tower10.p49.test_environment
```

第一条仅核验本发布清单及Python语法、JSON；第二条复用已有6项环境测试，覆盖原规则、合法动作/阻挡、钥匙、存活边界、完整路线及浏览器桥接。第二条需NumPy，不需训练或GPU。运行结果见`PUBLICATION_VALIDATION.json`。

本次没有重跑耗时的整图穷举、训练、搜索基准或AutoDL部署。既有穷举与对象图认证原件在P49目录，按历史结果引用，不冒充本次重新认证。

## 如需后续复现实验（本次未执行）

先读P49的EXPERIMENT_PLAN/CONTROLLED_PLAN和各experiment.py、P48 perf/README及control.py的CLI，再在隔离环境按各自冻结依赖准备。根requirements和P49 learning/requirements的软件版本不同，需分别处理；不要为了“统一环境”改写历史证据。

P49实验CLI会启动采集、搜索和训练，不属于这里的只读复核。要重跑时必须重新冻结预算、数据来源、种子、评价与停止规则，另立输出目录，不覆盖原档案。P48现场资源与软件也要重测后桥接历史基线。

大型权重缺失使原权重逐位复演不可用；可以通过重新执行冻结代码产生新run验证，但不得把新结果冒称原run，也不得对未经重跑的新依赖承诺逐位复现。

