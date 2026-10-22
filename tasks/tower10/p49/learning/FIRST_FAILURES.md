# 首次失败与修复

## 正式实验之前

1. 环境预检：系统Python缺少torch，报ModuleNotFoundError。没有开始采集或SGD。
2. 首次安装：尝试向系统Python安装torch2.10.0 CPU，因系统site-packages不可写而Permission denied。
   修复：在/workspace/p49-venv创建独立虚拟环境，安装torch2.10.0+cpu，不修改系统依赖。
   原始日志保留在工作区p49-torch-install.log与p49-torch-venv-install.log，均未涉及训练。
3. 代码核对发现原P45 Search.begin(evaluation=True)写死800模拟。这是已存在的接口冲突，未执行错误预算的正式评价。
   修复：P49 Search子类仅提供明确的evaluation_new_simulations=100入口，原P45默认不改。

接口核对两次均通过，均无SGD；第二次增加正常可达的必死战斗mask核对。
没有丢弃错误训练模型或删除失败实验结果。

## 正式实验

首次正式SGD的检查结果见chain/first-formal-step.json。
如出现异常，原始progress.json、events.jsonl与traceback为准；本文件不预先声明正式实验通过。

正式结果：没有采集、评价、SGD或重载执行异常，无须从N0重开。56步应做SGD全部完成，首次正式更新检查通过。

## 收尾首次身份核对失败

ARTIFACTS初版在report进程中生成，但控制器随后更新了最终progress.json，因此仅progress.json的SHA不匹配。
首次核对明确报告artifact_identity_mismatches=['progress.json']。
保留ARTIFACTS.before-finalization.json，修复为最终状态写入后生成清单，并对全部归档文件核对。
这是收尾身份清单时序问题，不涉及采集/训练/评价数值；没有删除原始结果、重置预算或另起实验。
最终代码仅增加末尾清单刷新；本轮实际执行源码另存executed_source/，其SHA和正式前固定提交可复核。

## 远端读回的换行差异

首次固定提交111文件身份核对中，只有evaluation_summary.csv的Git blob SHA不符。
原因是交付脚本用read_text读取CSV，将原CRLF换行为LF；数值和统计没有变化，完整证据包仍保留原字节。
修复为read_bytes().decode('utf-8')上传原始UTF-8内容，恢复CSV的精确字节身份；随后重新按最终固定提交核对。
该修复只涉及远端交付序列化，没有重新训练或评价。
