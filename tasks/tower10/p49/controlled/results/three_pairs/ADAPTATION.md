# 原CNN受控对照适配

所有科学条件见冻结CONTROLLED_PLAN.md，实际结果见REPORT.md。
Network直接alias旧地图Network，无新网络类。原P45 Block和forward保持同一函数对象，123994参数，参数名/形状/dtype及配对BN缓冲一致。
A复用原地图Observation、FastMask/InteractionMask及环境；B复用已认证拓扑Session与Observation.tensors，固定e00—e20顺序完整序列化1456+74为34×5×9。B为容器，不是物理地图，不具有重排不变性，不是消息网络。
显式A/B动作字典在离线数据准备阶段读取独立验收映射；B正式worker不读物理映射，21项动作与A对齐，其余692项永久屏蔽。
原P45 Runner源码原样执行，仍用713数组；绑定同一个Network工厂（deepcopy配对权威模板）、Search及原contracts。没有修改原P45、地图学习或旧关系网络文件。
Search仅继承原begin/expand/simulate/finish，在原finish返回后读出Q、先验、访问数和资源差；无随机调用或选择公式变化。
独立进程隔离phase/seed/condition；原噪声/动作种子公式和NumPy SGD样本流，评价保存/恢复全部训练RNG。模板只随机生成一次/配对；worker meta加载及deepcopy不再抽初态；实际torch/NumPy/Python RNG与优化器记录在完整检查点与hash账本。
阶段一56步拟合直接复用原CE/MSE/L2/SGD公式，固定batch索引，不伪称在线闭环；阶段二原Runner完整闭环。
所有12个正式首步都检查输入/标签/损失/梯度/参数有限性；首次预检未执行SGD。BN训练计数逐步等于实做SGD，评价不改变模型hash，保存重载与跨condition/phase拒绝已通过。
旧76849状态/266443转移的语义认证复用；新边界只检查52条真实轨迹940位置、编码往返、source state/obs hash、动作/标签/资源后果。没有重跑穷举或30000随机局。
B六个正式worker（拟合三条、在线三条）禁用格点函数profile调用为0，物理图/坐标映射/答案拒绝记录为空；见VALIDATION.json。
原CNN网络相同不意味着同输入下N0行为相同。环境实现为既有语义等价的不同表示，不声称字节完全相同；config中原T1.0是保留的历史样本合同字段，实际表示由environment_representation与独立schema/initial/rules标识。
所有旧交付192文件SHA未变，固定初始化/池/批次/配置/源码/父链/优化器RNG与原始轨迹随包，不混入旧权重。
