"""Independent post-experiment reporting; the only consumer of acceptance answers."""
import csv
import gzip
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def report(root):
    root=Path(root);began=time.monotonic();repo=Path(__file__).resolve().parents[4]
    acceptance=repo/'tasks/tower10/p49/acceptance'
    optimum=json.loads((acceptance/'exhaustive.json').read_text())['max_success_hp']
    random_sample=json.loads((acceptance/'sample.json').read_text())
    random_mean=sum(b['mean_goal_hp']*b['wins'] for b in random_sample['batches'])/random_sample['wins']
    progress=json.loads((root/'progress.json').read_text());config=json.loads((root/'config.json').read_text())
    models=json.loads((root/'chain/models.json').read_text());all_rows=[];table=[]
    for identity in models:
        for condition in config['evaluation']['conditions']:
            path=root/'evaluation'/(identity['id']+'-'+condition+'.jsonl.gz')
            rows=[]
            if path.exists():
                with gzip.open(path,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
            all_rows.extend(rows);scores=[r['final_hp'] for r in rows if r['outcome']=='success']
            counts={outcome:sum(r['outcome']==outcome for r in rows) for outcome in ['success','legal_failure','exception','unfinished']}
            entry={'generation':identity['generation'],'model_id':identity['id'],'weight_hash':identity['hash'],'condition':condition,'episodes':len(rows),'planned':len(config['evaluation']['seeds']),**counts,'mean_hp':statistics.mean(scores) if scores else None,'median_hp':statistics.median(scores) if scores else None,'min_hp':min(scores) if scores else None,'max_hp':max(scores) if scores else None,'optimal_count':sum(hp==optimum for hp in scores),'optimal_fraction':sum(hp==optimum for hp in scores)/len(rows) if rows else None,'decisions':sum(r['decisions'] for r in rows),'new_simulations':sum(r['new_simulations'] for r in rows),'network_forwards':sum(r['network_forwards'] for r in rows),'seconds':sum(r['seconds'] for r in rows)}
            table.append(entry)
    with (root/'evaluation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write_json(root/'evaluation_summary.json',{'certified_optimum_read_only_in_report':optimum,'random_baseline_read_only_in_report':random_mean,'rows':table})
    reflow=json.loads((root/'N5-reflow.json').read_text()) if (root/'N5-reflow.json').exists() else None
    trainings=[json.loads(p.read_text()) for p in sorted((root/'chain').glob('training-N*.json'))]
    closed=progress['complete'] and len(models)==6 and len(trainings)==5 and bool(reflow)
    comparisons={}
    for condition in config['evaluation']['conditions']:
        a=next((r for r in table if r['generation']==0 and r['condition']==condition),None)
        b=next((r for r in table if r['generation']==5 and r['condition']==condition),None)
        comparisons[condition]={'N0_mean':a['mean_hp'] if a else None,'N5_mean':b['mean_hp'] if b else None,'delta_N5_minus_N0':b['mean_hp']-a['mean_hp'] if a and b and a['mean_hp'] is not None and b['mean_hp'] is not None else None}
    n0_mcts=next((r for r in table if r['generation']==0 and r['condition']=='MCTS100_no_noise_maxN'),None)
    ceiling=bool(n0_mcts and n0_mcts['episodes']==16 and n0_mcts['optimal_count']==16)
    summary={'interface':'formal first-step audit and preflight receipts required','closed_loop_complete':closed,'comparisons':comparisons,'N0_MCTS_ceiling':ceiling,'evaluation':table,'rounds':progress['rounds'],'models':models,'N5_reflow_complete':bool(reflow),'total_SGD_steps':sum(t['steps_done'] for t in trainings),'training_seconds':sum(t['finished_monotonic']-t['started_monotonic'] for t in trainings),'evaluation_seconds':sum(r['seconds'] for r in all_rows),'formal_seconds_before_report':progress['experiment_seconds_before_report'],'exceptions':progress.get('error'),'denied_behavior_reads':progress['denied_behavior_reads']}
    write_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    for condition,label,color in [('direct_policy_sample','Policy only: sampled','#4477aa'),('MCTS100_no_noise_maxN','Policy + MCTS100: maxN','#cc7733')]:
        rows=[r for r in table if r['condition']==condition and r['mean_hp'] is not None]
        xs=[r['generation'] for r in rows];ys=[r['mean_hp'] for r in rows]
        ax.plot(xs,ys,'o-',color=color,label=label)
        ax.fill_between(xs,[r['min_hp'] for r in rows],[r['max_hp'] for r in rows],color=color,alpha=.10)
    ax.axhline(optimum,color='#777777',linestyle='--',linewidth=1,label=f'Certified optimum: {optimum}')
    ax.axhline(random_mean,color='#999999',linestyle=':',linewidth=1,label=f'Uniform random baseline: {random_mean:.2f}')
    ax.set(xlabel='Published generation',ylabel='Final HP',title='P49 one initialization: N0 to N5 (16 games / condition)',xticks=list(range(6)),ylim=(0,650));ax.grid(alpha=.15);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(root/'evaluation_curves.svg');fig.savefig(root/'evaluation_curves.png',dpi=150);plt.close(fig)
    lines=['# P49地图输入网络＋对象交互搜索的5轮学习实验','',
           '地图、数值、初态和Goal未变。网络主体2个残差块×32通道，输入34×5×9，713输出中21个绑定对象、692个永久mask。',
           '原P45 Runner源码原样在P49依赖命名空间执行；采集、SGD、回放、发布与恢复合同沿用，P45本身未修改。','',
           '## 三项独立结论','',
           f"1. 接口：见ADAPTATION.md、interface.json与chain/first-formal-step.json；正式异常={progress.get('error')!r}。",
           f'2. 闭环：N0→N5及重载N5后的4局回流完成={closed}；累计SGD={summary["total_SGD_steps"]}步。',
           '3. 学习收益：按以下全部六轮结果分别判断，不以参数变化或通关率代替学习证据。','',
           '|模型|直接策略均值|中位数|范围|最优比例|MCTS100均值|中位数|范围|最优比例|',
           '|---|---:|---:|---|---:|---:|---:|---|---:|']
    for gen in range(len(models)):
        a=next(r for r in table if r['generation']==gen and r['condition']=='direct_policy_sample')
        b=next(r for r in table if r['generation']==gen and r['condition']=='MCTS100_no_noise_maxN')
        def fmt(r):return f"{r['mean_hp']:.2f}|{r['median_hp']:.1f}|{r['min_hp']}–{r['max_hp']}|{r['optimal_count']}/{r['episodes']}" if r['mean_hp'] is not None else '未完成|—|—|—'
        lines.append(f'|N{gen}|{fmt(a)}|{fmt(b)}|')
    lines+=['','![两条件完整曲线](evaluation_curves.svg)','',
            '阴影为逐局最小值至最大值，曲线为均值；全部逐局结果与决策记录保留在evaluation/。',
            '直接策略按合法softmax分布采样；MCTS100无根噪声、maxN，确定性引擎与固定权重下16个seed可能产生相同路线，不能当16条独立随机路线解释。','']
    for condition,c in comparisons.items():lines.append(f"- {condition}：N0={c['N0_mean']}，N5={c['N5_mean']}，变化={c['delta_N5_minus_N0']} HP。")
    gains=[c['delta_N5_minus_N0'] for c in comparisons.values() if c['delta_N5_minus_N0'] is not None]
    if not gains or max(gains)<=0:lines+=['','最小闭环已跑通，学习收益尚未得到验证。' if closed else '闭环未完成，学习收益证据不足。']
    else:lines+=['','在本初始化、预登记的16局评价中观察到上述均值变化；它不是跨初始化稳定收益的统计证明，也不要求逐轮单调。']
    if ceiling:lines+=['','N0＋MCTS100已稳定最优，此条件存在评价天花板；仍按原计划完成五轮，没有改地图或搜索预算。']
    lines+=['','## 训练与模型回流','',
            '|更新|实际新局|实际新位置|应做/实做SGD|采集＋训练＋发布秒|','|---|---:|---:|---:|---:|']
    for r in progress['rounds']:lines.append(f"|N{r['generation']-1}→N{r['generation']}|{r['new_episodes']}|{r['new_positions']}|{r['SGD_target']}/{r['SGD_steps']}|{r['collect_train_publish_seconds']:.3f}|")
    lines+=['',f"独立评价耗时{summary['evaluation_seconds']:.3f}秒；训练阶段{summary['training_seconds']:.3f}秒。正式实验至汇总前{summary['formal_seconds_before_report']:.3f}秒；含图表与收尾的最终墙钟见progress.json。",'适配开发时间另列在ADAPTATION.md，不计入正式3600秒预算。',
            '', '所有正式新局按原规则完整终止；没有复制位置或丢弃低分局。每轮在64个新位置后等4局cohort结束，SGD=ceil(8×本轮新位置/64)。优化器持续保留动量；回放窗口8192。',
            'N5回流4局独立登记在N5-reflow.json，只验证重新加载后的权重消费，不触发N6。',
            '', '## 证据与局限','',
            '- evaluation/*.jsonl.gz：六模型、两条件、全部逐局和逐决策结果，含模拟数与真实前向数。',
            '- chain/episode-*.json.gz：全部正式采集轨迹、根访问标签及真实终局HP标签。',
            '- chain/training-N*.json、first-formal-step.json：实际SGD账本、采样位置、数据梯度、损失和首次更新检查。',
            '- chain/models.json、config.json、sources.json、schema.json：模型父链、权重hash、配置与源码身份。',
            '- checkpoint与权重另列ARTIFACTS.json；异常/未完成不会生成失败标签。',
            '- 正式运行进程安装读取guard，禁止旧行为与认证答案；最优值及随机基线只由本独立报告进程读取。',
            '- 一条seed=49001学习链、每条件16局；不能推断跨初始化稳定性。',
            '', '## 拓扑架构边界','',
            '规则与可达性由原拓扑引擎执行；内部图仍含可通行格顶点；搜索动作是21个对象交互；网络接收34×5×9二维平面并采用CNN。',
            '本轮结果属于地图输入网络＋对象交互搜索，不能称为完整拓扑表征的学习验证或神经网络直接学习拓扑关系图。该方案源于任务设计允许地图输入/CNN的选项。',
            '目标架构要求网络直接接收事件/对象节点特征与连接、阻挡、资源条件，空地/走廊不是学习或逐步选择的决策节点；平面坐标用于展示、导出及回放。',
            '后续需替换观测/网络输入、对象策略映射、关系批处理、schema与缓存/checkpoint身份；预计复用原规则、对象交互、PUCT、访问分布标签、终局回报、SGD/回放及评价逻辑，但仍需核查走廊收缩和自动事件语义、恢复与批处理合同。',
            '本轮不实现新架构，不预定某类图神经网络；无论本轮是否有收益，都不能据此肯定或否定尚未实现的目标拓扑架构。',
            '', '## 下一步','',
            '优先核查直接策略与MCTS根访问分布之间的差距。N5直接采样仍未取得最优路线且较N4回落；现有证据不足以归因于训练量、损失或网络架构。本轮不追加实验。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='ARTIFACTS.json':files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    write_json(root/'ARTIFACTS.json',{'files':files,'note':'models/checkpoints and full samples retained; small summary excludes weights','report_seconds':time.monotonic()-began})
    print(json.dumps({'closed_loop':closed,'comparisons':comparisons,'N0_MCTS_ceiling':ceiling,'evaluation_games':len(all_rows),'report_seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':report(sys.argv[1])
