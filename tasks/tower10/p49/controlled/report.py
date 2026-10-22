"""Independent audit/report; no answers are imported by actors or SGD."""
import csv,gzip,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=[49001,49002,49003];CONDITIONS=['direct_policy_sample','direct_policy_argmax','MCTS100_no_noise_maxN']


def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root):
    root=Path(root);fit=[];online=[];games=[];verification=[];forks=[]
    for phase in ['fit','online']:
        for seed in SEEDS:
            witnesses=[]
            for condition in ['A','B']:
                job=root/phase/str(seed)/condition;progress=load(job/'progress.json');assert progress['complete']
                if condition=='B':assert progress['runtime_no_grid_profile']['forbidden_grid_calls']==0 and not progress['denied_input_reads']
                consume=load(job/'initial-consumption.json');witnesses.append(consume['actual_model_hash']);assert consume['phase']==phase
                verification.append({'phase':phase,'seed':seed,'condition':condition,'initial_hash':consume['actual_model_hash'],
                   'runtime_no_grid_profile':progress['runtime_no_grid_profile'],'denied_input_reads':progress['denied_input_reads'],'worker_seconds':progress['worker_seconds']})
                result=job/'fit' if phase=='fit' else job
                if phase=='fit':
                    ledger=load(result/'training.json');assert len(ledger)==56
                    schedule=np.load(root/'data'/f'batches-{seed}.npy')
                    assert all(row['sample_indices']==schedule[i].tolist() and row['finite'] for i,row in enumerate(ledger))
                    for row in load(result/'metrics.json'):fit.append({'seed':seed,'condition':condition,**row})
                    models={f'P49-controlled-fit-{seed}-{condition}-S{s}':s for s in [0,56]}
                else:
                    models={m['id']:m['generation'] for m in load(job/'chain/models.json')};assert len(models)==6
                    status=load(job/'chain/status.json');assert status['generation']==5
                    reflow=load(job/'N5-reflow.json');assert reflow['reload'] and len(reflow['episodes'])==4 and not reflow['N6_triggered']
                    for row in load(job/'rounds.json'):
                        assert row['new_positions']>=64 and row['new_episodes']%4==0
                        assert row['SGD_steps']==row['SGD_target']==math.ceil(8*row['new_positions']/64)
                        trainer=load(job/'chain'/f"training-N{row['generation']}.json")
                        assert trainer['steps_done']==trainer['steps_target']==row['SGD_steps']
                    for epfile in sorted((job/'chain').glob('episode-*.gz')):
                        ep=json.loads(gzip.open(epfile,'rt').read());assert ep['evaluation'] is False
                        assert ep['end']=='GOAL' and ep['z']==ep['task_hp']/900
                        for rec in ep['records']:
                            assert rec['new_simulations']==100 and sum(rec['added_N'])==100
                            assert [a+b for a,b in zip(rec['inherited_N'],rec['added_N'])]==rec['root_N']
                            assert rec['weight_hash']==ep['weight_hash']
                for f in sorted((result/'evaluation').glob('*.gz')):
                    group=[json.loads(line) for line in gzip.open(f,'rt')];ec=group[0]['condition']
                    assert len(group)==(16 if ec=='direct_policy_sample' else 1)
                    assert [r['seed'] for r in group]==(list(range(49201,49217)) if ec=='direct_policy_sample' else [49201])
                    for g in group:
                        assert g['outcome']=='success' and g['error'] is None
                        assert g['new_simulations']==(100*g['decisions'] if ec=='MCTS100_no_noise_maxN' else 0)
                        for rec in g['records']:
                            assert rec['selected_action_id'] in rec['legal_action_ids'] and all(i<21 for i in rec['legal_action_ids'])
                            if ec=='MCTS100_no_noise_maxN':assert sum(rec['added_N'])==100
                        generation=models[g['model_id']]
                        games.append({'phase':phase,'pair_seed':seed,'condition':condition,'generation_or_step':generation,
                           **{k:v for k,v in g.items() if k!='records'}})
                        # Event names read only in this reporting process.
                        for rec in g['records']:
                            if rec['resource_delta']['atk'] or rec['resource_delta']['def'] or rec['resource_delta']['hp']:
                                forks.append({'phase':phase,'pair_seed':seed,'condition':condition,'model':g['model_id'],
                                  'evaluation_condition':ec,'evaluation_seed':g['seed'],'step':rec['step'],
                                  'target':rec['target'],'before':rec['resources_before'],'after':rec['resources_after'],
                                  'prior':rec['network_prior'],'value':rec['prediction'],'Q':rec['action_Q'],'selected':rec['selected_action_id']})
                    hps=[g['final_hp'] for g in group]
                    summary={'seed':seed,'condition':condition,'generation_or_step':models[group[0]['model_id']],
                      'evaluation_condition':ec,'episodes':len(group),'mean_hp':statistics.mean(hps),'median_hp':statistics.median(hps),
                      'min_hp':min(hps),'max_hp':max(hps),'success':len(group),'legal_failure':0,'exception':0,'unfinished':0,
                      'decisions':sum(g['decisions'] for g in group),'simulations':sum(g['new_simulations'] for g in group),
                      'forwards':sum(g['network_forwards'] for g in group),'seconds':sum(g['seconds'] for g in group)}
                    # Separate evaluation tables keep offline/online interpretation distinct.
                    summary['phase']=phase;online.append(summary)
            assert len(set(witnesses))==1,'PAIRED_INITIAL_HASH_MISMATCH'
    assert len(games)==864
    fit_eval=[r for r in online if r['phase']=='fit'];online_eval=[r for r in online if r['phase']=='online']
    write(root/'fitting_metrics.json',fit);write(root/'evaluation_summary.json',online)
    with (root/'evaluation_games.jsonl').open('w') as f:
        for g in games:f.write(json.dumps(g,ensure_ascii=False)+'\n')
    write(root/'posthoc_resource_events.json',forks)
    verification_result={'passed':True,'jobs':verification,'evaluation_games':len(games),
      'fit_games':sum(g['phase']=='fit' for g in games),'online_games':sum(g['phase']=='online' for g in games),
      'fit_SGD_per_condition':56,'all_pair_initial_hashes_equal':True,'frozen_batches_used_exactly':True,
      'source_hashes_unchanged':all(sha(root/'executed_source'/rel)==expected for rel,expected in load(root/'sources.json')['files'].items())}
    assert verification_result['source_hashes_unchanged'];write(root/'VALIDATION.json',verification_result)
    with (root/'evaluation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(online[0]));writer.writeheader();writer.writerows(online)
    paired_fit=[];paired_online=[]
    for seed in SEEDS:
        for step in [0,10,27,37,47,56]:
            pair={c:next(r for r in fit if r['seed']==seed and r['condition']==c and r['step']==step) for c in ['A','B']}
            paired_fit.append({'seed':seed,'step':step,'B_minus_A':{split:{m:pair['B'][split][m]-pair['A'][split][m] for m in ['CE','KL','value_MSE','constant_MSE','prediction_mean','prediction_std']} for split in ['train','held']}})
        for generation in range(6):
            for ec in CONDITIONS:
                pair={c:next(r for r in online_eval if r['seed']==seed and r['condition']==c and r['generation_or_step']==generation and r['evaluation_condition']==ec) for c in ['A','B']}
                paired_online.append({'seed':seed,'generation':generation,'evaluation_condition':ec,'A':pair['A']['mean_hp'],'B':pair['B']['mean_hp'],'B_minus_A':pair['B']['mean_hp']-pair['A']['mean_hp']})
    write(root/'paired_differences.json',{'fitting':paired_fit,'online':paired_online})
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for i,seed in enumerate(SEEDS):
        for j,ec in enumerate(CONDITIONS):
            ax=axes[i,j]
            for c in ['A','B']:
                rows=sorted([r for r in online_eval if r['seed']==seed and r['condition']==c and r['evaluation_condition']==ec],key=lambda r:r['generation_or_step'])
                ax.plot([r['generation_or_step'] for r in rows],[r['mean_hp'] for r in rows],marker='o',label=c)
            ax.set_title(f'{seed}: {ec}');ax.set_ylabel('Final HP');ax.set_xlabel('N generation');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Same original CNN: A physical map vs B serialized object topology (container only)');fig.tight_layout()
    fig.savefig(root/'online_curves.svg');fig.savefig(root/'online_curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,10))
    for i,seed in enumerate(SEEDS):
        for j,(split,m) in enumerate([('train','CE'),('held','CE'),('train','value_MSE'),('held','value_MSE')]):
            ax=axes[i,j]
            for c in ['A','B']:
                rows=sorted([r for r in fit if r['seed']==seed and r['condition']==c],key=lambda r:r['step'])
                ax.plot([r['step'] for r in rows],[r[split][m] for r in rows],marker='o',label=c)
            if m=='value_MSE':ax.axhline(rows[0][split]['constant_MSE'],color='gray',ls='--',label='train-mean constant')
            ax.set_title(f'{seed}: {split} {m}');ax.set_xlabel('SGD step');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Fixed experience, paired state_dict and identical 56 minibatches');fig.tight_layout()
    fig.savefig(root/'fitting_curves.svg');fig.savefig(root/'fitting_curves.png',dpi=140);plt.close(fig)
    lines=['# P49 原CNN不变的地图／对象拓扑序列化受控对照','',
       'A：原地图输入＋原CNN。B：对象拓扑序列化＋同一个原CNN；34×5×9仅为序列化容器，不是物理地图，不是图消息网络。',
       '两阶段全部三对初始化完成；网络类、参数签名与123,994参数一致，配对完整state_dict（含BN）hash相同。详见VALIDATION.json。',
       '', '## 阶段一：固定经验拟合','',
       '真实训练池44局801位置；独立保留集8局139位置；保留集未参与更新，也不宣称新状态或跨地图泛化。每对A/B共享固定batch索引，均执行56步SGD。',
       '下表完整列出所有预登记检查点；CE/KL越小越好，价值MSE需结合固定常数基线与预测离散程度。常数为训练池均值0.32458315993896514，保留集不重新拟合。','',
       '|seed|输入|步|训练CE|训练KL|训练MSE|保留CE|保留KL|保留MSE|保留常数MSE|保留预测均值±标准差|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for seed in SEEDS:
        for c in ['A','B']:
            for r in sorted([r for r in fit if r['seed']==seed and r['condition']==c],key=lambda r:r['step']):
                tr=r['train'];h=r['held'];lines.append(f"|{seed}|{c}|{r['step']}|{tr['CE']:.6f}|{tr['KL']:.6f}|{tr['value_MSE']:.6f}|{h['CE']:.6f}|{h['KL']:.6f}|{h['value_MSE']:.6f}|{h['constant_MSE']:.6f}|{h['prediction_mean']:.6f}±{h['prediction_std']:.6f}|")
    lines+=['','![固定经验拟合](fitting_curves.svg)','','配对B−A的所有指标差值见paired_differences.json；完整预测、三组数据梯度、参数变化、BN缓冲与优化器/RNG身份均随归档保存。','',
       '### 固定经验第0和56步游戏评价','','|seed|输入|步|直接采样16局均值|argmax一局|MCTS100一局|','|---|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        for c in ['A','B']:
            for s in [0,56]:
                vals=[next(r['mean_hp'] for r in fit_eval if r['seed']==seed and r['condition']==c and r['generation_or_step']==s and r['evaluation_condition']==ec) for ec in CONDITIONS]
                lines.append(f'|{seed}|{c}|{s}|'+ '|'.join(f'{v:.3f}' for v in vals)+'|')
    lines+=['','## 阶段二：全新五轮闭环','','每阶段从种子重新构造随机N0，阶段二不读阶段一或历史训练权重；相同种子与同CNN会重现相同N0，阶段隔离的是训练状态和经验池。',
       '原均值Q PUCT/100新增模拟/根噪声/采样/4局cohort/64新位置/batch64/exposure8/replay8192/SGD/HP900标签保持不变。','',
       '|seed|代数|A直接采样均值|B直接采样均值|B−A|A argmax|B argmax|A MCTS100|B MCTS100|',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    gains=[]
    for seed in SEEDS:
        for n in range(6):
            vals={c:{ec:next(r['mean_hp'] for r in online_eval if r['seed']==seed and r['condition']==c and r['generation_or_step']==n and r['evaluation_condition']==ec) for ec in CONDITIONS} for c in ['A','B']}
            a=vals['A'];b=vals['B'];lines.append(f'|{seed}|N{n}|{a[CONDITIONS[0]]:.3f}|{b[CONDITIONS[0]]:.3f}|{b[CONDITIONS[0]]-a[CONDITIONS[0]]:+.3f}|{a[CONDITIONS[1]]:.0f}|{b[CONDITIONS[1]]:.0f}|{a[CONDITIONS[2]]:.0f}|{b[CONDITIONS[2]]:.0f}|')
        for c in ['A','B']:
            gains.append({'seed':seed,'condition':c,**{ec:next(r['mean_hp'] for r in online_eval if r['seed']==seed and r['condition']==c and r['generation_or_step']==5 and r['evaluation_condition']==ec)-next(r['mean_hp'] for r in online_eval if r['seed']==seed and r['condition']==c and r['generation_or_step']==0 and r['evaluation_condition']==ec) for ec in CONDITIONS}})
    lines+=['','![在线五轮全部模型](online_curves.svg)','','### 实际经验及SGD传播','','|seed|输入|更新|新局|新位置|应做/实做SGD|秒|','|---|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for c in ['A','B']:
            for r in load(root/'online'/str(seed)/c/'rounds.json'):
                lines.append(f"|{seed}|{c}|N{r['generation']}|{r['new_episodes']}|{r['new_positions']}|{r['SGD_target']}/{r['SGD_steps']}|{r['seconds']:.3f}|")
    lines+=['','在线输入影响动作、经验和应做SGD，这是同规则闭环的传播结果；没有强行抹平。六条链均重载N5并完整采集4局，不触发N6。',
       'MCTS是单次确定性maxN结果，未重复16次冒充独立随机证据。直接采样统一49201—49216，argmax/MCTS统一49201；评价保存恢复训练RNG，独立进程隔离各条件。','',
       '## 证据、边界与解释','',
       '运行时A沿用原物理图与原加速路径索引，B沿用已认证对象拓扑内核；环境并非字节相同。复用既有76849状态/266443转移的规则等价认证，本轮未重跑穷举或30000局随机采样。',
       'B保留冻结对象ID顺序，完整节点/关系/全局状态按C顺序序列化，无原始x/y、地图尺寸或平面位置摆放。它不具备节点重排不变性；CNN卷积邻近假设未必适合该序列化，这是预登记局限。',
       '三对初始化只构成小规模重复观察；B好坏均只归属于此编码与原CNN组合，不外推拓扑表示或图网络总体优劣。历史两个实验只作背景，不进入本轮配对成绩。',
       '本报告的梯度和参数变化只能证明更新通路；学习收益必须依据完整拟合指标和实际游戏评价。',
       '后验资源/战损事件保存在posthoc_resource_events.json，不作为训练先验、标签或强制验收路线。全部864局逐决策轨迹包含真实资源差、先验、价值、继承/新增/最终访问及Q。','',
       '完整证据含原始来源轨迹、配对输入、标签/动作字典、固定batch索引、初态、检查点、优化器/RNG、模型父链、训练账本、执行源码及配置。详见ARTIFACTS.json和归档清单。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'summary.json',{'complete':True,'fitting_final':[r for r in fit if r['step']==56],
       'online_N5_minus_N0':gains,'evaluation':online,'validation':verification_result})
    print(json.dumps({'complete':True,'evaluation_games':len(games),'online_N5_minus_N0':gains}),flush=True)


if __name__=='__main__':run(sys.argv[1])
