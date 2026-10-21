"""Persistent-budget experiment controller and small human-facing CLI."""
import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from perf.common import SOURCE, BASELINE, TOLERANCES, dump, load, stamp, utc, verify_source, event

HOME=Path(os.environ.get('P48_PERF_HOME','/root/autodl-tmp/p48-perf')).resolve()

def revision():return subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
def lock(root):
    f=(root/'controller.lock').open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();raise RuntimeError('EXPERIMENT_ALREADY_RUNNING')
    return f
def alive(root):
    try:f=lock(root);f.close();return False
    except RuntimeError:return True
def environment():
    import platform,psutil
    q=Path('/sys/fs/cgroup/cpu.max');text=q.read_text().strip() if q.exists() else None
    quota=None
    if text and text.split()[0]!='max':quota=int(text.split()[0])/int(text.split()[1])
    model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None)
    return {'cpu_model':model,'cpu_max':text,'quota_cores':quota,'visible_cpus':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),
            'effective_cpu_ceiling':min(len(os.sched_getaffinity(0)),quota or len(os.sched_getaffinity(0))),
            'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
            'memory_visible':psutil.virtual_memory()._asdict(),'platform':platform.platform(),'python':sys.version,
            'disk':shutil.disk_usage(HOME)._asdict(),'source_commit':revision(),'baseline_commit':BASELINE,
            'cpu_env':sys.executable,'gpu_env':load(HOME/'installation.json',{}).get('gpu_python')}
def child_env():
    return dict(os.environ,PYTHONPATH=str(SOURCE),PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')

def monitor(pid):
    import psutil
    try:procs=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True)
    except psutil.Error:return [],0
    rows=[]
    for p in procs:
        try:
            c=p.cpu_times();rows.append({'pid':p.pid,'created':p.create_time(),'cpu_seconds':c.user+c.system,'rss_bytes':p.memory_info().rss})
        except psutil.Error:pass
    return rows,sum(x['rss_bytes'] for x in rows)

def prepare_initials(root,state):
    import numpy as np,torch
    from perf.common import configure
    from tasks.tower10.p45.network import Network,model_hash
    configure();initials={};directory=root/'fresh-initials';directory.mkdir(exist_ok=True)
    for seed in (4801,4802,4803):
        torch.manual_seed(seed);net=Network();path=directory/f'seed{seed}.npz'
        np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
        initials[str(seed)]={'path':str(path),'hash':model_hash(net),'seed':seed,'torch':str(torch.__version__),'generated_utc':utc(),'provenance':'Fresh randomCPU2.14.1 N0 generated inside THIS experiment, not a historical weight; shared across backends for identical initialization'}
    dump(root/'initials.json',initials);state['initials']=initials;dump(root/'state.json',state)

def task(root,state,label,argv,python,limit,held,trial=None):
    """Every launch/failure/warmup/audit is within the shared absolute deadline."""
    directory=root/'attempts'/f'{len(state["attempts"]):02d}-{label}';directory.mkdir(parents=True)
    record={'label':label,'directory':str(directory.relative_to(root)),'started_utc':utc(),'argv':argv,'status':'running','python':python}
    if trial:record['trial']=trial
    state['attempts'].append(record);state['active']=len(state['attempts'])-1;dump(root/'state.json',state)
    start=time.monotonic();cpu0=time.process_time();children0=resource.getrusage(resource.RUSAGE_CHILDREN)
    children_cpu0=children0.ru_utime+children0.ru_stime;peak=0;totals={};gpu_next=0
    stop_at=min(time.time()+limit,stamp(state['deadline_utc'])-22)
    with (directory/'console.log').open('wb') as log:
        p=subprocess.Popen([python,*argv],stdout=log,stderr=log,stdin=subprocess.DEVNULL,cwd=SOURCE,env=child_env(),start_new_session=True,pass_fds=(held.fileno(),))
        record['pid']=p.pid;dump(root/'state.json',state);requested=None
        while p.poll() is None:
            rows,rss=monitor(p.pid);peak=max(peak,rss)
            for row in rows:totals[(row['pid'],row['created'])]=max(totals.get((row['pid'],row['created']),0),row['cpu_seconds'])
            import psutil
            controller_rss=psutil.Process().memory_info().rss
            peak=max(peak,rss+controller_rss)
            sample={'utc':utc(),'elapsed':time.monotonic()-start,'processes':rows,'child_tree_sum_rss_bytes':rss,'controller_rss_bytes':controller_rss,'sum_rss_bytes':rss+controller_rss}
            if time.monotonic()>=gpu_next and shutil.which('nvidia-smi'):
                try:
                    g=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                    sample['gpu_machine_csv']=g.stdout.strip();sample['gpu_exit_code']=g.returncode
                except subprocess.TimeoutExpired:sample['gpu_error']='nvidia-smi timeout'
                gpu_next=time.monotonic()+2
            event(directory/'resources.jsonl',**sample)
            if time.time()>=stop_at or (root/'STOP').exists():
                if requested is None:
                    requested=time.monotonic()
                    if trial:(Path(trial['run'])/'STOP').touch()
                    else:
                        try:os.killpg(p.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                if time.monotonic()-requested>7:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            time.sleep(.5)
        rc=p.wait()
    children1=resource.getrusage(resource.RUSAGE_CHILDREN)
    related_cpu=time.process_time()-cpu0+children1.ru_utime+children1.ru_stime-children_cpu0
    record.update(status='finished' if rc==0 and requested is None else 'incomplete',exit_code=rc,
                  wall_seconds=time.monotonic()-start,finished_utc=utc(),sampled_peak_sum_rss_bytes=peak,
                  sampled_cpu_seconds_lower_bound=sum(totals.values()),controller_monitor_cpu_seconds=time.process_time()-cpu0,
                  all_related_cpu_seconds=related_cpu,cpu_accounting_complete=rc==0,
                  resource_sampling_seconds=.5,gpu_sampling_seconds=2,
                  resource_note='sumRSS counts shared pages more than once; sampled peak is a lower bound. Worker result provides exact waited-descendant CPU for normal exits. GPU utilisation/memory is machine-wide, may include other jobs.')
    if trial:
        result=load(Path(trial['run'])/'result.json')
        if result:
            record['result']=result
            record['valid_full_loop']=bool(record['status']=='finished' and result['pass'] and result['full_window_complete'] and result['model_updates']>=1 and result['sgd_steps']>=1 and result['updated_model_decisions']>=1)
        else:record['status']='incomplete'
    state['active']=None;dump(root/'state.json',state);return record

def trial(root,state,label,opts,seconds,seed,held,cohorts=0,python=None,learning_gate=False):
    # Unique path on every attempt. Interrupted segments are never concatenated.
    path=root/'runs'/f'{len(state["attempts"]):02d}-{label}';path.mkdir(parents=True)
    opts=dict(opts)
    if state.get('initials'):
        opts['initial_state']=state['initials'][str(seed)]['path'];opts['initial_hash']=state['initials'][str(seed)]['hash']
    dump(path/'options.json',opts)
    argv=['-B','-m','perf.worker','--root',str(path),'--options',str(path/'options.json'),
          '--seconds',str(seconds),'--deadline',str(stamp(state['deadline_utc'])-25),'--seed',str(seed)]
    if cohorts:argv+=['--cohorts',str(cohorts)]
    if learning_gate:argv+=['--learning-gate']
    return task(root,state,label,argv,python or sys.executable,seconds+60,held,
                {'run':str(path),'options':opts,'requested_seconds':seconds,'seed':seed,'cohorts':cohorts})

def good(state,label):
    return next((r for r in reversed(state['attempts']) if r['label']==label and r.get('valid_full_loop')),None)

def summarize(root,state):
    rows=[]
    for r in state['attempts']:
        x=r.get('result',{});duration=x.get('collection_seconds')
        rows.append({'experiment':r['label'],'status':r['status'],'valid_full_loop':r.get('valid_full_loop',False),
                     'outer_wall_seconds':r.get('wall_seconds'),'collection_seconds':duration,
                     'qualified_positions':x.get('qualified_positions'),'complete_episodes':x.get('complete_episodes'),
                     'committed_episodes':x.get('committed_episodes'),'decisions':x.get('decisions'),'simulations':x.get('simulations'),
                     'model_updates':x.get('model_updates'),'sgd_steps':x.get('sgd_steps'),'sgd_steps_due':x.get('sgd_steps_due'),
                     'sgd_steps_unfinished':x.get('sgd_steps_unfinished'),'updated_model_decisions':x.get('updated_model_decisions'),
                     'positions_per_hour_extrapolated':x.get('qualified_positions',0)/duration*3600 if duration else None,
                     'complete_episodes_per_hour_extrapolated':x.get('complete_episodes',0)/duration*3600 if duration else None,
                     'committed_episodes_per_hour_extrapolated':x.get('committed_episodes',0)/duration*3600 if duration else None,
                     'cpu_tree_seconds':x.get('cpu_tree_seconds'),'all_related_cpu_seconds':r.get('all_related_cpu_seconds'),
                     'avg_used_cores':r.get('all_related_cpu_seconds',0)/r['wall_seconds'] if r.get('wall_seconds') else None,
                     'quota_utilization_pct':100*r.get('all_related_cpu_seconds',0)/r['wall_seconds']/load(root/'environment.json',{}).get('effective_cpu_ceiling',1) if r.get('wall_seconds') else None,
                     'sampled_peak_sum_rss_bytes':r.get('sampled_peak_sum_rss_bytes'),
                     'exit_code':r.get('exit_code')})
    with (root/'summary.csv').open('w',newline='') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pairs=[]
    for base,candidate in [('baseline600','candidate600'),('repeat-baseline300','repeat-candidate300')]:
        a=good(state,base);b=good(state,candidate)
        if a and b:
            x=a['result'];y=b['result'];ratio=(y['qualified_positions']/y['collection_seconds'])/(x['qualified_positions']/x['collection_seconds'])
            owed=max(0,y['sgd_steps_unfinished']-x['sgd_steps_unfinished'])
            pairs.append({'baseline':base,'candidate':candidate,'qualified_positions_rate_ratio':ratio,
                          'outer_wall_including_init_and_reference_audit_rate_ratio':(y['qualified_positions']/b['wall_seconds'])/(x['qualified_positions']/a['wall_seconds']),
                          'candidate_additional_unfinished_sgd_steps':owed,'training_backlog_qualified':owed==0})
    recommend=len(pairs)==2 and all(x['qualified_positions_rate_ratio']>=1.2 and x['training_backlog_qualified'] for x in pairs)
    summary={'experiment_status':state['status'],'comparisons':pairs,'performance_improvement':'pending' if not pairs else 'see measured per-pair ratios',
             'default_recommendation':'eligible for review, not automatically changed' if recommend else 'keep original P45 default; insufficient repeated>=20% evidence or training debt',
             'script_delivery_and_performance_are_separate':True,'hourly_rates_are_extrapolations_not_one_hour_measurements':True,
             'elapsed_controller_seconds':time.time()-stamp(state['started_utc']),'budget_seconds':state['budget_seconds'],
             'remaining_seconds':max(0,stamp(state['deadline_utc'])-time.time()),'errors':[{'label':r['label'],'status':r['status'],'exit_code':r.get('exit_code')} for r in state['attempts'] if r['status']!='finished'],
             'controller_total_cpu_seconds':time.process_time(),'controller_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'all_recorded_related_cpu_seconds':sum(r.get('all_related_cpu_seconds',0)-r.get('controller_monitor_cpu_seconds',0) for r in state['attempts'])+time.process_time(),
             'selection':state.get('selection'),'decisions':state.get('decisions',[])}
    dump(root/'summary.json',summary)
    lines=['P48自动性能探测','状态：'+state['status'],f'已运行{summary["elapsed_controller_seconds"]:.1f}秒 / 总预算{state["budget_seconds"]}秒',
           '短测每小时产能均为外推，不是60分钟实测。','默认入口未改变。']
    for row in rows:
        lines.append(f'{row["experiment"]}: {row["status"]}; 合格位置={row["qualified_positions"]}; 更新={row["model_updates"]}; SGD={row["sgd_steps"]}; 欠SGD={row["sgd_steps_unfinished"]}')
    for x in pairs:lines.append(f'{x["candidate"]} / {x["baseline"]}: {x["qualified_positions_rate_ratio"]:.3f}倍；额外欠SGD={x["candidate_additional_unfinished_sgd_steps"]}')
    (root/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')

def export(root,state):
    with (root/'export.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        return _export(root,state)

def _export(root,state):
    summarize(root,state);target=root/'results.zip';tmp=root/'results.zip.tmp';started=time.time()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix in ('.pt','.tmp','.zip','.npz','.lock'):continue
            if p.is_symlink():continue
            if p.suffix=='.log' and p.stat().st_size>5_000_000:
                with p.open('rb') as f:f.seek(-5_000_000,2);z.writestr(str(p.relative_to(root))+'.tail',f.read())
            else:z.write(p,p.relative_to(root),compress_type=zipfile.ZIP_STORED if p.name.endswith('.gz') else zipfile.ZIP_DEFLATED)
        marker={'measured_utc':utc(),'controller_elapsed_including_payload_pack_seconds':time.time()-stamp(state['started_utc']),
                'payload_pack_seconds':time.time()-started,'late_past_global_deadline_seconds':max(0,time.time()-stamp(state['deadline_utc'])),
                'budget_seconds':state['budget_seconds'],'measurement_point':'All result payloads packed; only this tiny marker and ZIP central directory/atomic rename follow.'}
        z.writestr('FINAL_BUDGET_MARKER.json',json.dumps(marker,indent=2))
    os.replace(tmp,target)
    dump(root/'export.json',{'path':str(target),'bytes':target.stat().st_size,'pack_seconds':time.time()-started,'finished_utc':utc(),
                             'late_past_global_deadline_seconds':max(0,time.time()-stamp(state['deadline_utc'])),
                             'total_controller_seconds_including_pack':time.time()-stamp(state['started_utc']),
                             'excluded':'weights, optimizer/checkpoints, temporary files, credentials/environment variables. Only this experiment directory is read.'})
    return target

def controller(root):
    with lock(root) as held:
        state=load(root/'state.json')
        if state['code_commit']!=revision():raise RuntimeError('USE_ORIGINAL_EXPERIMENT_CODE_COMMIT')
        if state['started_utc'] is None:
            state['started_utc']=utc();state['deadline_utc']=dt.datetime.fromtimestamp(time.time()+state['budget_seconds'],dt.timezone.utc).isoformat()
        state['controller_pid']=os.getpid();state['status']='running';dump(root/'state.json',state)
        # Reconcile a normally completed child after controller interruption.
        for r in state['attempts']:
            if r['status']=='running':
                x=load(Path(r.get('trial',{}).get('run','/nonexistent'))/'result.json')
                r['status']='incomplete';r['recovered_after_controller_restart']=True
                if x:
                    r['result']=x;r['valid_full_loop']=bool(x['pass'] and x['full_window_complete'] and x['model_updates'] and x['sgd_steps'] and x['updated_model_decisions'])
        def remaining():return stamp(state['deadline_utc'])-time.time()
        def allowed(seconds=0):return not (root/'STOP').exists() and remaining()>seconds+30
        try:
            if not state.get('initials') and allowed(1500):prepare_initials(root,state)
            if not good(state,'baseline600'):
                if not allowed(1500):raise RuntimeError('INSUFFICIENT_BUDGET_FOR_REQUIRED_BASELINE_AND_CANDIDATE')
                baseline=trial(root,state,'baseline600',{},600,4801,held)
            else:baseline=good(state,'baseline600')
            if not baseline.get('valid_full_loop'):raise RuntimeError('BASELINE600_INCOMPLETE_OR_NO_REAL_UPDATE')
            baseline_path=Path(baseline['trial']['run'])
            if 'screened' not in state:
                micro_dir=root/'micro-cpu'
                cpu_probe=task(root,state,'micro-cpu',['-B','-m','perf.micro','--baseline',str(baseline_path),'--out',str(micro_dir)],sys.executable,120,held)
                micro=load(micro_dir/'micro.json',{})
                if not micro.get('correctness',{}).get('cpu1',{}).get('pass'):raise RuntimeError('CPU_MICRO_CORRECTNESS_GATE_FAILED')
                gpu_python=state.get('installation',{}).get('gpu_python');gpu={}
                if gpu_python and allowed(1400):
                    task(root,state,'micro-gpu',['-B','-m','perf.micro','--baseline',str(baseline_path),'--out',str(root/'micro-gpu'),'--gpu','--reference',str(micro_dir/'cpu-reference.npz')],gpu_python,150,held)
                    gpu=load(root/'micro-gpu/micro.json',{})
                else:state['decisions'].append('GPU unavailable/not installed or insufficient budget; no GPU success claimed')
                quota=load(root/'environment.json')['effective_cpu_ceiling']
                options=[('serial-packed',{'packed':True}),('serial-hash',{'direct_hash':True}),('serial-combined',{'packed':True,'direct_hash':True})]
                for n in (2,4):
                    row=micro.get('nn',{}).get('cpu'+str(n))
                    if row and row['forward_seconds']['1']<micro['nn']['cpu1']['forward_seconds']['1']*.95:
                        options.append((f'serial-threads{n}',{'threads':n}))
                        state['decisions'].append(f'CPU{n} threads batch1 micro improved>5%; include full4-episode screen')
                    else:state['decisions'].append(f'NoCPU{n}-thread serial screen: no>5% batch1 micro gain')
                for n in (2,4):
                    if n>quota:state['decisions'].append(f'skip{n} workers: exceeds effective CPU ceiling');continue
                    options += [(f'cpu{n}',{'workers':n}),
                                (f'cpu{n}-persistent',{'workers':n,'persistent_cache':True}),
                                (f'cpu{n}-combined',{'workers':n,'persistent_cache':True,'packed':True,'direct_hash':True,'completion_order':True})]
                best_threads=min(micro.get('nn',{'cpu1':{}}),key=lambda k:micro['nn'][k]['sgd_step_seconds'])
                if best_threads!='cpu1':options.append(('cpu4-trainthreads',{'workers':min(4,int(quota)),'persistent_cache':True,'packed':True,'direct_hash':True,'threads':micro['nn'][best_threads]['threads']}))
                if quota>=4:options.append(('cpu-shared-batch',{'workers':4,'persistent_cache':True,'packed':True,'direct_hash':True,'rpc_device':'cpu'}))
                if gpu.get('correctness',{}).get('cuda',{}).get('pass'):
                    options.append(('gpu-serial',{'serial_gpu':True,'gpu_train':True}))
                    if quota>=4:
                        best_gpu_cpu=min((k for k in gpu['nn'] if k.startswith('cpu')),key=lambda k:gpu['nn'][k]['sgd_step_seconds'])
                        train_gpu=gpu['nn']['cuda']['sgd_step_seconds']<gpu['nn'][best_gpu_cpu]['sgd_step_seconds']*.95
                        for wait in (.0002,.001):
                            options.append((f'gpu-shared-{wait}',{'workers':4,'persistent_cache':True,'packed':True,'direct_hash':True,'completion_order':True,'rpc_device':'cuda','rpc_batch_wait':wait,'gpu_train':train_gpu,'threads':1 if train_gpu else gpu['nn'][best_gpu_cpu]['threads']}))
                else:state['decisions'].append('CUDA numeric/availability gate did not pass; GPU full-loop candidates excluded')
                state['decisions'].append('8/16 actors not run: original logical cohort is4, synchronous training barrier permits at most4 active episodes. More processes only idle; no cohort expansion/training delay allowed.')
                state['decisions'].append('Object-plane cache not retried: three prior full-loop slowdowns; no new mechanism to overcome key/copy/miss costs.')
                screen_end=min(time.time()+1200,stamp(state['deadline_utc'])-1740)
                screens=[]
                # Same seed, same N0, same four complete episodes; order fixed before seeing results.
                ref=trial(root,state,'screen-reference',{},60,4801,held,1)
                screens.append(ref)
                for label,opt in options:
                    if not allowed(1740) or time.time()+75>screen_end:
                        state['decisions'].append('screening stopped to reserve required600s and paired300s repeats');break
                    python=gpu_python if opt.get('rpc_device')=='cuda' or opt.get('serial_gpu') else sys.executable
                    candidate=trial(root,state,'screen-'+label,opt,60,4801,held,1,python)
                    screens.append(candidate)
                    if label=='cpu4' and candidate.get('result',{}).get('fixed_work_complete'):
                        two=next((x for x in screens if x['label']=='screen-cpu2'),None)
                        if two and two.get('result',{}).get('fixed_work_complete') and candidate['result']['collection_seconds']>two['result']['collection_seconds']*1.2:
                            state['decisions'].append('plain4 workers regressed>20% versus2; no higher actor count tested; cache/IO variants retained as distinct hypotheses')
                ranked=[]
                for row in screens:
                    x=row.get('result',{})
                    if row['status']!='finished' or not x.get('pass') or not x.get('fixed_work_complete'):continue
                    opts=row['trial']['options'];key='cuda' if opts.get('gpu_train') else 'cpu'+str(opts.get('threads',1))
                    nn=(gpu if row['python']==gpu_python else micro).get('nn',{}).get(key)
                    if not nn:continue
                    io=micro['serialization']['packed' if opts.get('packed') else 'original']['median']
                    # This is a scheduling prediction, NOT a measured full-loop speedup.
                    score=x['collection_seconds']/x['qualified_positions']+.125*(nn['sgd_step_seconds']+io)
                    ranked.append((score,row))
                if not ranked:raise RuntimeError('NO_COMPLETE_CORRECT_SHORT_CANDIDATE')
                ranked.sort(key=lambda x:x[0]);chosen=ranked[0][1]
                # If original wins, still run one actual optimization candidate for the required comparison.
                if chosen['label']=='screen-reference' and len(ranked)>1:chosen=ranked[1][1]
                state['selection']={'label':chosen['label'],'options':chosen['trial']['options'],'python':chosen['python'],
                                    'basis':'fixed4-episode time/qualified position plus measured one-stepSGD+checkpoint cost/8 positions; prediction only, not speedup; full600s decides',
                                    'ranking':[{'label':r['label'],'predicted_seconds_per_position':s} for s,r in ranked]}
                state['screened']=True;dump(root/'state.json',state)
            selected=state['selection']
            gate=next((r for r in state['attempts'] if r['label']=='selected-learning-gate' and r.get('result',{}).get('learning_gate_pass')),None)
            if gate is None:
                if not allowed(980):raise RuntimeError('INSUFFICIENT_BUDGET_FOR_PRECONDITION_AND600S')
                gate=trial(root,state,'selected-learning-gate',selected['options'],300,4803,held,python=selected['python'],learning_gate=True)
                if not gate.get('result',{}).get('learning_gate_pass'):raise RuntimeError('SELECTED_CANDIDATE_FAILED_REAL_TRAINING_RECOVERY_OR_MODEL_CONSUMPTION_GATE')
            if not good(state,'candidate600') and allowed(650):trial(root,state,'candidate600',selected['options'],600,4801,held,python=selected['python'])
            if not good(state,'repeat-candidate300') and allowed(650):trial(root,state,'repeat-candidate300',selected['options'],300,4802,held,python=selected['python'])
            if not good(state,'repeat-baseline300') and allowed(340):trial(root,state,'repeat-baseline300',{},300,4802,held)
            state['status']='stopped' if (root/'STOP').exists() else ('finished' if good(state,'candidate600') else 'incomplete')
        except Exception as exc:
            state['status']='stopped' if (root/'STOP').exists() else 'incomplete';state['error']=repr(exc)
            event(root/'events.jsonl',event='controller_error',error=repr(exc))
        finally:
            state['active']=None;state['finished_utc']=utc();dump(root/'state.json',state)
            export(root,state)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','start','resume','status','stop','export','_controller']);p.add_argument('name',nargs='?');p.add_argument('--budget',type=int,default=3600);a=p.parse_args()
    HOME.mkdir(parents=True,exist_ok=True)
    if a.command=='check':
        import torch
        from perf.common import configure
        from tasks.tower10.p45.environment import Environment
        from tasks.tower10.p45.network import Network
        configure();count=verify_source();env=Environment()
        with torch.inference_mode():logits,v=Network().eval()(torch.from_numpy(env.observation.encode(env.initial,0)[None]))
        if logits.shape!=(1,713) or not torch.isfinite(logits).all() or not torch.isfinite(v).all():raise ValueError('NETWORK_SELF_CHECK_FAILED')
        e=environment();e['torch']=str(torch.__version__);e['actual_check_device']='cpu';dump(HOME/'environment-check.json',e)
        print(f'检查通过：{count}个P45文件未变；CPU配额={e["quota_cores"]}核，可见={e["visible_cpus"]}，亲和={len(e["affinity"])}；GPU环境={e["gpu_env"] or "不可用/未安装"}');return
    if not a.name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.name):p.error('需要简单实验名称')
    root=HOME/'experiments'/a.name
    if a.command=='start':
        if not 0<a.budget<=3600:p.error('预算必须为1..3600秒；正式协议需要3600秒')
        root.mkdir(parents=True,exist_ok=False)
        state={'name':a.name,'status':'prepared','code_commit':revision(),'baseline_commit':BASELINE,'budget_seconds':a.budget,
               'started_utc':utc(),'deadline_utc':dt.datetime.fromtimestamp(time.time()+a.budget,dt.timezone.utc).isoformat(),'attempts':[],'decisions':[],'active':None,'numeric_tolerances':TOLERANCES,
               'installation':load(HOME/'installation.json',{})}
        dump(root/'state.json',state)
        dump(root/'environment.json',environment())
        gpu_log=HOME/'gpu-install.log'
        if gpu_log.exists():
            text=gpu_log.read_text(errors='replace')[-64000:]
            text=re.sub(r'(https?://)[^/\s@]+@',r'\1<redacted>@',text)
            text=re.sub(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}','<redacted-token>',text)
            (root/'gpu-install-tail.log').write_text(text)
    elif not (root/'state.json').exists():raise RuntimeError('EXPERIMENT_NOT_FOUND')
    if a.command in ('start','resume'):
        if alive(root):raise RuntimeError('EXPERIMENT_ALREADY_RUNNING')
        state=load(root/'state.json')
        if a.command=='resume' and state['deadline_utc'] and time.time()>=stamp(state['deadline_utc']):raise RuntimeError('GLOBAL_BUDGET_EXHAUSTED_NEVER_RESET')
        if a.command=='resume':(root/'STOP').unlink(missing_ok=True)
        with (root/'controller.log').open('ab') as f:
            child=subprocess.Popen([sys.executable,'-m','perf.control','_controller',a.name],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True,env=child_env())
        for _ in range(100):
            if child.poll() is not None:raise RuntimeError('启动失败，见 '+str(root/'controller.log'))
            state=load(root/'state.json')
            if state.get('controller_pid')==child.pid and alive(root):print('后台实验已启动；总截止时间='+state['deadline_utc']+'；可关闭终端。');return
            time.sleep(.05)
        raise RuntimeError('启动确认超时，先查看日志')
    if a.command=='_controller':
        controller(root);return
    state=load(root/'state.json')
    if a.command=='stop':
        (root/'STOP').touch()
        if state.get('active') is not None:
            run=state['attempts'][state['active']].get('trial',{}).get('run')
            if run:(Path(run)/'STOP').touch()
        print('已请求安全停止；现有结果保留，总预算不会重置。');return
    if a.command=='export':print('结果包：'+str(export(root,state)));return
    running=alive(root);remaining=max(0,stamp(state['deadline_utc'])-time.time()) if state['deadline_utc'] else state['budget_seconds']
    label=state['attempts'][state['active']]['label'] if state.get('active') is not None else '无'
    print(f'状态：{state["status"] if running or state["status"]!="running" else "控制进程异常退出"}；当前：{label}；剩余总预算：{remaining:.0f}秒')
    if state.get('active') is not None:
        active=state['attempts'][state['active']];s=load(Path(active.get('trial',{}).get('run','/nonexistent'))/'status.json',{})
        if s:print(f'本项：提交局={len(s.get("episodes",[]))}，合格位置={s.get("total_positions",0)}，更新={s.get("generation",0)}，SGD={s.get("sgd_steps",0)}')
    print(f'已记录{len(state["attempts"])}项；结果包：{root/"results.zip"}；错误日志：{root/"controller.log"}')
    packaging=load(root/'export.json')
    if packaging:print(f'含打包总耗时：{packaging["total_controller_seconds_including_pack"]:.1f}秒；超出截止：{packaging["late_past_global_deadline_seconds"]:.3f}秒')
    if state.get('error'):print('原因：'+state['error'])

if __name__=='__main__':
    try:main()
    except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
