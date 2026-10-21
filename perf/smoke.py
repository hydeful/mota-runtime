"""600-second path smoke test; no throughput claims, old experiments never read."""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile
from perf.common import dump,load,utc,stamp
from perf.control import HOME,SOURCE,revision,lock,alive,environment,child_env,task,export

PLAN=[('fixture',65,[]),('serialization',65,['fixture']),('micro',35,['fixture']),
      ('cpu_parallel',65,[]),('service',25,[]),('gpu_micro',80,['fixture','micro']),
      ('learning',125,['fixture']),('controlled_failure',3,[]),('controlled_timeout',2,[])]

def coverage(root,state):
    rows=state.get('coverage',[])
    dump(root/'coverage.json',{'smoke_only':True,'not_performance_comparison':True,'paths':rows})
    import csv
    with (root/'coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','status','seconds','detail']);w.writeheader()
        for x in rows:w.writerow({k:x.get(k) for k in w.fieldnames})
    names={'pass':'通过','fail':'失败','timeout':'超时','not_run':'未执行','device_unavailable':'设备不可用'}
    text=['P48 600秒流程冒烟：不证明提速，不计正式产能。']
    text += [x['path']+'：'+names[x['status']]+'；'+x.get('detail','') for x in rows]
    (root/'COVERAGE.txt').write_text('\n'.join(text)+'\n')

def schedule_check(root,state,held):
    from perf.gates import blocked,choose,dispatch_comparisons
    # Production helpers, controlled metadata only; no invented timing results.
    micro={'json_correctness':{'status':'pass'},'hash_correctness':{'status':'pass'},'serialization':{'original':{'status':'pass'},'packed':{'status':'fail'}}}
    assert blocked({'packed':True},micro)==['packed_restore'];assert not blocked({},micro)
    rows=[(.1,{'label':'screen-reference'}),(.2,{'label':'screen-cpu2'})]
    selection=choose(rows);assert selection['label']=='screen-cpu2'
    sequence=['screen-reference','screen-cpu2','selected-learning-gate'];completed={}
    def run(label,seconds,seed):
        rec=task(root,state,'dispatch-'+label,['-c','print("SMOKE_COMPRESSED_DISPATCH_ONLY")'],sys.executable,3,held)
        assert rec['status']=='finished'
        completed[label]=True;sequence.append(label)
    dispatch_comparisons(run,lambda label:completed.get(label,False),lambda requested:time.time()<stamp(state['deadline_utc'])-25)
    assert sequence[-3:]==['candidate600','repeat-candidate300','repeat-baseline300']
    # Dispatch real controller.task subprocesses under compressed work (not600s trials).
    state['selection']={'label':selection['label'],'smoke_control_metadata_only':True}
    return {'status':'pass','detail':'production dependency/selection/final scheduling functions; actual subprocess dispatch with compressed work, not formal performance','sequence':sequence}

def controller(root):
    with lock(root) as held:
        state=load(root/'state.json')
        if revision()!=state['code_commit']:raise RuntimeError('SMOKE_CODE_COMMIT_CHANGED')
        state['controller_pid']=os.getpid();state['status']='running';dump(root/'state.json',state)
        for rec in state['attempts']:
            if rec['status']=='running':
                rec['status']='incomplete';state['coverage'].append({'path':rec['label'],'status':'not_run','seconds':0,'detail':'interrupted attempt retained; no automatic repeat'})
        deadline=stamp(state['deadline_utc']);finished={r['path']:r['status'] for r in state['coverage']}
        try:
            for name,cap,deps in PLAN:
                if name in finished:continue # no automatic repeats, even failures/timeouts
                remaining=deadline-time.time();out=root/'cases'/name
                if (root/'STOP').exists() or remaining<cap+42:
                    row={'path':name,'status':'not_run','seconds':0,'detail':'stopped or reserved downstream/pack budget insufficient'}
                elif any(finished.get(x)!='pass' for x in deps):
                    row={'path':name,'status':'not_run','seconds':0,'detail':'blocked dependencies: '+','.join(x for x in deps if finished.get(x)!='pass')}
                else:
                    python=sys.executable
                    if name=='gpu_micro':
                        python=state.get('installation',{}).get('gpu_python')
                        if not python:
                            row={'path':name,'status':'device_unavailable','seconds':0,'detail':'no verified reusable CUDA environment; no installation during budget'}
                            state['coverage'].append(row);finished[name]=row['status'];coverage(root,state);dump(root/'state.json',state);continue
                    end=min(deadline-40,time.time()+cap-3)
                    argv=['-B','-m','perf.smoke_cases',name,'--root',str(root),'--out',str(out),'--deadline',str(end)]
                    rec=task(root,state,name,argv,python,cap,held)
                    result=load(out/'case.json',{})
                    status=result.get('status','timeout' if rec['status']!='finished' and rec.get('exit_code',0)<0 else 'fail')
                    row={'path':name,'status':status,'seconds':rec['wall_seconds'],'detail':result.get('error',result.get('reason','see cases/'+name+'/case.json'))}
                    if name=='controlled_failure':
                        row['expected_fault_observed']=status=='fail' and 'EXPECTED_INJECTED' in result.get('error','')
                    if name=='controlled_timeout':row['expected_timeout_observed']=status=='timeout'
                state['coverage'].append(row);finished[name]=row['status'];coverage(root,state);dump(root/'state.json',state)
            # Always retain scheduler/exception/export evidence, independent of earlier numeric failures.
            if (root/'STOP').exists() or time.time()>deadline-30:
                state['coverage'].append({'path':'candidate_selection_schedule','status':'not_run','seconds':0,'detail':'stopped or final pack reserve; no new subprocesses'})
            else:
                check=schedule_check(root,state,held);dump(root/'scheduler.json',check)
                state['coverage'].append({'path':'candidate_selection_schedule','status':'pass','seconds':0,'detail':check['detail']})
            fail=next((x for x in state['coverage'] if x['path']=='controlled_failure'),{})
            timeout=next((x for x in state['coverage'] if x['path']=='controlled_timeout'),{})
            ok=fail.get('expected_fault_observed') and timeout.get('expected_timeout_observed') and any(x['path']=='candidate_selection_schedule' and x['status']=='pass' for x in state['coverage'])
            state['coverage'].append({'path':'failure_isolation','status':'pass' if ok else 'not_run','seconds':0,'detail':'expected failure then timeout then scheduler/export continued; injected faults are not candidate passes'})
            state['status']='stopped' if (root/'STOP').exists() else 'finished'
        except Exception as exc:
            import traceback
            state['coverage'].append({'path':'controller_or_schedule','status':'fail','seconds':0,'detail':repr(exc)})
            state['status']='incomplete';state['error']=repr(exc);(root/'controller-error.txt').write_text(traceback.format_exc())
        finally:
            for name,cap,deps in PLAN:
                if not any(x['path']==name for x in state['coverage']):state['coverage'].append({'path':name,'status':'not_run','seconds':0,'detail':'controller interrupted before dispatch'})
            state['active']=None;state['finished_utc']=utc();coverage(root,state);dump(root/'state.json',state)
            export(root,state)
            # Verify export contains evidence even when cases failed/timed out.
            with zipfile.ZipFile(root/'results.zip') as z:assert 'coverage.json' in z.namelist() and z.testzip() is None
            dump(root/'export-verification.json',{'status':'pass','zip_integrity':True,'coverage_included':True,'wall_including_first_pack':time.time()-stamp(state['started_utc'])})

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','status','stop','resume','export','_controller']);p.add_argument('name');p.add_argument('--budget',type=int,default=600);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.name):p.error('invalid name')
    root=HOME/'smoke-tests'/a.name
    if a.command=='start':
        if not 0<a.budget<=600:p.error('smoke budget must be1..600')
        root.mkdir(parents=True,exist_ok=False)
        state={'name':a.name,'smoke_only':True,'code_commit':revision(),'budget_seconds':a.budget,'started_utc':utc(),'deadline_utc':dt.datetime.fromtimestamp(time.time()+a.budget,dt.timezone.utc).isoformat(),'status':'prepared','attempts':[],'coverage':[],'active':None,'installation':load(HOME/'installation.json',{}),'decisions':['No formal productivity/learning claims. Synthetic threshold fixtures identified separately. Never reads cloud600 or probe1.']}
        dump(root/'state.json',state);dump(root/'environment.json',environment())
    elif not (root/'state.json').exists():raise RuntimeError('SMOKE_NOT_FOUND')
    if a.command in ('start','resume'):
        if alive(root):raise RuntimeError('SMOKE_ALREADY_RUNNING')
        state=load(root/'state.json')
        if a.command=='resume' and state['status']=='finished':raise RuntimeError('SMOKE_FINISHED_NO_AUTOMATIC_REPEAT')
        if time.time()>=stamp(state['deadline_utc']):raise RuntimeError('SMOKE_BUDGET_EXHAUSTED_NEVER_RESET')
        if a.command=='resume':(root/'STOP').unlink(missing_ok=True)
        with (root/'controller.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-B','-m','perf.smoke','_controller',a.name],stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=child_env(),start_new_session=True)
        for _ in range(100):
            if child.poll() is not None:raise RuntimeError('SMOKE_START_FAILED see '+str(root/'controller.log'))
            if load(root/'state.json').get('controller_pid')==child.pid and alive(root):print('600秒流程冒烟已后台启动；可关闭终端；截止='+state['deadline_utc']);return
            time.sleep(.05)
        raise RuntimeError('SMOKE_START_CONFIRM_TIMEOUT')
    if a.command=='_controller':controller(root);return
    state=load(root/'state.json')
    if a.command=='stop':(root/'STOP').touch();print('已请求停止；不会重置预算；结果将自动导出。');return
    if a.command=='export':coverage(root,state);print(export(root,state));return
    label=state['attempts'][state['active']]['label'] if state.get('active') is not None else '无'
    print('流程冒烟：'+state['status']+'；当前='+label+'；剩余='+str(round(max(0,stamp(state['deadline_utc'])-time.time())))+'秒；进程锁='+str(alive(root)))
    for row in state['coverage']:print(row['path']+': '+row['status'])
    print('结果包：'+str(root/'results.zip'))
if __name__=='__main__':main()
