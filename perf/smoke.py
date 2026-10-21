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
from perf.smoke_export import export
from perf.smoke_inputs import valid,valid_reference
from perf.control import HOME,SOURCE,revision,lock,alive,environment,child_env,task

# Actual dependencies, not an artificial fixed sequence. Independent paths have[].
PLAN=[('reserve_inputs',30,[]),('fixture',50,[]),('serialization',30,['fixture']),
      ('micro',25,['fixture']),('cpu_parallel',45,[]),('service',20,[]),
      ('gpu_micro',40,['fixture','micro']),('gpu_service',25,[]),
      ('learning',80,['fixture']),('orphan_parent',6,[]),('shutdown_timeout',12,[]),
      ('worker_failure',15,[]),('controlled_failure',8,[]),('controlled_timeout',5,[]),
      ('fallback_serialization',20,['reserve_inputs']),('real_candidate_worker',25,['micro'])]

def coverage(root,state):
    rows=list({x['path']:x for x in state.get('coverage',[])}.values())
    dump(root/'coverage.json',{'smoke_only':True,'not_performance_comparison':True,'paths':rows})
    import csv
    with (root/'coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','status','chain_status','link_status','input_source','seconds','exit_code','error_path','detail']);w.writeheader()
        for x in rows:w.writerow({k:x.get(k) for k in w.fieldnames})
    names={'pass':'通过','fail':'失败','timeout':'超时','not_run':'未执行','device_unavailable':'设备不可用'}
    text=['P48 600秒流程冒烟：不证明提速，不计正式产能。']
    text += [x['path']+'：自身='+names[x['status']]+'；衔接='+x.get('link_status','not_applicable')+'；整条依赖链='+x.get('chain_status','not_applicable')+'；输入='+x.get('input_source','independent')+'；'+x.get('detail','') for x in rows]
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
    state['scheduling_logic_selection']={'label':selection['label'],'smoke_control_metadata_only':True}
    return {'status':'pass','detail':'调度逻辑检查：正式选择/调度函数，print子进程占位；不代表真实worker或完整闭环','sequence':sequence}

def inputs_for(root,finished,forced=False):
    primary=valid(root)
    if not forced and finished.get('fixture')=='pass' and primary:
        return root,primary,'upstream_real','pass','validated actual fixture output'
    backup=valid(root/'reserve')
    reason='controlled missing-upstream injection' if forced else 'fixture failed/missing or input hash invalid'
    link='fail' if forced or finished.get('fixture') in ('fail','timeout','pass') else 'not_verified'
    if backup:return root/'reserve',backup,'validated_fallback',link,reason
    return None,None,'missing',link,reason+'; no validated backup checkpoint/episode/tree'

def gpu_inputs(root,finished):
    selected,inputs,source,link,detail=inputs_for(root,finished)
    reference=valid_reference(root/'cases/micro',inputs) if selected and finished.get('micro')=='pass' else None
    if reference:return selected,reference,source,link,detail
    backup=valid(root/'reserve');ref=valid_reference(root/'reserve/cases/micro',backup) if backup else None
    if ref:return root/'reserve',ref,'validated_fallback_with_matching_CPU_reference','fail' if finished.get('micro') in ('fail','timeout') else 'not_verified','primary CPU reference missing/invalid; matching input+reference bundle substituted'
    return None,None,'missing','not_verified','missing CPU golden bound to same model and encoded samples; no validated matching backup'

def real_worker(root,state,held,cap,finished):
    from perf.gates import choose
    micro=load(root/'cases/micro/micro.json',{})
    source='upstream_micro';link='pass'
    if finished.get('micro')!='pass' or micro.get('hash_correctness',{}).get('status')!='pass':
        micro=load(root/'reserve/cases/micro/micro.json',{});source='validated_fallback_micro';link='fail' if finished.get('micro') in ('fail','timeout') else 'not_verified'
    if micro.get('hash_correctness',{}).get('status')=='pass':
        candidates=[(sum(micro['hash']['copy_sha256']),{'label':'screen-reference','options':{}}),
                    (sum(micro['hash']['memoryview_sha256']),{'label':'screen-hash','options':{'direct_hash':True}})]
        basis='actual hash micro gate/timings; exercise candidate entry, no performance recommendation'
    else:
        candidates=[(0.,{'label':'screen-reference','options':{}})]
        source='fresh_frozen_reference_no_valid_candidate';link='not_verified'
        basis='no valid optimization gate: independently exercise frozen reference worker with fresh random network; candidate chain NOT verified'
    chosen=choose(candidates);state['selection']={'label':chosen['label'],'options':chosen['options'],'basis':basis}
    path=root/'real-candidate-run';path.mkdir();dump(path/'options.json',chosen['options'])
    argv=['-B','-m','perf.worker','--root',str(path),'--options',str(path/'options.json'),'--seconds',str(max(1,cap-5)),'--deadline',str(stamp(state['deadline_utc'])-35),'--seed','4812','--until-decisions','1']
    rec=task(root,state,'real_candidate_worker',argv,sys.executable,cap,held,{'run':str(path),'options':chosen['options'],'requested_seconds':max(1,cap-5),'seed':4812,'smoke_until_decisions':1})
    data=rec.get('result',{});ok=rec['status']=='finished' and data.get('pass') and data.get('decisions',0)>=1 and data.get('simulations',0)>=100
    summary={'status':'pass' if ok else 'fail','input_source':source,'link_status':link,'official_worker_argv':argv,'options':chosen['options'],'real_computation':True,'formal_full_loop':False,'exit_code':rec.get('exit_code'),'result_path':str(path/'result.json'),'software':data.get('software'),'metrics':{k:data.get(k) for k in ('decisions','simulations','model_updates','sgd_steps','qualified_positions','full_window_complete','pass')}}
    dump(root/'real-worker-summary.json',summary)
    return {'path':'real_candidate_worker','status':summary['status'],'link_status':link,'input_source':source,'seconds':rec['wall_seconds'],'exit_code':rec.get('exit_code'),'error_path':rec['directory']+'/console.log','detail':'actual selected options -> official worker -> search/audit -> parsed result/summary; not a full learning-loop comparison'}

def controller(root):
    with lock(root) as held:
        state=load(root/'state.json')
        if revision()!=state['code_commit']:raise RuntimeError('SMOKE_CODE_COMMIT_CHANGED')
        if state.get('finished_utc'):
            state.setdefault('prior_execution_endings',[]).append({'finished_utc':state.pop('finished_utc'),'elapsed':state.pop('execution_elapsed_seconds',None)})
        state['controller_pid']=os.getpid();state['status']='running';dump(root/'state.json',state)
        for rec in state['attempts']:
            if rec['status']=='running':
                rec['status']='incomplete';state['coverage'].append({'path':rec['label'],'status':'not_run','link_status':'not_verified','seconds':0,'detail':'interrupted attempt retained; no automatic repeat'})
        deadline=stamp(state['deadline_utc']);finished={r['path']:r['status'] for r in state['coverage']}
        dump(root/'plan.json',{'stages':[{'name':n,'cap_seconds':c,'normal_dependencies':d} for n,c,d in PLAN],'fallback_policy':'validate fresh backup before primary stage; downstream own-check and link-check reported separately','total_stage_caps':sum(c for n,c,d in PLAN),'final_reserve_seconds':35})
        try:
            for index,(name,cap,deps) in enumerate(PLAN):
                if name in finished:continue
                out=root/'cases'/name;source='independent';link='not_applicable';detail='independent stage'
                row={'path':name,'status':'not_run','seconds':0,'link_status':link,'input_source':source,'detail':detail}
                try:
                    remaining=deadline-time.time()
                    # Each later stage gets a launch opportunity; never let one
                    # early case consume the entire diagnostic budget.
                    cap=min(cap,max(0,remaining-35-2*(len(PLAN)-index-1)))
                    if (root/'STOP').exists() or cap<1:
                        row['detail']='stopped or final packaging reserve; no new computation'
                    elif name=='real_candidate_worker':row=real_worker(root,state,held,cap,finished)
                    else:
                        selected=root;reference=None
                        if name in ('serialization','micro','learning','fallback_serialization','gpu_micro'):
                            selected,inputs,source,link,detail=inputs_for(root,finished,forced=name=='fallback_serialization')
                            if name=='gpu_micro':selected,reference,source,link,detail=gpu_inputs(root,finished)
                        row.update(link_status=link,input_source=source,detail=detail)
                        if selected is None:row['detail']=detail
                        elif name in ('gpu_micro','gpu_service') and not state.get('installation',{}).get('gpu_python'):
                            row.update(status='device_unavailable',link_status='not_verified' if link=='pass' else link,detail='no verified CUDA interpreter; input hashes may validate but GPU consumption/transition is not verified')
                        else:
                            python=state['installation']['gpu_python'] if name in ('gpu_micro','gpu_service') else sys.executable
                            end=min(deadline-35,time.time()+cap-2)
                            case='serialization' if name=='fallback_serialization' else name
                            argv=['-B','-m','perf.smoke_cases',case,'--root',str(root),'--out',str(out),'--deadline',str(end)]
                            if selected!=root:argv+=['--input-root',str(selected)]
                            if reference:argv+=['--reference',str(reference)]
                            if name=='fixture' and state.get('inject_fixture_failure'):argv+=['--inject-fixture-failure']
                            rec=task(root,state,name,argv,python,cap,held)
                            result=load(out/'case.json',{})
                            status=result.get('status','timeout' if rec['status']!='finished' and rec.get('exit_code',0)<0 else 'fail')
                            if rec['status']!='finished' and status=='pass':status='fail'
                            if name=='orphan_parent':
                                ok=rec.get('exit_code')==17 and rec['cleanup']['before_cleanup'] and rec['cleanup']['complete']
                                status='pass' if ok else 'fail';detail='controlled parent exit17; detached descendant identified/reaped; primary exit retained in attempt'
                            if status!='pass' and link=='pass':row['link_status']='not_verified'
                            row.update(status=status,seconds=rec['wall_seconds'],exit_code=rec.get('exit_code'),error_path=rec['directory']+'/console.log',detail=result.get('error',detail),cleanup_complete=rec.get('cleanup',{}).get('complete'))
                            if name=='controlled_failure':row['expected_fault_observed']=status=='fail' and 'EXPECTED_INJECTED' in result.get('error','')
                            if name=='controlled_timeout':row['expected_timeout_observed']=status=='timeout' and (out/'timeout-entered.json').exists()
                except Exception as exc:
                    import traceback
                    row.update(status='fail',detail=repr(exc),traceback=traceback.format_exc())
                parents=[x for x in state['coverage'] if x['path'] in deps]
                if row['status']=='fail' or row.get('link_status')=='fail' or any(x.get('chain_status')=='fail' for x in parents):row['chain_status']='fail'
                elif row['status']!='pass' or row.get('link_status')=='not_verified' or any(x.get('chain_status')=='not_verified' for x in parents):row['chain_status']='not_verified'
                else:row['chain_status']='pass' if deps else 'not_applicable'
                state['coverage'].append(row);finished[name]=row['status'];coverage(root,state);dump(root/'state.json',state)
            if not (root/'STOP').exists() and time.time()<deadline-25:
                try:
                    check=schedule_check(root,state,held);dump(root/'scheduler.json',check)
                    state['coverage'].append({'path':'scheduling_logic','status':'pass','link_status':'simulation_only','input_source':'controlled metadata + print children','seconds':0,'detail':check['detail']})
                except Exception as exc:state['coverage'].append({'path':'scheduling_logic','status':'fail','link_status':'simulation_only','seconds':0,'detail':repr(exc)})
            fail=next((x for x in state['coverage'] if x['path']=='controlled_failure'),{})
            timeout=next((x for x in state['coverage'] if x['path']=='controlled_timeout'),{})
            ok=fail.get('expected_fault_observed') and timeout.get('expected_timeout_observed') and finished.get('real_candidate_worker')=='pass'
            state['coverage'].append({'path':'failure_isolation','status':'pass' if ok else 'not_run','link_status':'not_applicable','input_source':'actual controlled faults then real worker','seconds':0,'detail':'injected failure + timeout retained; later official worker computed real search and summary'})
            state['status']='stopped' if (root/'STOP').exists() else 'finished'
        except Exception as exc:
            import traceback
            state['status']='incomplete';state['error']=repr(exc);(root/'controller-error.txt').write_text(traceback.format_exc())
        finally:
            for name,cap,deps in PLAN:
                if not any(x['path']==name for x in state['coverage']):state['coverage'].append({'path':name,'status':'not_run','link_status':'not_verified','seconds':0,'detail':'controller interrupted before dispatch'})
            state['active']=None;state['work_finished_utc']=utc();coverage(root,state);dump(root/'state.json',state)
            export(root,state,finalizing=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','status','stop','resume','export','_controller']);p.add_argument('name');p.add_argument('--budget',type=int,default=600);p.add_argument('--inject-fixture-failure',action='store_true');p.add_argument('--full',action='store_true');a=p.parse_args()
    if a.full and a.command!='export':p.error('--full only applies to explicit export')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.name):p.error('invalid name')
    root=HOME/'smoke-tests'/a.name
    if a.command=='start':
        if not 0<a.budget<=600:p.error('smoke budget must be1..600')
        root.mkdir(parents=True,exist_ok=False)
        state={'name':a.name,'smoke_only':True,'code_commit':revision(),'budget_seconds':a.budget,'started_utc':utc(),'deadline_utc':dt.datetime.fromtimestamp(time.time()+a.budget,dt.timezone.utc).isoformat(),'inject_fixture_failure':a.inject_fixture_failure,'status':'prepared','attempts':[],'coverage':[],'active':None,'installation':load(HOME/'installation.json',{}),'decisions':['No formal productivity/learning claims. Synthetic threshold fixtures identified separately. Never reads cloud600 or probe1.']}
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
    if a.command=='export':
        if a.full:
            from perf.smoke_export import export_full
            print('完整资料（可能包含测试断点/权重）：'+str(export_full(root)))
        else:coverage(root,state);print(export(root,state))
        return
    label=state['attempts'][state['active']]['label'] if state.get('active') is not None else '无'
    print('流程冒烟：'+state['status']+'；当前='+label+'；剩余='+str(round(max(0,stamp(state['deadline_utc'])-time.time())))+'秒；进程锁='+str(alive(root)))
    elapsed=state.get('execution_elapsed_seconds',time.time()-stamp(state['started_utc']))
    print('实际执行耗时='+str(round(elapsed,2))+'秒；结束时间='+str(state.get('finished_utc','尚未结束'))+'；绝对预算截止='+state['deadline_utc'])
    for row in {r['path']:r for r in state['coverage']}.values():print(row['path']+': 自身='+row['status']+'；衔接='+row.get('link_status','not_applicable')+'；整条依赖链='+row.get('chain_status','not_applicable')+'；输入='+row.get('input_source','independent'))
    print('结果包：'+str(root/'results.zip'))
if __name__=='__main__':main()
