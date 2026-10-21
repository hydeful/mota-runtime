"""Bounded diagnostic cases. Synthetic fixtures never enter production runs/rates."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import concurrent.futures
import threading
import numpy as np
import torch
from perf.common import configure,dump,load,budget,SOURCE,verify_source,stamp
from perf.semantic import compare,report
from perf.patches import packed_save,direct_hash
from tasks.tower10.p45 import runtime
from tasks.tower10.p45.environment import Environment
from tasks.tower10.p45.search import Search,Node
from tasks.tower10.p45.network import model_hash


def env(deadline):return Environment(config=budget(load(SOURCE/'tasks/tower10/p45/config.json'),deadline))
def restore_copy(source,target,checkpoint=None):
    shutil.copytree(source,target)
    (target/'STOP').unlink(missing_ok=True)
    if checkpoint is not None:shutil.copyfile(checkpoint,target/'checkpoint.pt')

def fixture(root,out,deadline):
    from perf.worker import execute
    global_deadline=stamp(load(root/'state.json')['deadline_utc']) if (root/'state.json').exists() else deadline
    r,data=execute(root/'fixture',{},max(1,deadline-time.time()-5),global_deadline,4811,cohorts=1)
    assert data['fixed_work_complete'],'FOUR_REAL_EPISODES_NOT_FINISHED'
    # Freeze a second fixture in the middle of a 100-simulation decision.
    r.collect(time.monotonic()+max(1,deadline-time.time()-3),no_train=True,until_simulations=r.s['total_simulations']+7)
    assert r.s['actor'] and r.s['actor']['decision'] and 0<r.s['actor']['decision']['completed']<100
    value=torch.load(r.root/'checkpoint.pt',weights_only=False)
    runtime.atomic_torch(root/'pending.pt',value)
    dump(root/'fixture.json',{'source':'Fresh seed4811 original P45, four actual complete episodes plus partial search; smoke-only, never formal productivity','result':data,'pending_simulations':r.s['actor']['decision']['completed']})
    from perf.smoke_inputs import seal
    r.validate_pending_actor()
    seal(root,{'kind':'upstream_real_output','seed':4811,'four_real_complete_episodes':True,'formal_productivity':False})
    return {'real_complete_episodes':data['complete_episodes'],'real_qualified_positions':data['qualified_positions'],'partial_tree_and_actor_generators':True}

def reserve_inputs(root,out,deadline):
    from tasks.tower10.p45.parallel import OneEpisodeRunner,EpisodeComplete
    from tasks.tower10.p45.contracts import validate_episode
    from perf.smoke_inputs import seal,seal_reference
    reserve=root/'reserve';reserve.mkdir()
    global_deadline=stamp(load(root/'state.json')['deadline_utc'])
    e=env(global_deadline);r=OneEpisodeRunner(reserve/'fixture',e,create=True,seed=4811,run_id='P48-smoke-reserve')
    try:r.collect(time.monotonic()+max(1,deadline-time.time()-8),no_train=True)
    except EpisodeComplete:pass
    assert len(r.s['cohort'])==1,'NO_VALID_BACKUP_EPISODE'
    ep=r.load_episode(r.s['cohort'][0]['episode_id']);validate_episode(e,ep,reference=True)
    # Explicit diagnostic one-episode ledger; never a formal four-episode commit.
    r.s['episodes']=r.s['cohort'];r.s['cohort']=[]
    n=len(ep['records']);r.s['total_positions']=n;r.s['new_positions']=n
    r.s['replay']=[(ep['episode_id'],i) for i in range(n)];r.save()
    r.collect(time.monotonic()+max(1,deadline-time.time()-7),no_train=True,until_simulations=r.s['total_simulations']+7)
    assert r.s['actor'] and r.s['actor']['decision']['completed']==7
    r.validate_pending_actor();saved=torch.load(r.root/'checkpoint.pt',weights_only=False)
    compare(saved,copy.deepcopy(saved));runtime.atomic_torch(reserve/'pending.pt',saved)
    inputs=seal(reserve,{'kind':'fresh_fallback_test_input','seed':4811,'source':'one real original-search episode, independently reference-replayed before any downstream use','synthetic_one_episode_commit':True,'formal_productivity':False})
    from perf.micro import run
    reference=run(reserve/'fixture',reserve/'cases/micro')
    assert reference['correctness']['cpu1']['pass']
    seal_reference(reserve/'cases/micro',inputs)
    return {'input_manifest':inputs,'CPU_reference_validated':True,'NOT_A_PRODUCTION_LEDGER':True}

def legacy_first(a,b,path='$'):
    if isinstance(a,dict):
        for k in sorted(a,key=repr):
            p=legacy_first(a[k],b[k],path+'['+repr(k)+']')
            if p:return p
    elif isinstance(a,(tuple,list)):
        for i,(x,y) in enumerate(zip(a,b)):
            p=legacy_first(x,y,path+'['+str(i)+']')
            if p:return p
    elif not isinstance(a,(torch.Tensor,np.ndarray)) and repr(a)!=repr(b):return path+' (legacy repr differs for '+type(a).__name__+')'

def serialization(root,out,deadline):
    source=root/'fixture';saved=torch.load(root/'pending.pt',weights_only=False);results={}
    for name,writer in [('original',runtime.atomic_torch),('packed',packed_save)]:
        try:
            path=out/(name+'.pt');writer(path,saved);restored=torch.load(path,weights_only=False)
            results[name]={'legacy_hash_equal':runtime.structure_hash(saved)==runtime.structure_hash(restored),'legacy_first_difference':legacy_first(saved,restored),'comparison':report(saved,restored)}
            compare(saved,restored)
            original_rng=copy.deepcopy(saved['state']['actor']['noise_rng'].bit_generator.state)
            compare(saved,restored);compare(original_rng,saved['state']['actor']['noise_rng'].bit_generator.state)
            bad=copy.deepcopy(restored);bad['state']['actor']['decision']['completed']+=1
            negative=report(saved,bad);assert negative['status']=='fail';results[name]['negative_control']=negative
            bad=copy.deepcopy(restored);bad['state']['actor']['noise_rng'].bit_generator.state=np.random.default_rng(99).bit_generator.state
            assert report(saved,bad)['status']=='fail'
            # Both resume branches consume identical restored RNG and next search.
            fingerprints=[]
            for branch,ckpt in [('control',root/'pending.pt'),('roundtrip',path)]:
                target=out/(name+'-'+branch);restore_copy(source,target,ckpt)
                e=Environment(config=saved['state']['full_config']);r=runtime.Runner(target,e,create=False)
                r.collect(time.monotonic()+max(1,deadline-time.time()-2),no_train=True,until_decisions=r.s['total_decisions']+1)
                assert r.s['total_decisions']>saved['state']['total_decisions']
                fingerprints.append(r.fingerprint())
            compare(*fingerprints);results[name]['next_100_simulation_decision_equal']=True
            results[name]['status']='pass'
        except Exception as exc:results[name]=dict(results.get(name,{}),status='fail',error=str(exc),traceback=traceback.format_exc())
        dump(out/'details.json',results)
    # Readonly arrays explicitly covered, independently from actual fixture layout.
    flags={}
    for name,writer in [('original',runtime.atomic_torch),('packed',packed_save)]:
        a=np.arange(8);a.flags.writeable=False;x={'state':{'readonly':a}}
        p=out/'readonly.pt';writer(p,x);flags[name]=report(x,torch.load(p,weights_only=False))
    # Original Torch serialization loses this synthetic readonly flag. Report it
    # as a detected negative, never silently normalize the compared objects.
    assert flags['original']['status']=='fail' and 'writeable' in flags['original']['first_difference']
    assert flags['packed']['status']=='pass'
    dump(out/'readonly-negative.json',{'synthetic_readonly_control':flags,'actual_pending_checkpoint_results':results})
    assert all(x['status']=='pass' for x in results.values()),'SERIALIZATION_SUBITEM_FAILED_SEE_DETAILS'
    return results

def cpu_parallel(root,out,deadline):
    from perf.worker import execute
    r,data=execute(out/'run',{'workers':2,'persistent_cache':True},max(1,deadline-time.time()-5),deadline,4811,cohorts=1)
    assert data['fixed_work_complete'] and data['pass']
    profiles=data['parallel_profiles'];pids={x['worker_pid'] for x in profiles}
    assert len(pids)>=2,'TWO_WORKERS_NOT_OBSERVED'
    assert len(r.s['episodes'])==4 and not r.s['cohort']
    # Single coordinator's shared training loop, on the actual cohort, one real step.
    # Injected threshold accounting is diagnostic only, explicitly separate from production.
    assert r.s['recipe']['min_new_positions']==512
    r.s['new_positions']=512;r.begin_training();r.train(time.monotonic()+max(1,deadline-time.time()-2),steps_limit=1)
    assert r.s['sgd_steps']==1
    return {'actor_pids':sorted(pids),'single_shared_coordinator':os.getpid(),'actual_cohort_positions':data['qualified_positions'],'real_sgd_steps':1,'synthetic_trigger_counter':512,'formal_threshold_unchanged':True,'NOT_A_PRODUCTION_LEDGER':True}

def service(root,out,deadline,device='cpu'):
    from perf.devices import Service,RemoteForward,DeviceNetwork
    e=env(deadline)
    if device=='cuda' and not torch.cuda.is_available():return {'status':'device_unavailable','reason':'CUDA not usable in selected interpreter'}
    net=DeviceNetwork() if device=='cuda' else runtime.Network();net.eval()
    obs=e.observation.encode(e.initial,0)
    svc=Service(device,batch_wait=.02)
    try:
        svc.publish(net);h=model_hash(net)
        barrier=threading.Barrier(4)
        def call(i,batched=False):
            remote=RemoteForward(svc.address,svc.auth,h,e.schema_hash)
            # Different observations force a real multi-row kernel (not just dedup).
            x=obs.copy();x.flat[i%4]+=np.float32(i%4)*1e-5
            try:
                if batched:barrier.wait(timeout=10)
                return remote(torch.from_numpy(x[None]))
            finally:remote.close()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:values=list(pool.map(lambda i:call(i,True),range(4)))
        for _ in range(2):call(0)
        # Actual SGD in selected device, full batch64, original loss/optimizer.
        net.train();opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=0)
        x=torch.from_numpy(np.repeat(obs[None],64,0));mask=torch.from_numpy(np.repeat(e.mask(e.initial)[None],64,0))
        logits,v=net(x);pi=mask.float()/mask.sum(1,keepdim=True)
        from tasks.tower10.p45.network import manual_l2
        loss=-(pi*torch.log_softmax(logits.masked_fill(~mask,-1e9),1)).sum(1).mean()+v.square().mean()+1e-4*manual_l2(net)
        opt.zero_grad();loss.backward();opt.step();new=model_hash(net);assert new!=h
        svc.publish(net)
        try:call(0)
        except RuntimeError as exc:assert 'STALE_MODEL' in str(exc)
        else:raise AssertionError('STALE_MODEL_ACCEPTED')
        remote=RemoteForward(svc.address,svc.auth,new,e.schema_hash)
        try:remote(torch.from_numpy(obs[None]))
        finally:remote.close()
        stats=svc.call('stats');assert stats['invalidations']==2 and stats['cache_hits']>=1 and stats['max_batch']>=2
        return {'status':'pass','actual_device':str(next(net.parameters()).device),'real_sgd_steps':1,'batch_kernel_rows_at_least':stats['max_batch'],'updated_model_inference':True,'stale_rejected':True,'stats':stats,'synthetic_inputs_not_training_production':True}
    finally:
        from perf.devices import cleanup_record
        primary=sys.exc_info()[1];closed=svc.close();cleanup_record(out/'service-close.json',closed)
        if closed.get('cleanup_errors') or not closed['cleanup_complete']:
            if primary is not None:primary.add_note('Service cleanup: '+str(closed))
            else:raise RuntimeError('SERVICE_CLEANUP_FAILED '+str(closed))

def learning(root,out,deadline):
    saved=torch.load(root/'fixture/checkpoint.pt',weights_only=False)
    # Use the four real episode files; only threshold counter is injected.
    # Build near-complete checkpoint using 63 REAL SGD steps, unchanged batch/exposure.
    target=out/'build';restore_copy(root/'fixture',target,root/'fixture/checkpoint.pt')
    e=Environment(config=saved['state']['full_config']);r=runtime.Runner(target,e,create=False)
    r.s['actor']=None;r.s['new_positions']=512;r.begin_training()
    save=r.save;r.save=lambda:None  # fixture generation only, not an optimization candidate
    r.train(time.monotonic()+max(1,deadline-time.time()-15),steps_limit=63);r.save=save;r.save()
    assert r.s['trainer'] and r.s['trainer']['steps_done']==63,'NEAR_COMPLETION_FIXTURE_TIMEOUT'
    checkpoint=torch.load(target/'checkpoint.pt',weights_only=False);fingerprints=[];details={}
    for name,writer in [('original',runtime.atomic_torch),('packed',packed_save)]:
        branch=out/name;restore_copy(target,branch)
        writer(branch/'checkpoint.pt',checkpoint);compare(checkpoint,torch.load(branch/'checkpoint.pt',weights_only=False))
        restored=runtime.Runner(branch,e,create=False);before=restored.s['sgd_steps'];old=restored.s['model_id']
        restored.train(time.monotonic()+max(1,deadline-time.time()-3),steps_limit=1)
        assert restored.s['generation']==1 and restored.s['sgd_steps']==before+1
        assert restored.s['trainer'] is None and not restored.pv_cache
        restored.collect(time.monotonic()+max(1,deadline-time.time()-2),until_decisions=restored.s['total_decisions']+1)
        assert restored.s['actor']['model_id']!=old and restored.s['actor']['records'][-1]['new_simulations']==100
        fingerprints.append(restored.fingerprint());details[name]={'published_model':restored.s['model_id'],'next_search_model':restored.s['actor']['model_id'],'real_branch_sgd_steps':1,'new_simulations':100}
    compare(*fingerprints)
    dump(out/'fixture-provenance.json',{'smoke_only':True,'actual_fixture_construction_sgd':63,'real_branch_steps':2,'injected_new_positions_counter':512,'actual_positions':saved['state']['total_positions'],'no_formal_productivity_or_learning_claim':True,'all_original_training_hyperparameters_unchanged':True})
    # STOP/resume on same run, immutable budget; duplicate lock is real flock.
    from perf.control import lock
    branch=out/'original';restored=runtime.Runner(branch,e,create=False);budget_before=copy.deepcopy(restored.s['budget'])
    (branch/'STOP').touch();n=restored.s['total_simulations'];restored.collect(time.monotonic()+1);assert restored.s['total_simulations']==n
    with lock(branch):
        try:other=lock(branch)
        except RuntimeError:pass
        else:other.close();raise AssertionError('DUPLICATE_LOCK_ACCEPTED')
    (branch/'STOP').unlink();restored=runtime.Runner(branch,e,create=False)
    assert restored.s['budget']['deadline_utc']==budget_before['deadline_utc']
    restored.collect(time.monotonic()+max(1,deadline-time.time()-2),until_decisions=restored.s['total_decisions']+1)
    assert restored.s['total_simulations']>n
    return {'save_variants':details,'next_sgd_optimizer_rng_search_equal':True,'safe_stop_resume_budget_lock':True,'synthetic_threshold_fixture':True}

def micro(root,out,deadline):
    from perf.micro import run
    result=run(root/'fixture',out,inject_save_failure='packed')
    assert result['serialization']['packed']['status']=='fail' and 'EXPECTED_INJECTED_SAVE' in result['serialization']['packed']['error']
    assert result['serialization']['original']['status']=='pass'
    assert result['correctness']['cpu1']['pass']
    from perf.gates import blocked
    assert blocked({'packed':True},result)==['packed_restore'] and not blocked({},result)
    from perf.smoke_inputs import valid,seal_reference
    seal_reference(out,valid(root))
    return {'cpu_correctness':result['correctness'],'serialization':result['serialization']}

def gpu_micro(root,out,deadline,reference=None):
    if not torch.cuda.is_available():return {'status':'device_unavailable'}
    from perf.micro import run
    result=run(root/'fixture',out,True,reference)
    assert result['correctness'].get('cuda',{}).get('pass'),'CUDA_NUMERIC_GATE_FAILED'
    return {'status':'pass','numeric':result['correctness']['cuda']}

def orphan_parent(root,out,deadline):
    # Child creates its own session, proving cleanup is not only killpg(parent).
    import subprocess
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'],start_new_session=True)
    dump(out/'orphan.json',{'child_pid':child.pid,'parent_exit':17,'separate_session':True})
    os._exit(17)

def shutdown_timeout(root,out,deadline):
    from perf.devices import Service
    svc=Service('cpu',close_timeout=.2,_test_shutdown_stall=True)
    primary='INJECTED_PRIMARY_WORKER_ERROR'
    try:raise RuntimeError(primary)
    except RuntimeError as exc:
        original=repr(exc)
        result=svc.close()
    assert result['cleanup_complete'] and result['cleanup_errors']
    assert 'RPC_CONTROL_TIMEOUT shutdown' in str(result['cleanup_errors'])
    assert not svc.process.is_alive()
    return {'status':'pass','primary_error_preserved':original,'cleanup':result,'controlled_actual_service_shutdown_stall':True}

def worker_failure(root,out,deadline):
    from perf.parallel import ParallelRunner
    e=env(deadline);r=ParallelRunner(out/'run',e,create=True,seed=4811,run_id='P48-controlled-worker-error',workers=2,options={'rpc_device':'cpu'})
    original=r.resolved_job
    def faulty(job):
        value=original(job);value['snapshot']=str(out/'MISSING_WORKER_SNAPSHOT.pt');return value
    r.resolved_job=faulty
    try:r.collect_probe(time.monotonic()+max(1,deadline-time.time()-2),max_cohorts=1)
    except FileNotFoundError as exc:
        assert 'MISSING_WORKER_SNAPSHOT' in str(exc)
        assert (r.root/'checkpoint.pt').exists() and (r.root/'rpc-metrics.json').exists()
        cleanup=load(r.root/'rpc-metrics.json');assert cleanup['cleanup_complete']
        return {'status':'pass','controlled_fault':'real parallel job missing snapshot','primary_error_preserved':repr(exc),'primary_error_message':str(exc),'primary_traceback':traceback.format_exc(),'finally_checkpoint_saved':True,'service_cleanup':cleanup}
    raise AssertionError('EXPECTED_REAL_WORKER_ERROR_NOT_OBSERVED')

def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['reserve_inputs','fixture','serialization','cpu_parallel','service','gpu_service','learning','micro','gpu_micro','orphan_parent','shutdown_timeout','worker_failure','controlled_failure','controlled_timeout']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--input-root',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--inject-fixture-failure',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        # micro.run owns configure; avoid setting interop threads twice.
        if a.case not in ('fixture','cpu_parallel','micro','gpu_micro'):configure()
        verify_source()
        if a.case=='fixture' and a.inject_fixture_failure:raise RuntimeError('EXPECTED_UPSTREAM_FIXTURE_FAILURE')
        if a.case=='controlled_failure':raise RuntimeError('EXPECTED_INJECTED_SUBITEM_FAILURE')
        if a.case=='controlled_timeout':
            dump(a.out/'timeout-entered.json',{'actual_timeout_work_entered':True});time.sleep(60)
        selected=a.input_root or a.root
        if a.case in ('serialization','learning','micro','gpu_micro'):
            from perf.smoke_inputs import valid
            inputs=valid(selected);assert inputs,'INVALID_SMOKE_INPUT_BUNDLE'
            dump(a.out/'input-consumption.json',{'basis_id':inputs['basis_id'],'source':str(selected),'integrity_verified_at_entry':True,'provenance':inputs['provenance']})
        if a.case=='gpu_service':value=service(selected,a.out,a.deadline,'cuda')
        elif a.case=='gpu_micro':value=gpu_micro(selected,a.out,a.deadline,a.reference)
        else:value=globals()[a.case](selected,a.out,a.deadline)
        dump(a.out/'case.json',{'status':'pass',**(value or {})})
    except Exception as exc:
        dump(a.out/'case.json',{'status':'fail','error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
