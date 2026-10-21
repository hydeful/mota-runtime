"""One isolated experiment. Nothing imports historical runs or cloud600."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import numpy as np
import torch
from perf.common import budget, configure, dump, load, utc, verify_source
from perf.patches import install, instrument_runner
from tasks.tower10.p45.environment import Environment
from tasks.tower10.p45 import runtime
from tasks.tower10.p45.network import model_hash

def audit(runner):
    from tasks.tower10.p45.contracts import validate_episode
    s=runner.s; root=runner.root; identities={m['id']:m for m in s['models']}
    completed=[];ids=set();positions={};consumed=0;all_sims=0;search_totals={}
    for p in sorted(root.glob('episode-*.json.gz')):
        with gzip.open(p,'rt') as f:ep=json.load(f)
        validate_episode(runner.env,ep,reference=True)
        if ep['episode_id'] in ids:raise ValueError('DUPLICATE_EPISODE')
        ids.add(ep['episode_id']);positions[ep['episode_id']]=len(ep['records'])
        identity=identities[ep['model_id']]
        if ep['weight_hash']!=identity['hash']:raise ValueError('EPISODE_MODEL_HASH')
        if any(r['new_simulations']!=100 for r in ep['records']):raise ValueError('SEARCH_BUDGET')
        all_sims+=100*len(ep['records'])
        for k,v in ep['search_stats'].items():
            if k!='max_depth':search_totals[k]=search_totals.get(k,0)+v
        if identity['generation']>0:consumed+=len(ep['records'])
        completed.append({'episode_id':ep['episode_id'],'positions':len(ep['records']),
                          'model_id':ep['model_id'],'weight_hash':ep['weight_hash'],'end':ep['end'],
                          'z':ep['z'],'terminal_utc':ep['terminal_utc']})
    committed_ids=[e['episode_id'] for e in s['episodes']]
    if len(committed_ids)!=len(set(committed_ids)):raise ValueError('DUPLICATE_COMMIT')
    if sum(positions[i] for i in committed_ids)!=s['total_positions']:raise ValueError('POSITION_LEDGER')
    steps=0;due=0;draws=0
    trainers=[load(p) for p in sorted(root.glob('training-N*.json'))]
    if s['trainer'] is not None:trainers.append(s['trainer'])
    for trainer in trainers:
        if trainer['steps_target']!=math.ceil(8*trainer['new_positions']/64):raise ValueError('EXPOSURE_TARGET')
        if len(trainer['losses'])!=trainer['steps_done']:raise ValueError('TRAINING_STEP_LEDGER')
        due+=trainer['steps_target'];steps+=trainer['steps_done']
        for row in trainer['losses']:
            if len(row['sample_pairs'])!=64:raise ValueError('BATCH_CHANGED')
            for eid,t in row['sample_pairs']:
                if eid not in committed_ids or not 0<=t<positions[eid]:raise ValueError('UNQUALIFIED_DRAW')
            draws+=64
    if steps!=s['sgd_steps']:raise ValueError('SGD_TOTAL')
    if s['trainer'] is None and s['new_positions']>=512:due+=math.ceil(8*s['new_positions']/64)
    for i,m in enumerate(s['models']):
        if m['generation']!=i or (i and m['parent']!=s['models'][i-1]['id']):raise ValueError('MODEL_CHAIN')
    pending_consumed=0
    if s['actor'] is not None:
        runner.validate_pending_actor()
        if identities[s['actor']['model_id']]['generation']>0:pending_consumed=len(s['actor']['records'])
    return {'pass':True,'reference_replayed':True,'complete_episodes':len(completed),
            'committed_episodes':len(committed_ids),'qualified_positions':s['total_positions'],
            'decisions':s['total_decisions'],'simulations':s['total_simulations'],
            'complete_episode_simulations':all_sims,'model_updates':s['generation'],
            'sgd_steps':steps,'sgd_steps_due':due,'sgd_steps_unfinished':due-steps,
            'actual_training_draws':draws,'updated_model_decisions':consumed+pending_consumed,'complete_episode_updated_model_decisions':consumed,'pending_updated_model_decisions':pending_consumed,'episodes':completed,
            'completed_episode_search_totals':search_totals,'publication_cache_clears':len(s['models']),
            'model_chain':[{k:m[k] for k in ('id','hash','generation','parent','initial_random','positions','sgd_steps')} for m in s['models']]}

def execute(root,options,seconds,deadline,seed,cohorts=0,until_decisions=None,learning_gate=False):
    configure();verify_source()
    config=budget(load(Path(__file__).resolve().parents[1]/'tasks/tower10/p45/config.json'),deadline)
    # Construct under the frozen audit, then declare the engineering thread override.
    env=Environment(config=config);profile=install(options,env)
    torch.set_num_threads(options.get('threads',1))
    env.compatibility['runtime']['threads']=torch.get_num_threads()
    factory=runtime.Network
    if options.get('gpu_train') or options.get('serial_gpu'):
        if not torch.cuda.is_available():raise RuntimeError('GPU_ENV_NOT_USABLE')
        from perf.devices import DeviceNetwork
        factory=DeviceNetwork
    if options.get('initial_state'):
        initial=np.load(options['initial_state'],allow_pickle=False)
        values={k:torch.from_numpy(initial[k].copy()) for k in initial.files}
        def common_initial(config):
            net=factory(config);net.load_state_dict(values);return net
        runtime.Network=common_initial
    else:runtime.Network=factory
    create=not (root/'checkpoint.pt').exists()
    cls=runtime.Runner
    extra={}
    if options.get('workers',1)>1:
        from perf.parallel import ParallelRunner
        cls=ParallelRunner;extra={'workers':options['workers'],'options':options}
    runner=cls(root,env,create=create,seed=seed,run_id='P48-perf-s'+str(seed),**extra)
    if options.get('initial_hash') and runner.s['models'][0]['hash']!=options['initial_hash']:raise ValueError('COMMON_FRESH_N0_HASH_MISMATCH')
    instrument_runner(runner,profile)
    started=time.monotonic();end=min(started+seconds,started+max(0,deadline-time.time()))
    dump(root/'collection-window.json',{'started_utc':utc(),'requested_seconds':seconds})
    recovery_equal=None
    if learning_gate:
        if options.get('workers',1)>1:runner.collect_probe(end,until_generation=1)
        else:runner.collect(end,until_generation=1)
        if runner.s['generation']<1:raise ValueError('LEARNING_GATE_DID_NOT_REACH_REAL_PUBLICATION')
        before=runner.fingerprint()
        runner=cls(root,env,create=False,**extra)
        recovery_equal=before==runner.fingerprint()
        if not recovery_equal:raise ValueError('RESTORE_FINGERPRINT_DIFFERENCE')
        instrument_runner(runner,profile)
        if options.get('workers',1)>1:runner.collect_probe(end,max_cohorts=1)
        else:runner.collect(end,until_decisions=runner.s['total_decisions']+1)
    elif options.get('workers',1)>1:runner.collect_probe(end,cohorts)
    else:runner.collect(end,max_cohorts=cohorts,until_decisions=until_decisions)
    returned=time.monotonic();elapsed=returned-started
    collection_profile=dict(profile.buckets);collection_calls=dict(profile.calls)
    audit_started=time.monotonic();data=audit(runner);audit_seconds=time.monotonic()-audit_started
    data.update(collection_seconds=elapsed,requested_seconds=seconds,collection_cleanup_seconds=max(0,returned-end) if not cohorts and not until_decisions else None,
                software={'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'torch_cuda':torch.version.cuda,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'coordinator_parameter_device':str(next(runner.model.parameters()).device),'actor_device':'cpu','inference_service_device':options.get('rpc_device')},
                profile_exclusive_seconds=collection_profile,profile_calls=collection_calls,
                audit_wall_seconds=audit_seconds,post_collect_audit_profile_seconds={k:v-collection_profile.get(k,0.) for k,v in profile.buckets.items()},
                original_stats=runner.s['stats'],options=options,seed=seed,
                cpu_tree_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                child_cpu_accounting='self plus all waited descendants; includes launch, warmup, validation and shutdown',
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                rpc_metrics=load(root/'rpc-metrics.json'),
                parallel_jobs=runner.s.get('parallel_jobs'),
                parallel_profiles=runner.s.get('parallel_profile',[]),
                coordinator_wait_seconds=runner.s.get('parallel_coordinator_wait_seconds',0),
                model_distribution_seconds=runner.s.get('model_distribution_seconds',0),
                reconcile_inclusive_seconds=runner.s.get('reconcile_inclusive_seconds',0),
                timing_note='profile buckets exclusive within one process; actor buckets overlap across processes. Original search counters and profile buckets overlap: do not add them. collection excludes interpreter/init and post-collection reference audit; outer wall time includes all.',
                training_profile=[{'generation':p.stem,'profile':load(p).get('profile',{})} for p in sorted(root.glob('training-N*.json'))],
                finished_utc=utc())
    # A screen is fixed work, not a truncated time window.
    data['fixed_work_complete']=len(data['episodes'])==4 if cohorts==1 else None
    data['learning_gate_pass']=bool(learning_gate and recovery_equal and data['model_updates']>=1 and data['sgd_steps']>=1 and data['updated_model_decisions']>=1)
    data['recovery_fingerprint_equal']=recovery_equal
    data['full_window_complete']=bool(not learning_gate and not cohorts and not until_decisions and elapsed>=seconds-.05 and not (root/'STOP').exists())
    data['qualified_positions_per_hour_extrapolated']=data['qualified_positions']/elapsed*3600
    dump(root/'result.json',data)
    return runner,data

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--options',required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--seed',type=int,default=4801);p.add_argument('--cohorts',type=int,default=0)
    p.add_argument('--learning-gate',action='store_true');p.add_argument('--until-decisions',type=int)
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    # Independent watchdog also works if the controller dies. Stop before the
    # global deadline; all child processes inherit this process group.
    done=threading.Event()
    def watchdog():
        if not done.wait(max(0,a.deadline-time.time())):
            (root/'STOP').touch()
            if not done.wait(5):os.killpg(os.getpgrp(),signal.SIGTERM)
    threading.Thread(target=watchdog,daemon=True).start()
    try:execute(root,load(a.options),a.seconds,a.deadline,a.seed,a.cohorts,until_decisions=a.until_decisions,learning_gate=a.learning_gate)
    finally:done.set()

if __name__=='__main__':main()
