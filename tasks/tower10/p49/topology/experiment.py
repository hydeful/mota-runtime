"""Fixed five-round experiment controller, wrapping the original P45 Runner."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from .environment import Environment,make_config
from .runtime import Runner,atomic_json
from .evaluation import evaluate,BudgetStop
from .guard import install

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
DEFAULT_RUN='/workspace/p49-object-topology-run-49001'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(destination):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    plan=HERE/'EXPERIMENT_PLAN.md'
    if not plan.is_file():raise ValueError('MISSING_FROZEN_PLAN')
    certificate=json.loads((HERE/'CERTIFICATION.json').read_text())
    if not certificate['passed']:raise ValueError('UNCERTIFIED_TOPOLOGY_ADAPTER')
    config=make_config()
    now=datetime.datetime.now(datetime.timezone.utc)
    began=time.monotonic();deadline=began+3600;work_deadline=deadline-120
    config['budget'].update(session_start_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(seconds=3600)).isoformat())
    atomic_json(destination/'config.json',config)
    (destination/'EXPERIMENT_PLAN.md').write_bytes(plan.read_bytes())
    sources={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    for relative in ['tasks/tower10/p45/network.py','tasks/tower10/p45/runtime.py','tasks/tower10/p45/search.py','tasks/tower10/p45/contracts.py','tasks/tower10/p45/config.json','tasks/tower10/neural/common.py','tasks/tower10/neural/vendor/kernel.py','tasks/tower10/neural/vendor/topology_engine.py','tasks/tower10/p49/learning/evaluation.py','tasks/tower10/p49/topology/initial.json','tasks/tower10/p49/topology/identity.json']:
        sources[relative]=sha(ROOT/relative)
    atomic_json(destination/'sources.json',{'base_commit':'61899150c73d401d73ff0ac45626d921e98e5d5c','files':sources,'plan_sha256':sha(plan),'local_checkout':'selected source snapshot; no local Git HEAD; private remote branch is authoritative'})
    status={'phase':'initialize','generation':0,'complete':False,'formal_start_utc':now.isoformat(),'budget_seconds':3600,'rounds':[],'evaluation_files':[]}
    def heartbeat(event):
        status['elapsed_seconds']=time.monotonic()-began;status['last_event']=event
        atomic_json(destination/'progress.json',status)
        with (destination/'events.jsonl').open('a') as f:f.write(json.dumps({'elapsed_seconds':status['elapsed_seconds'],**event},ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps({'heartbeat':event,'elapsed_seconds':status['elapsed_seconds']},ensure_ascii=False,allow_nan=False),flush=True)
    denied,runtime_counts=install(ROOT,destination)
    runner=None
    try:
        env=Environment(config);atomic_json(destination/'schema.json',env.schema_description)
        runner=Runner(destination/'chain',env,create=True,seed=49001,run_id='P49-topology-seed49001')
        status['phase']='evaluation';heartbeat({'event':'evaluate_start','generation':0})
        evaluate(env,destination/'chain'/f"{runner.s['model_id']}.pt",destination/'evaluation',work_deadline,heartbeat)
        for generation in range(1,6):
            status.update(phase='collect_and_train',generation=runner.s['generation']);start=time.monotonic()
            heartbeat({'event':'round_start','target_generation':generation})
            positions=runner.s['total_positions'];steps=runner.s['sgd_steps'];episodes=len(runner.s['episodes'])
            runner.collect(work_deadline,until_generation=generation)
            if runner.s['generation']!=generation:raise BudgetStop('COLLECTION_OR_TRAINING_BUDGET')
            training=json.loads((runner.root/f'training-N{generation}.json').read_text())
            receipt={'generation':generation,'new_positions':runner.s['total_positions']-positions,'new_episodes':len(runner.s['episodes'])-episodes,'SGD_steps':runner.s['sgd_steps']-steps,'SGD_target':training['steps_target'],'collect_train_publish_seconds':time.monotonic()-start,'model_id':runner.s['model_id'],'weight_hash':runner.s['models'][-1]['hash']}
            if receipt['SGD_steps']!=receipt['SGD_target']:raise ValueError('UNPAID_SGD')
            status['rounds'].append(receipt);status.update(generation=generation,phase='evaluation');heartbeat({'event':'round_published',**receipt})
            evaluate(env,runner.root/f"{runner.s['model_id']}.pt",destination/'evaluation',work_deadline,heartbeat)
        status['phase']='N5_reload_reflow';heartbeat({'event':'N5_reload_start'})
        runner=Runner(destination/'chain',env,create=False)
        before=len(runner.s['episodes']);start=time.monotonic()
        runner.collect(work_deadline,max_cohorts=1,no_train=True)
        fresh=runner.s['episodes'][before:]
        if len(fresh)!=4 or runner.s['generation']!=5 or len(runner.s['models'])!=6 or runner.s['trainer'] is not None:raise ValueError('N5_REFLOW_INCOMPLETE')
        if not all(ep['model_id']==runner.s['model_id'] and ep['weight_hash']==runner.s['models'][-1]['hash'] for ep in fresh):raise ValueError('N5_REFLOW_WEIGHT_MISMATCH')
        atomic_json(destination/'N5-reflow.json',{'reload':True,'model_id':runner.s['model_id'],'weight_hash':runner.s['models'][-1]['hash'],'episodes':fresh,'seconds':time.monotonic()-start,'N6_triggered':False})
        status.update(phase='summarize',complete=True);heartbeat({'event':'learning_chain_complete','generation':5})
    except Exception as exc:
        status['error']=repr(exc);status['traceback']=traceback.format_exc();status['phase']='stopped'
        if runner is not None:runner.save()
        heartbeat({'event':'formal_stopped','error':repr(exc)})
    finally:
        sys.setprofile(None)
        status['runtime_no_grid_profile']=dict(runtime_counts)
        status['denied_behavior_reads']=denied
        status['experiment_seconds_before_report']=time.monotonic()-began
        atomic_json(destination/'progress.json',status)
    remaining=deadline-time.monotonic()
    if remaining>0:
        # Separate process: only this independent report reads certified optimum
        # and random baseline. Neither value enters the actor/trainer process.
        result=subprocess.run([sys.executable,'-m','tasks.tower10.p49.topology.report',str(destination)],cwd=ROOT,timeout=remaining,check=False)
        status['report_returncode']=result.returncode
    status['total_formal_wall_seconds']=time.monotonic()-began
    status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(destination/'progress.json',status)
    # The report runs before the final progress write. Refresh identities only
    # after that write so the manifest cannot describe a stale progress.json.
    artifact_path=destination/'ARTIFACTS.json'
    if artifact_path.exists():
        artifact=json.loads(artifact_path.read_text())
        artifact['files']={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*')) if p.is_file() and p!=artifact_path}
        atomic_json(artifact_path,artifact)
    print(json.dumps(status,ensure_ascii=False,allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','status']);p.add_argument('--root',default=DEFAULT_RUN);args=p.parse_args()
    if args.command=='run':run(args.root)
    else:
        path=Path(args.root)/'progress.json'
        if not path.exists():raise SystemExit('Run has not started.')
        s=json.loads(path.read_text());print(json.dumps({k:v for k,v in s.items() if k not in ['traceback','evaluation_files']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
