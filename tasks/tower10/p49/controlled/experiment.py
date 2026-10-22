"""One deadline for both stages and all pairs; workers isolate RNG and B guards."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from .network import Network,model_hash
from .environment import make_config,assert_configs
from .runtime import seed_all,atomic_json,atomic_torch,structure_hash,rng_state

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
SEEDS=[49001,49002,49003]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def initialize(root,phase,seed):
    # Exactly one authoritative random state per pair, copied to both workers.
    seed_all(seed);model=Network(make_config('A',seed,phase)['network'])
    assert sum(p.numel() for p in model.parameters())==123994
    signature=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in model.named_parameters()]
    path=root/'initializations'/phase/f'initial-{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'hash':model_hash(model),'signature':signature,'seed':seed,'phase':phase,
           'rng':rng_state(),'rng_hash':structure_hash(rng_state()),'parameter_count':123994}
    atomic_torch(path,state)
    atomic_json(path.with_suffix('.json'),{k:v for k,v in state.items() if k not in ['model','rng']})
    return path


def run(root,data,execution_commit):
    root=Path(root);data=Path(data);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert_configs(make_config('A'),make_config('B'))
    preflight=json.loads((HERE/'PREFLIGHT.json').read_text());assert preflight['passed'] and preflight['SGD_steps']==0
    manifest=json.loads((data/'MANIFEST.json').read_text());assert sha(data/'MANIFEST.json')==json.loads((HERE/'DATA_FREEZE.json').read_text())['manifest_sha256']
    for rel,expected in manifest['files'].items():assert sha(data/rel)==expected,rel
    shutil.copytree(data,root/'data')
    began=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc);deadline=began+3600;work_deadline=deadline-120
    end=start+datetime.timedelta(seconds=3480)
    status={'phase':'fixed_experience_fitting','complete':False,'formal_start_utc':start.isoformat(),
       'formal_budget_seconds':3600,'execution_commit':execution_commit,'completed_jobs':[]}
    sources={str(p.relative_to(REPO)):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']}
    for old in ['learning','topology']:
        previous=json.loads((Path('/workspace/p49-learning-run-49001' if old=='learning' else '/workspace/p49-object-topology-run-49001')/'sources.json').read_text())
        for rel in previous['files']:sources[rel]=sha(REPO/rel)
    atomic_json(root/'sources.json',{'execution_commit':execution_commit,'files':sources,'plan_sha256':sha(HERE/'CONTROLLED_PLAN.md'),
       'base_commit':'5da003c726eeb8d5939b62e6c8700612652e5a14'})
    for rel in sources:
        dest=root/'executed_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    def progress(event):
        status.update(last_completed_step=event,elapsed_seconds=time.monotonic()-began,last_update_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(root/'progress.json',status);print(json.dumps({'heartbeat':event,'elapsed_seconds':status['elapsed_seconds']}),flush=True)
    active=None
    try:
        for phase in ['fit','online']:
            status['phase']='fixed_experience_fitting' if phase=='fit' else 'online_closed_loops'
            for seed in SEEDS:
                initial=initialize(root,phase,seed)
                for condition in ['A','B']:
                    if time.monotonic()>=work_deadline:raise TimeoutError('SHARED_FORMAL_BUDGET')
                    job=root/phase/str(seed)/condition;job.mkdir(parents=True)
                    status.update(pair_seed=seed,condition=condition,active_job=str(job.relative_to(root)))
                    progress({'event':'job_start','phase':phase,'seed':seed,'condition':condition})
                    command=[sys.executable,'-u','-m','tasks.tower10.p49.controlled.worker','--phase',phase,'--condition',condition,
                      '--seed',str(seed),'--initial',str(initial),'--data',str(root/'data'),'--output',str(job),
                      '--start-utc',start.isoformat(),'--deadline-utc',end.isoformat()]
                    with (job/'worker.log').open('w') as log:
                        active=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                        while active.poll() is None:
                            if time.monotonic()>=work_deadline:
                                # Workers check this deadline too; no new stage or budget reset.
                                try:active.wait(timeout=5)
                                except subprocess.TimeoutExpired:active.terminate();active.wait(timeout=5)
                                raise TimeoutError('SHARED_FORMAL_BUDGET')
                            try:active.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child=job/'progress.json'
                                if child.exists():status['worker_progress']=json.loads(child.read_text())
                                progress({'event':'worker_heartbeat','phase':phase,'seed':seed,'condition':condition})
                    if active.returncode!=0:raise RuntimeError(f'WORKER_FAILED {phase}/{seed}/{condition}: '+str(active.returncode))
                    child=json.loads((job/'progress.json').read_text());assert child['complete']
                    status['completed_jobs'].append({'phase':phase,'seed':seed,'condition':condition,'seconds':child['worker_seconds']})
                    progress({'event':'job_complete','phase':phase,'seed':seed,'condition':condition})
        status['calculations_complete']=True;status['phase']='local_report';progress({'event':'all_calculations_complete'})
        result=subprocess.run([sys.executable,'-m','tasks.tower10.p49.controlled.report',str(root)],cwd=REPO,timeout=max(1,deadline-time.monotonic()),check=False)
        if result.returncode:raise RuntimeError('REPORT_FAILED '+str(result.returncode))
        status['complete']=True
    except BaseException as exc:
        status.update(error=repr(exc),phase='stopped');progress({'event':'stopped','error':repr(exc)})
    finally:
        status.update(total_formal_wall_seconds=time.monotonic()-began,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        progress({'event':'formal_finish','complete':status['complete']})
        atomic_json(root/'ARTIFACTS.json',{'files':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='ARTIFACTS.json'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','status']);parser.add_argument('--root',default='/workspace/p49-controlled-run');parser.add_argument('--data',default='/workspace/p49-controlled-data');parser.add_argument('--execution-commit')
    args=parser.parse_args()
    if args.command=='run':
        if not args.execution_commit:raise SystemExit('Exact frozen source commit required')
        run(args.root,args.data,args.execution_commit)
    else:print((Path(args.root)/'progress.json').read_text())
