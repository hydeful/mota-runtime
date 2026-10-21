"""P45 V0: retain frozen P43 learner/contracts, parallelize whole episodes.
All files are isolated probe runs. No P43 formal state is modified.
"""
import argparse,copy,datetime,hashlib,json,multiprocessing as mp,os,resource,shutil,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.p45.environment import Environment,install_guard
from tasks.tower10.p45.runtime import Runner,atomic_json,atomic_torch,model_hash,structure_hash,rng_state
from tasks.tower10.p45.contracts import validate_episode

_ENV=None
_CONFIG=None
class EpisodeComplete(Exception): pass
class OneEpisodeRunner(Runner):
    def finish_episode(self):
        super().finish_episode()
        raise EpisodeComplete()

def setup(config):
    global _ENV,_CONFIG
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    _CONFIG=config;_ENV=Environment(config=_CONFIG)

def episode_job(job):
    """Worker owns its environment, full actor tree, two RNGs and local cache."""
    path=Path(job['path']);start=time.perf_counter();cpu=time.process_time()
    install_guard(path)
    create=not (path/'checkpoint.pt').exists()
    snap=torch.load(job['snapshot'],map_location='cpu',weights_only=False) if create else None
    runner=OneEpisodeRunner(path,_ENV,create=create,seed=job['seed'],run_id=job['run_id'],initial_snapshot=snap,worker_episode_id=job['episode_id'])
    if runner.s['model_id']!=job['model_id'] or model_hash(runner.model)!=job['weight_hash']:raise ValueError('PROBE_WORKER_MODEL_BINDING')
    if runner.s['cohort']:
        if len(runner.s['cohort'])!=1 or runner.s['cohort'][0]['episode_id']!=job['episode_id']:raise ValueError('PROBE_WORKER_TERMINAL_ID')
    else:
        from tasks.tower10.p45.search import Search
        original_simulate=Search.simulate
        if job.get('observe_recovery'):
            def observed_simulate(search,node,decision):
                value=original_simulate(search,node,decision)
                atomic_json(path/'recovery-progress.json',{'episode_id':job['episode_id'],'completed_decision_simulations':decision['completed'],'total_search_simulations':search.stats['new_simulations'],'monotonic':time.monotonic()})
                return value
            Search.simulate=observed_simulate
        try:
            runner.collect(job['deadline'],no_train=True,until_simulations=job.get('until_simulations'))
        except EpisodeComplete:pass
        finally:Search.simulate=original_simulate
    done=bool(runner.s['cohort'])
    result={'episode_id':job['episode_id'],'finished':done,'decisions':runner.s['total_decisions'],'simulations':runner.s['total_simulations'],'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'checkpoint_path':str(path/'checkpoint.pt'),'weight_hash':model_hash(runner.model),'worker_pid':os.getpid()}
    atomic_json(path/'worker-result.json',result)
    return result

class MultiActorRunner(Runner):
    execution_mode="parallel"
    def __init__(self,*args,workers=2,**kwargs):
        if workers!=2:raise ValueError('V0_REQUIRES_TWO_WORKERS')
        self.workers=workers
        super().__init__(*args,**kwargs)
        self.s.setdefault('parallel_jobs',None);self.s.setdefault('parallel_profile',[])
        self.s.setdefault('execution_mode','parallel')
        if self.s['actor'] is not None:raise ValueError('PROBE_SERIAL_PENDING_NOT_SUPPORTED')
    def status(self):
        super().status()
        status=json.loads((self.root/'status.json').read_text())
        status['execution_mode']='parallel';status['parallel_jobs']=self.s.get('parallel_jobs')
        atomic_json(self.root/'status.json',status)
    def begin_training(self):
        if self.s.get('parallel_jobs'):raise ValueError('TRAINING_WITH_PENDING_PARALLEL_JOBS')
        return super().begin_training()
    def resolved_job(self,job):
        out=dict(job)
        for key in ('path','snapshot'):
            value=Path(job[key])
            if value.is_absolute() or '..' in value.parts:raise ValueError('NONPORTABLE_JOB_PATH')
            out[key]=str(self.root/value)
        return out
    def allocate(self):
        if self.s['parallel_jobs'] is not None:return
        if self.s['cohort']:raise ValueError('PROBE_ALLOCATE_NONEMPTY_COHORT')
        generation=self.s['generation'];snap_path=self.root/f'worker-model-N{generation}.pt'
        if not snap_path.exists():atomic_torch(snap_path,{'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'models':self.s['models'],'generation':generation,'model_id':self.s['model_id'],'parent':self.s['parent'],'budget':self.s['budget']})
        first=self.s['next_episode'];jobs=[]
        for eid in range(first,first+self.s['recipe']['logical_cohort']):
            jobs.append({'episode_id':eid,'path':str(Path('actors')/f'episode-{eid:06d}'),'seed':self.s['seed'],'run_id':self.s['run_id'],'snapshot':snap_path.name,'model_id':self.s['model_id'],'weight_hash':self.s['models'][-1]['hash'],'accounted_decisions':0,'accounted_simulations':0,'finished':False,'imported':False})
        self.s['next_episode']+=len(jobs);self.s['parallel_jobs']=jobs;self.save()
    def reconcile(self,job,result):
        saved=torch.load(Path(self.resolved_job(job)['path'])/'checkpoint.pt',map_location='cpu',weights_only=False);s=saved['state']
        if s['model_id']!=job['model_id'] or saved['model_hash']!=job['weight_hash'] or saved['training_hash']!=self.s['models'][-1]['training_hash']:raise ValueError('PROBE_RECONCILE_IDENTITY')
        if s['sgd_steps']!=0 or s['episodes'] or s['replay']:raise ValueError('PROBE_WORKER_UNEXPECTED_TRAINING')
        for counter in ('decisions','simulations'):
            total=s['total_'+counter];prev=job['accounted_'+counter]
            if total<prev:raise ValueError('PROBE_COUNTER_ROLLBACK')
            self.s['total_'+counter]+=total-prev;job['accounted_'+counter]=total
        job['finished']=bool(s['cohort'])
        if job['finished'] and not job['imported']:
            brief=s['cohort'][0]
            if brief['episode_id']!=job['episode_id']:raise ValueError('PROBE_IMPORT_EPISODE_ID')
            source=Path(self.resolved_job(job)['path'])/brief['path'];data=source.read_bytes()
            if hashlib.sha256(data).hexdigest()!=brief['sha256']:raise ValueError('PROBE_IMPORT_HASH')
            import gzip
            ep=json.loads(gzip.decompress(data));validate_episode(self.env,ep)
            target=self.root/brief['path'];tmp=target.with_suffix(target.suffix+'.tmp');tmp.write_bytes(data);os.replace(tmp,target)
            if any(x['episode_id']==brief['episode_id'] for x in self.s['cohort']+self.s['episodes']):raise ValueError('PROBE_DUPLICATE_IMPORT')
            self.s['cohort'].append(brief);self.s['cohort'].sort(key=lambda x:x['episode_id']);job['imported']=True
        job['durable_checkpoint_sha256']=hashlib.sha256((Path(self.resolved_job(job)['path'])/'checkpoint.pt').read_bytes()).hexdigest()
        self.s['parallel_profile'].append(result);self.save()
    def collect_parallel(self,deadline,max_cohorts=0,no_train=False,worker_until_simulations=None,observe_recovery=False):
        from tasks.tower10.p45.runtime import cap_deadline
        deadline=cap_deadline(deadline,self.s['budget']);cohorts=0
        self.s.setdefault('first_collection_utc',datetime.datetime.now(datetime.timezone.utc).isoformat());self.s.setdefault('first_collection_monotonic',time.monotonic())
        started=time.perf_counter();pool_start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),initializer=setup,initargs=(self.env.config,)) as pool:
            while time.monotonic()<deadline and not self.should_stop():
                if self.s['trainer'] is not None:
                    self.train(deadline)
                    if self.s['trainer'] is not None:break
                if len(self.s['cohort'])==self.s['recipe']['logical_cohort']:
                    self.s['parallel_jobs']=None;self.save();self.commit_cohort();cohorts+=1
                    if not no_train and self.s['new_positions']>=self.s['recipe']['min_new_positions']:
                        self.begin_training();continue
                    if max_cohorts and cohorts>=max_cohorts:break
                self.allocate();jobs=self.s['parallel_jobs'];futures=[]
                for job in jobs:
                    if job['finished']:continue
                    payload=dict(self.resolved_job(job),deadline=deadline,observe_recovery=observe_recovery)
                    if worker_until_simulations is not None:payload['until_simulations']=worker_until_simulations
                    futures.append((job,pool.submit(episode_job,payload)))
                if not futures:continue
                wait_started=time.perf_counter()
                for job,future in futures:self.reconcile(job,future.result())
                self.s.setdefault('parallel_coordinator_wait_seconds',0.);self.s['parallel_coordinator_wait_seconds']+=time.perf_counter()-wait_started
                if worker_until_simulations is not None:break
                if not all(j['finished'] for j in jobs):break
        self.s['parallel_pool_lifetime_seconds']=self.s.get('parallel_pool_lifetime_seconds',0.)+time.perf_counter()-pool_start
        self.s['parallel_collect_wall_seconds']=self.s.get('parallel_collect_wall_seconds',0.)+time.perf_counter()-started
        self.save()
    def parallel_fingerprint(self):
        out={'parent':super().fingerprint(),'jobs':[]}
        for job in self.s.get('parallel_jobs') or []:
            path=Path(self.resolved_job(job)['path'])/'checkpoint.pt'
            if not path.exists():out['jobs'].append({'episode_id':job['episode_id'],'not_started':True});continue
            # Restore in a fresh subprocess for each actor when independently validating;
            # here hash all logical actor fields without changing coordinator RNG.
            x=torch.load(path,map_location='cpu',weights_only=False);s=x['state'];a=s['actor']
            def tree(n):
                if n is None:return None
                return {'state':self.env.dynamic(n.state),'t':n.t,'ids':n.ids,'prior':n.prior,'n':n.n,'w':n.w,'terminal':n.terminal,'value':n.value,'obs_hash':n.obs_hash,'model_id':n.model_id,'children':{k:tree(v) for k,v in sorted(n.children.items())}}
            decision=None if a is None or a['decision'] is None else {k:v for k,v in a['decision'].items() if k!='stats_before'}
            records=[] if a is None else copy.deepcopy(a['records'])
            for r in records:r.pop('search_cost',None)
            out['jobs'].append({'episode_id':job['episode_id'],'model':x['model_hash'],'optimizer':structure_hash(x['optimizer']),'global_rng':structure_hash(x['rng']),'actor_rng':None if a is None else structure_hash((a['noise_rng'].bit_generator.state,a['action_rng'].bit_generator.state)),'tree':structure_hash(tree(a['root']) if a else None),'decision':structure_hash(decision),'records':structure_hash(records),'simulations':s['total_simulations'],'decisions':s['total_decisions'],'finished':bool(s['cohort'])})
        return out
