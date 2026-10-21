"""Four-episode cohort, one shared synchronous learner, persistent actor caches."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import resource
import time
import torch
from perf.common import configure, dump
from perf.patches import install, instrument_runner
from tasks.tower10.p45.environment import Environment
from tasks.tower10.p45.runtime import Runner, cap_deadline, atomic_torch
from tasks.tower10.p45.parallel import MultiActorRunner, OneEpisodeRunner, EpisodeComplete
from tasks.tower10.p45.network import model_hash

_env=None;_options=None;_cache=None;_cache_hash=None;_invalidations=0;_profile=None;_rpc=None

def setup(config,options,rpc):
    global _env,_options,_cache,_profile,_rpc
    configure();_env=Environment(config=config);_options=options;_rpc=rpc
    _profile=install(options,_env);_cache=OrderedDict()

def job(payload):
    global _cache_hash,_invalidations
    began=time.perf_counter();cpu=time.process_time();path=Path(payload['path'])
    create=not (path/'checkpoint.pt').exists()
    snapshot=torch.load(payload['snapshot'],map_location='cpu',weights_only=False) if create else None
    r=OneEpisodeRunner(path,_env,create=create,seed=payload['seed'],run_id=payload['run_id'],initial_snapshot=snapshot,worker_episode_id=payload['episode_id'])
    if model_hash(r.model)!=payload['weight_hash']:raise ValueError('WORKER_WEIGHT_BINDING')
    if _options.get('persistent_cache'):
        if _cache_hash!=payload['weight_hash']:
            _cache.clear();_cache_hash=payload['weight_hash'];_invalidations+=1
        r.pv_cache=_cache
    remote=None
    if _rpc:
        from perf.devices import RemoteForward
        remote=RemoteForward(*_rpc,payload['weight_hash'],_env.schema_hash)
        r.model.forward=remote
    instrument_runner(r,_profile)
    before=dict(_profile.buckets)
    try:
        if not r.s['cohort']:r.collect(payload['deadline'],no_train=True)
    except EpisodeComplete:pass
    finally:
        if remote:
            error=remote.close()
            if error:
                import sys
                if sys.exc_info()[1] is not None:sys.exc_info()[1].add_note('Remote connection cleanup: '+error)
    result={'episode_id':payload['episode_id'],'finished':bool(r.s['cohort']),
            'decisions':r.s['total_decisions'],'simulations':r.s['total_simulations'],
            'wall_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'worker_pid':__import__('os').getpid(),'weight_hash':model_hash(r.model),'profile_exclusive_seconds':{k:v-before.get(k,0) for k,v in _profile.buckets.items()},
            'worker_cache_invalidations_lifetime':_invalidations,'rpc_wait_seconds':remote.wait_seconds if remote else 0.,
            'rpc_send_seconds':remote.send_seconds if remote else 0.,'rpc_response_wait_seconds':remote.response_wait_seconds if remote else 0.,
            'rpc_requests':remote.requests if remote else 0}
    dump(path/'worker-result.json',result)
    return result

class ParallelRunner(MultiActorRunner):
    def __init__(self,*args,workers=2,options=None,**kwargs):
        if workers not in (2,4):raise ValueError('COHORT_FOUR_SUPPORTS_TWO_OR_FOUR_ACTIVE_WORKERS')
        self.workers=workers;self.options=options or {}
        Runner.__init__(self,*args,**kwargs)
        self.s.setdefault('parallel_jobs',None);self.s.setdefault('parallel_profile',[])
        self.s.setdefault('execution_mode','parallel')
        if self.s['actor'] is not None:raise ValueError('SERIAL_ACTOR_IN_PARALLEL_RUN')

    def collect_probe(self,deadline,max_cohorts=0,until_generation=None):
        deadline=cap_deadline(deadline,self.s['budget']);cohorts=0;service=None
        try:
            if self.options.get('rpc_device'):
                from perf.devices import Service
                service=Service(self.options['rpc_device'],self.options.get('rpc_batch_wait',.0002))
            rpc=(service.address,service.auth) if service else None
            with ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),initializer=setup,initargs=(self.env.config,self.options,rpc)) as pool:
                sent=None
                while time.monotonic()<deadline and not self.should_stop():
                    if until_generation is not None and self.s['generation']>=until_generation:break
                    if self.s['trainer'] is not None:
                        self.train(deadline)
                        if self.s['trainer'] is not None:break
                        if until_generation is not None and self.s['generation']>=until_generation:break
                    if len(self.s['cohort'])==4:
                        self.s['parallel_jobs']=None;self.save();self.commit_cohort();cohorts+=1
                        if self.s['new_positions']>=512:
                            self.begin_training();continue
                        if max_cohorts and cohorts>=max_cohorts:break
                    if service and sent!=self.s['model_id']:
                        began=time.perf_counter();service.publish(self.model);sent=self.s['model_id']
                        self.s['model_distribution_seconds']=self.s.get('model_distribution_seconds',0.)+time.perf_counter()-began
                    self.allocate();futures=[]
                    for j in self.s['parallel_jobs']:
                        if not j['finished']:futures.append((j,pool.submit(job,dict(self.resolved_job(j),deadline=deadline))))
                    began=time.perf_counter()
                    ordered=[(j,f) for j,f in futures]
                    if self.options.get('completion_order'):
                        by_future={f:j for j,f in futures};ordered=((by_future[f],f) for f in as_completed(by_future))
                    try:
                        for j,f in ordered:
                            reconcile_started=time.perf_counter();self.reconcile(j,f.result())
                            self.s['reconcile_inclusive_seconds']=self.s.get('reconcile_inclusive_seconds',0.)+time.perf_counter()-reconcile_started
                    except Exception:
                        (self.root/'STOP').touch()
                        for _,f in futures:f.cancel()
                        raise
                    self.s['parallel_coordinator_wait_seconds']=self.s.get('parallel_coordinator_wait_seconds',0.)+time.perf_counter()-began
                    if not all(j['finished'] for j in self.s['parallel_jobs']):break
        finally:
            import sys
            from perf.devices import cleanup_record
            primary=sys.exc_info()[1];errors=[]
            if service:
                try:
                    result=service.close();cleanup_record(self.root/'rpc-metrics.json',result)
                    if result.get('cleanup_errors') or not result['cleanup_complete']:errors.append(result)
                except Exception as exc:errors.append('cleanup evidence: '+repr(exc))
            try:self.save()
            except Exception as exc:
                if primary is not None:primary.add_note('Final save failed: '+repr(exc))
                else:raise
            if errors:
                if primary is not None:primary.add_note('Service cleanup errors: '+str(errors))
                else:raise RuntimeError('SERVICE_CLEANUP_FAILED '+str(errors))
