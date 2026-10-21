"""Persistent single-chain collection, joint SGD, evaluation and recovery."""
import argparse,copy,gzip,hashlib,json,math,os,random,resource,time,datetime,io,shutil
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .network import Network,model_hash,manual_l2,gradients
from .search import Search,Node
from .contracts import validate_episode,validate_record
from tasks.tower10.neural.common import digest

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def restore_rng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_torch(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');began=time.perf_counter()
    buffer=io.BytesIO();torch.save(value,buffer);serialization=time.perf_counter()-began;began=time.perf_counter()
    with tmp.open('wb') as f:f.write(buffer.getbuffer());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return {'serialization_seconds':serialization,'write_sync_seconds':time.perf_counter()-began,'bytes':buffer.tell()}
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def utc_seconds(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def cap_deadline(deadline,budget):return min(deadline,time.monotonic()+max(0,utc_seconds(budget['deadline_utc'])-time.time()))
def structure_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(b'torch');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(b'numpy');h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda k:repr(k)):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value);return h.hexdigest()

class Runner:
    execution_mode="serial"
    def __init__(self,root,env,create=False,seed=4201,run_id='P43-V1-R1',initial_snapshot=None,worker_episode_id=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.env=env
        self.model=Network(env.config['network']);self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01,momentum=.9,weight_decay=0)
        self.pv_cache=OrderedDict();self.started=time.monotonic();self.last_saved=time.monotonic();self.last_cpu=time.process_time()
        if create:
            if (self.root/'checkpoint.pt').exists():raise ValueError('RUN_EXISTS')
            seed_all(seed);self.model=Network(env.config['network']);self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01,momentum=.9,weight_decay=0)
            self.s={'run_id':'P43-V1-R1','seed':seed,'generation':0,'model_id':'','parent':None,'models':[],'next_episode':0,'cohort':[],'episodes':[],'replay':[],'new_positions':0,'total_positions':0,'total_decisions':0,'total_simulations':0,'sgd_steps':0,'actor':None,'trainer':None,'stats':{},'elapsed_seconds':0.,'recipe':copy.deepcopy(env.config['training']),'recipe_frozen':False,'resume_count':0,'data_contract':env.config_hash,'full_config':env.config,'budget':{},'first_failures':[]}
            self.s['execution_mode']=self.execution_mode
            self.s['run_id']=run_id;self.s['compatibility']=env.compatibility;self.s['budget']=dict(env.config['budget'],active_seconds=0.);atomic_json(self.root/'recipe.json',self.s['recipe'])
            if initial_snapshot is None:self.publish(initial=True)
            else:
                snap=initial_snapshot
                self.model.load_state_dict(snap['model']);self.optimizer.load_state_dict(snap['optimizer'])
                for key in ('models','generation','model_id','parent','budget'):self.s[key]=copy.deepcopy(snap[key])
                self.s['next_episode']=worker_episode_id
                identity=self.s['models'][-1]
                if identity['id']!=self.s['model_id'] or identity['hash']!=model_hash(self.model):raise ValueError('WORKER_SNAPSHOT_BINDING')
                atomic_json(self.root/'models.json',self.s['models'])
                atomic_torch(self.root/(identity['id']+'.pt'),{'model':self.model.state_dict(),'identity':identity})
            self.save()
        else:
            saved=torch.load(self.root/'checkpoint.pt',map_location='cpu',weights_only=False)
            self.s=saved['state']
            if self.s['data_contract']!=env.config_hash:raise ValueError('RESTORE_DATA_CONTRACT_MISMATCH')
            for key,value in env.compatibility.items():
                if self.s['compatibility'].get(key)!=value:raise ValueError('RESTORE_COMPATIBILITY_'+key)
            if self.s['recipe']!=env.config['training']:raise ValueError('RESTORE_CONFIG_RECIPE_MISMATCH')
            if json.loads((self.root/'recipe.json').read_text())!=self.s['recipe'] or saved['training_hash']!=digest(self.s['recipe']):raise ValueError('RESTORE_RECIPE_MISMATCH')
            self.model.load_state_dict(saved['model'],strict=True);self.optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);self.s['resume_count']+=1
            self.s.setdefault('recovery_events',[]).append({'durable_decisions':self.s['total_decisions'],'durable_simulations':self.s['total_simulations'],'durable_sgd_steps':self.s['sgd_steps'],'note':'Uncommitted work after this checkpoint is not recoverable from the checkpoint alone. Supervisor interruption records distinguish clean pause from forced kill. Recomputed work is not new committed data. Completed-decision bound overlaps the search-simulation bound and must not be added as disjoint work. Initial-root expansion is separate from simulations; SGD bound applies only during training.','unsaved_search_simulations_upper_bound':min(self.env.config['runtime']['checkpoint_every_search_simulations'],self.env.config['search']['new_simulations']),'unsaved_initial_root_expansions_upper_bound':1,'unsaved_completed_decisions_upper_bound':1,'unsaved_sgd_steps_upper_bound':1})
            for group in self.optimizer.param_groups:
                if (group['lr'],group['momentum'],group['weight_decay'],group['dampening'],group['nesterov'],group['maximize'])!=(self.s['recipe']['lr'],self.s['recipe']['momentum'],0,0,False,False):raise ValueError('RESTORE_OPTIMIZER_RECIPE')
            if model_hash(self.model)!=saved['model_hash']:raise ValueError('RESTORE_MODEL_HASH')
            if self.s['model_id']!=self.s['models'][-1]['id'] or self.s['generation']!=self.s['models'][-1]['generation']:raise ValueError('RESTORE_MODEL_LEDGER')
            if self.s['trainer'] is None and saved['model_hash']!=self.s['models'][-1]['hash']:raise ValueError('RESTORE_PUBLISHED_MODEL_HASH')
            self.validate_pending_actor()
        self.model.eval()
    def should_stop(self):
        root=self.root.parent.parent if self.root.parent.name=='actors' else self.root
        return (root/'STOP').exists()
    def validate_pending_actor(self):
        a=self.s['actor']
        if a is None:return
        if a['model_id']!=self.s['model_id'] or a['weight_hash']!=self.s['models'][-1]['hash'] or a['root'].t!=len(a['records']):raise ValueError('RESTORE_ACTOR_MODEL_STEP')
        state=self.env.initial
        for t,r in enumerate(a['records']):
            validate_record(r,a['episode_id'],t,a['model_id'],self.env.schema_hash,self.env.config['search']['new_simulations'])
            if r['weight_hash']!=a['weight_hash']:raise ValueError('RESTORE_ACTOR_WEIGHT_BINDING')
            if self.env.state_id(state)!=r['state_id'] or self.env.observation_hash(self.env.observation.encode(state,t))!=r['observation_hash']:raise ValueError('RESTORE_PREFIX_STATE')
            if not np.array_equal(np.flatnonzero(self.env.mask(state)),r['legal_action_ids']):raise ValueError('RESTORE_PREFIX_LEGAL')
            state=self.env.step(state,r['selected_action_id'])['state']
        if self.env.state_id(state)!=self.env.state_id(a['root'].state):raise ValueError('RESTORE_NON_GOLDEN_ROOT')
        todo=[a['root']]
        while todo:
            node=todo.pop()
            if node.model_id and node.model_id!=a['model_id']:raise ValueError('RESTORE_STALE_TREE')
            todo.extend(node.children.values())
        d=a['decision']
        if d is not None:
            root=a['root'];obs_hash=self.env.observation_hash(self.env.observation.encode(root.state,root.t))
            if d['model_id']!=a['model_id'] or d['schema']!=self.env.schema_hash or d['root_observation']!=obs_hash or root.obs_hash!=obs_hash:raise ValueError('RESTORE_PENDING_BINDING')
            if not 0<=d['completed']<=self.env.config['search']['new_simulations'] or int((root.n-d['inherited_N']).sum())!=d['completed']:raise ValueError('RESTORE_PENDING_VISITS')
    def publish(self,initial=False):
        h=model_hash(self.model);mid=f"{self.s['run_id']}-N{self.s['generation']}"
        item={'id':mid,'hash':h,'parent':self.s['parent'],'generation':self.s['generation'],'seed':self.s['seed'],'recipe':copy.deepcopy(self.s['recipe']),'schema':self.env.schema_hash,'data_contract':self.env.config_hash,'positions':self.s['total_positions'],'sgd_steps':self.s['sgd_steps'],'initial_random':initial,'published_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'published_monotonic':time.monotonic()}
        item['compatibility']=self.env.compatibility;item['training_hash']=digest(self.s['recipe'])
        self.s['model_id']=mid;self.s['models'].append(item);self.pv_cache.clear()
        publication_started=time.perf_counter();publication_io=atomic_torch(self.root/(mid+'.pt'),{'model':self.model.state_dict(),'identity':item})
        self.s.setdefault('publication_profile',[]).append({'model_id':mid,'total_seconds':time.perf_counter()-publication_started,**publication_io})
        atomic_json(self.root/'models.json',self.s['models'])
    def save(self):
        save_started=time.monotonic();cpu=time.process_time();self.s['stats']['process_cpu_seconds']=self.s['stats'].get('process_cpu_seconds',0.)+cpu-self.last_cpu;self.last_cpu=cpu
        now=time.monotonic();delta=now-self.last_saved;self.s['elapsed_seconds']+=delta;self.s['budget']['active_seconds']+=delta;self.s['budget']['wall_elapsed_seconds']=time.time()-utc_seconds(self.s['budget']['session_start_utc']);self.s['budget']['remaining_seconds']=max(0,utc_seconds(self.s['budget']['deadline_utc'])-time.time());self.last_saved=now
        checkpoint_io=atomic_torch(self.root/'checkpoint.pt',{'state':self.s,'model':self.model.state_dict(),'model_hash':model_hash(self.model),'optimizer':self.optimizer.state_dict(),'rng':rng_state(),'training_hash':digest(self.s['recipe'])})
        self.s['stats']['checkpoint_writes']=self.s['stats'].get('checkpoint_writes',0)+1;self.s['stats']['checkpoint_seconds']=self.s['stats'].get('checkpoint_seconds',0.)+time.monotonic()-save_started
        self.status()
        checkpoint_event={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic':time.monotonic(),'event':'checkpoint','total_seconds':time.monotonic()-save_started,'total_simulations':self.s['total_simulations'],'total_decisions':self.s['total_decisions'],'sgd_steps':self.s['sgd_steps'],**checkpoint_io}
        with (self.root/'checkpoint-events.jsonl').open('a') as f:f.write(json.dumps(checkpoint_event)+'\n');f.flush();os.fsync(f.fileno())
    def status(self):
        actor=self.s['actor'];trainer=self.s['trainer']
        out={k:self.s[k] for k in ('run_id','seed','generation','model_id','next_episode','episodes','new_positions','total_positions','total_decisions','total_simulations','sgd_steps','elapsed_seconds','resume_count','budget','stats')}
        out['actor']=None if actor is None else {'episode_id':actor['episode_id'],'step':actor['root'].t,'model_id':actor['model_id'],'pending_simulations':actor['decision']['completed'] if actor['decision'] else 0,'task':actor['root'].state['task'],'hero':actor['root'].state['hero']}
        out['trainer']=trainer;out['rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(self.root/'status.json',out)
    def new_actor(self):
        eid=self.s['next_episode'];self.s['next_episode']+=1
        noise_seed=self.s['seed']*100000+eid*2;action_seed=noise_seed+1
        self.s['actor']={'episode_id':eid,'model_id':self.s['model_id'],'weight_hash':self.s['models'][-1]['hash'],'root':Node(self.env.initial,0),'decision':None,'records':[],'noise_rng':np.random.default_rng(noise_seed),'action_rng':np.random.default_rng(action_seed),'noise_seed':noise_seed,'action_seed':action_seed,'search_stats':{},'floors':set(),'started_elapsed':self.s['elapsed_seconds']}
    def finish_episode(self):
        a=self.s['actor'];terminal=self.env.terminal(a['root'].state)
        if terminal is None:raise ValueError('NOT_TERMINAL')
        ep={'origin':'P43_actual_episode','episode_id':a['episode_id'],'run_id':self.s['run_id'],'model_id':a['model_id'],'weight_hash':a['weight_hash'],'T':'T1.0','A':self.env.config['algorithm'],'config_hash':self.env.config_hash,'schema':self.env.schema_hash,'initial_sha':self.env.identity['initialSHA256'],'evaluation':False,'noise_seed':a['noise_seed'],'action_seed':a['action_seed'],'records':a['records'],'end':terminal['end'],'task_hp':terminal['task_hp'],'z':terminal['z'],'terminal':terminal,'terminal_state_id':self.env.state_id(a['root'].state),'terminal_state':self.env.dynamic(a['root'].state),'search_stats':a['search_stats'],'floors':sorted(a['floors']),'terminal_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminal_monotonic':time.monotonic()}
        validate_episode(self.env,ep)
        name=f"episode-{a['episode_id']:06d}.json.gz"
        path=self.root/name;tmp=path.with_suffix('.tmp')
        with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(ep,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        os.replace(tmp,path)
        if any(x['episode_id']==ep['episode_id'] for x in self.s['episodes']):raise ValueError('DUPLICATE_EPISODE')
        brief={k:ep[k] for k in ('episode_id','model_id','weight_hash','end','task_hp','z','terminal','floors','search_stats','terminal_utc','terminal_monotonic')};brief.update(path=name,positions=len(ep['records']),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.s['cohort'].append(brief);self.s['actor']=None;self.save()
        print(json.dumps({'event':'episode_terminal',**brief},ensure_ascii=False),flush=True)
    def commit_cohort(self):
        if len(self.s['cohort'])!=self.s['recipe']['logical_cohort']:raise ValueError('COHORT_NOT_FOUR')
        for ep in sorted(self.s['cohort'],key=lambda x:x['episode_id']):
            if any(x['episode_id']==ep['episode_id'] for x in self.s['episodes']):raise ValueError('DUPLICATE_COMMIT')
            self.s['episodes'].append(ep)
            self.s['replay'].extend((ep['episode_id'],i) for i in range(ep['positions']))
            self.s['new_positions']+=ep['positions'];self.s['total_positions']+=ep['positions']
        self.s['replay']=self.s['replay'][-8192:];self.s['cohort']=[];self.save()
    def collect(self,deadline,max_cohorts=0,no_train=False,until_decisions=None,until_simulations=None,until_generation=None):
        deadline=cap_deadline(deadline,self.s['budget'])
        search=None;search_actor=None;cohorts=0
        if 'first_collection_utc' not in self.s:
            self.s['first_collection_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();self.s['first_collection_monotonic']=time.monotonic()
        while time.monotonic()<deadline and not self.should_stop():
            if until_generation is not None and self.s['generation']>=until_generation:break
            if until_decisions is not None and self.s['total_decisions']>=until_decisions:break
            if until_simulations is not None and self.s['total_simulations']>=until_simulations:break
            if self.s['trainer'] is not None:
                self.train(deadline)
                if self.s['trainer'] is not None:break
                search=None
                if until_generation is not None and self.s['generation']>=until_generation:break
            if len(self.s['cohort'])==self.s['recipe']['logical_cohort']:
                self.commit_cohort();cohorts+=1
                if not no_train and self.s['new_positions']>=self.s['recipe']['min_new_positions']:
                    self.begin_training();continue
                if max_cohorts and cohorts>=max_cohorts:break
            if self.s['actor'] is None:self.new_actor();self.save()
            a=self.s['actor']
            if search_actor!=a['episode_id'] or search is None:
                search=Search(self.env,self.model,a['model_id'],self.env.config['search'],cache=self.pv_cache);search.stats.update(a['search_stats']);search_actor=a['episode_id']
            a['floors'].add(int(a['root'].state['hero']['floorId'][2:]))
            if self.env.terminal(a['root'].state) is not None:self.finish_episode();search=None;continue
            if a['decision'] is None:a['decision']=search.begin(a['root'],a['noise_rng'])
            d=a['decision']
            while d['completed']<d['budget'] and time.monotonic()<deadline and not self.should_stop():
                if until_simulations is not None and self.s['total_simulations']>=until_simulations:break
                search.simulate(a['root'],d);self.s['total_simulations']+=1
                if d['completed']%self.env.config['runtime']['checkpoint_every_search_simulations']==0:
                    a['search_stats']=dict(search.stats);search.assert_counters();self.save()
            # Mandatory even on a partial return: never lose current cost counters.
            a['search_stats']=dict(search.stats);search.assert_counters()
            if d['completed']<d['budget']:break
            before=a['root'].state;child,r=search.finish(a['root'],d,a['action_rng'])
            r.update(origin='P43_actual_episode',episode_id=a['episode_id'],state=self.env.dynamic(before),diagnostics=self.env.diagnostics(before,r['legal_action_ids']))
            if r['weight_hash']!=a['weight_hash']:raise ValueError('ACTOR_WEIGHT_HASH')
            a['records'].append(r);a['root']=child;a['decision']=None;a['search_stats']=dict(search.stats);self.s['total_decisions']+=1
            self.save()
            print(json.dumps({'event':'decision','episode_id':a['episode_id'],'model_id':a['model_id'],'weight_hash':a['weight_hash'],'step':r['step'],'action':r['selected_action_id'],'kind':self.env.kind(before,r['selected_action_id']),'legal':len(r['legal_action_ids']),'simulations':self.s['total_simulations'],'search_cost':r['search_cost']},ensure_ascii=False),flush=True)
        if search is not None and self.s['actor'] is not None:
            self.s['actor']['search_stats']=dict(search.stats);search.assert_counters()
        self.save()
    def load_episode(self,eid):
        path=self.root/f'episode-{eid:06d}.json.gz'
        brief=next((e for e in self.s['episodes']+self.s['cohort'] if e['episode_id']==eid),None)
        if brief is None or hashlib.sha256(path.read_bytes()).hexdigest()!=brief['sha256']:raise ValueError('EPISODE_ARTIFACT_HASH')
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    def fingerprint(self):
        """Logical recovery fingerprint; actual costs/timestamps are separate evidence."""
        actor=self.s['actor'];rng=rng_state()
        if actor is not None:rng.update(actor_noise=actor['noise_rng'].bit_generator.state,actor_action=actor['action_rng'].bit_generator.state)
        def semantic(ep):
            ep=copy.deepcopy(ep)
            for k in ('search_stats','terminal_utc','terminal_monotonic'):ep.pop(k,None)
            for r in ep.get('records',[]):r.pop('search_cost',None)
            return ep
        def tree(node):
            if node is None:return None
            return {'state':self.env.dynamic(node.state),'t':node.t,'ids':node.ids,'prior':node.prior,'n':node.n,'w':node.w,'terminal':node.terminal,'value':node.value,'obs_hash':node.obs_hash,'model_id':node.model_id,'children':{k:tree(v) for k,v in sorted(node.children.items())}}
        committed=[semantic(self.load_episode(ep['episode_id'])) for ep in self.s['episodes']]
        cohort=[semantic(self.load_episode(ep['episode_id'])) for ep in self.s['cohort']]
        decision=None if actor is None or actor['decision'] is None else {k:v for k,v in actor['decision'].items() if k!='stats_before'}
        trainer=None if self.s['trainer'] is None else {k:v for k,v in self.s['trainer'].items() if k not in ('profile','started_utc','started_monotonic','finished_utc','finished_monotonic')}
        return {'model':model_hash(self.model),'optimizer':structure_hash(self.optimizer.state_dict()),'rng':structure_hash(rng),'committed_samples':structure_hash(committed),'cohort_samples':structure_hash(cohort),'replay':structure_hash(self.s['replay']),'pending_records':structure_hash(semantic({'records':actor['records']}) if actor else []),'pending_tree':structure_hash(tree(actor['root']) if actor else None),'pending_decision':structure_hash(decision),'trainer':structure_hash(trainer),'logical':{'generation':self.s['generation'],'sgd_steps':self.s['sgd_steps'],'decisions':self.s['total_decisions'],'simulations':self.s['total_simulations'],'pending_step':actor['root'].t if actor else None,'pending_simulations':actor['decision']['completed'] if actor and actor['decision'] else 0}}
    def begin_training(self):
        if self.s['actor'] is not None or self.s['cohort']:raise ValueError('TRAINING_DURING_COHORT')
        n=self.s['new_positions']
        if n<self.s['recipe']['min_new_positions']:raise ValueError('TRAINING_TRIGGER')
        for ep in self.s['episodes']:
            validate_episode(self.env,self.load_episode(ep['episode_id']))
        self.s['recipe_frozen']=True
        self.s['trainer']={'new_positions':n,'steps_target':math.ceil(8*n/self.s['recipe']['batch']),'steps_done':0,'parent':self.s['model_id'],'parent_hash':model_hash(self.model),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_monotonic':time.monotonic(),'losses':[],'data_gradients':[],'before_group_hashes':self.parameter_hashes(),'profile':{'sample_load_encode_seconds':0.,'forward_backward_step_seconds':0.}};self.save()
        if self.s['generation']==0 and not (self.root/'pre-sgd-N0.pt').exists():
            witness_started=time.perf_counter();target=self.root/'pre-sgd-N0.pt';tmp=target.with_suffix('.tmp');shutil.copyfile(self.root/'checkpoint.pt',tmp)
            with tmp.open('rb') as f:os.fsync(f.fileno())
            os.replace(tmp,target)
            fd=os.open(self.root,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            atomic_json(self.root/'pre-sgd-N0.json',{'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'positions':self.s['total_positions'],'episodes':[e['episode_id'] for e in self.s['episodes']],'sgd_steps':0,'generation':0,'copy_seconds':time.perf_counter()-witness_started,'budget':self.s['budget']})
    def parameter_hashes(self):
        return {group:hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for name,p in self.model.named_parameters() if name.startswith(group))).hexdigest() for group in ('body','policy','value')}
    def train(self,deadline,steps_limit=None):
        deadline=cap_deadline(deadline,self.s['budget'])
        trainer=self.s['trainer']
        if trainer is None:self.begin_training();trainer=self.s['trainer']
        cache={};self.model.train();ran=0
        while trainer['steps_done']<trainer['steps_target'] and time.monotonic()<deadline and not self.should_stop():
            sample_started=time.monotonic()
            pairs=[self.s['replay'][int(i)] for i in np.random.randint(0,len(self.s['replay']),size=self.s['recipe']['batch'])]
            xs=[];masks=[];targets=[];zs=[]
            for eid,t in pairs:
                if eid not in cache:cache[eid]=self.load_episode(eid)
                ep=cache[eid];r=ep['records'][t];state=self.env.restore(r['state']);xs.append(self.env.observation.encode(state,t))
                mask=np.zeros(713,bool);mask[r['legal_action_ids']]=True;masks.append(mask)
                pi=np.zeros(713,np.float32);pi[r['legal_action_ids']]=r['pi'];targets.append(pi);zs.append(ep['z'])
            x=torch.from_numpy(np.stack(xs));mask=torch.from_numpy(np.stack(masks));pi=torch.from_numpy(np.stack(targets));z=torch.tensor(zs,dtype=torch.float32)
            trainer['profile']['sample_load_encode_seconds']+=time.monotonic()-sample_started;step_started=time.monotonic()
            self.optimizer.zero_grad(set_to_none=True);logits,value=self.model(x)
            lp=torch.log_softmax(logits.masked_fill(~mask,-1e9),dim=1);ce=-(pi*lp).sum(1).mean();mse=(value-z).square().mean();data=ce+mse
            data.backward(retain_graph=True);g=gradients(self.model)
            if not all(math.isfinite(v) for v in g.values()):raise ValueError('NONFINITE_DATA_GRADIENT '+repr(g))
            regularizer=1e-4*manual_l2(self.model);regularizer.backward();self.optimizer.step()
            trainer['profile']['forward_backward_step_seconds']+=time.monotonic()-step_started
            if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('NONFINITE_UPDATE')
            trainer['steps_done']+=1;self.s['sgd_steps']+=1;ran+=1
            trainer['losses'].append({'ce':float(ce.detach()),'mse':float(mse.detach()),'manual_l2':float(regularizer.detach()),'sample_pairs':pairs});trainer['data_gradients'].append(g)
            self.save();print(json.dumps({'event':'sgd','generation':self.s['generation'],'done':trainer['steps_done'],'target':trainer['steps_target'],'gradients':g,'loss':trainer['losses'][-1]}),flush=True)
            if steps_limit and ran>=steps_limit:break
        self.model.eval()
        if trainer['steps_done']==trainer['steps_target']:
            if not all(any(row[k]>0 for row in trainer['data_gradients']) for k in ('body','policy','value')):raise ValueError('MISSING_EFFECTIVE_DATA_GRADIENT')
            trainer['after_group_hashes']=self.parameter_hashes()
            if not all(trainer['before_group_hashes'][k]!=trainer['after_group_hashes'][k] for k in trainer['before_group_hashes']):raise ValueError('NO_GROUP_PARAMETER_CHANGE')
            trainer['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();trainer['finished_monotonic']=time.monotonic();parent=self.s['model_id'];self.s['parent']=parent;self.s['generation']+=1;self.s['new_positions']=0
            atomic_json(self.root/f"training-N{self.s['generation']}.json",trainer);self.s['trainer']=None;self.publish();self.save()
            print(json.dumps({'event':'model_ready','model_id':self.s['model_id'],'weight_hash':self.s['models'][-1]['hash'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic':time.monotonic(),'positions':self.s['total_positions'],'sgd_steps':self.s['sgd_steps']}),flush=True)
        else:self.save()
