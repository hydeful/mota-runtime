"""Single-player mean-Q PUCT with bounded exact-observation raw P/V cache."""
import math,time
from collections import OrderedDict
from dataclasses import dataclass,field
import numpy as np
import torch
from .network import model_hash

@dataclass
class Node:
    state:dict
    t:int
    ids:object=None
    prior:object=None
    n:object=None
    w:object=None
    children:dict=field(default_factory=dict)
    terminal:object=None
    value:float=0.
    obs_hash:str=''
    model_id:str=''

class Search:
    def __init__(self,env,model,model_id,config,cache=None):
        self.env=env;self.model=model;self.model_id=model_id;self.config=config
        self.weight_hash=model_hash(model);self.cache=cache if cache is not None else OrderedDict();self.cache_limit=config.get('pv_cache_entries',20000)
        self.stats={'new_simulations':0,'forwards':0,'expanded':0,'terminal_backups':0,'root_expansions':0,'encoding_seconds':0.,'network_seconds':0.,'rule_seconds':0.,'traversal_backup_seconds':0.,'cache_seconds':0.,'max_depth':0,'pv_cache_hits':0,'cache_evictions':0}
        model.eval()
    def expand(self,node):
        if node.model_id and node.model_id!=self.model_id:raise ValueError('TREE_MODEL_MISMATCH')
        node.model_id=self.model_id
        began=time.perf_counter();mask=self.env.mask(node.state);self.stats['rule_seconds']+=time.perf_counter()-began
        began=time.perf_counter();terminal=self.env.terminal(node.state,mask);self.stats['rule_seconds']+=time.perf_counter()-began
        if terminal is not None:
            node.terminal=terminal;node.value=terminal['z'];self.stats['terminal_backups']+=1;return node.value
        began=time.perf_counter();obs=self.env.observation.encode(node.state,node.t)
        node.obs_hash=self.env.observation_hash(obs);self.stats['encoding_seconds']+=time.perf_counter()-began
        key=(self.weight_hash,self.env.schema_hash,node.obs_hash)
        began=time.perf_counter();cached=self.cache.get(key) if self.cache_limit else None;self.stats['cache_seconds']+=time.perf_counter()-began
        if cached is None:
            began=time.perf_counter()
            with torch.inference_mode():logits,v=self.model(torch.from_numpy(obs[None]))
            raw_logits=logits[0].numpy().copy();value=float(v)
            self.stats['network_seconds']+=time.perf_counter()-began;self.stats['forwards']+=1
            if not np.isfinite(raw_logits).all() or not math.isfinite(value):raise ValueError('NONFINITE_NETWORK')
            if self.cache_limit:
                began=time.perf_counter();self.cache[key]=(raw_logits,value)
                if len(self.cache)>self.cache_limit:self.cache.popitem(last=False);self.stats['cache_evictions']+=1
                self.stats['cache_seconds']+=time.perf_counter()-began
        else:
            began=time.perf_counter();raw_logits,value=cached;self.cache.move_to_end(key);self.stats['pv_cache_hits']+=1;self.stats['cache_seconds']+=time.perf_counter()-began
        self.stats['expanded']+=1
        ids=np.flatnonzero(mask);raw=raw_logits[ids].astype(np.float64);raw-=raw.max();p=np.exp(raw);p/=p.sum()
        if not np.isfinite(p).all() or not math.isfinite(value):raise ValueError('NONFINITE_NETWORK')
        node.ids=ids;node.prior=p;node.n=np.zeros(len(ids),np.int64);node.w=np.zeros(len(ids),np.float64);node.value=value
        return node.value
    def begin(self,root,rng,evaluation=False):
        if root.model_id and root.model_id!=self.model_id:raise ValueError('TREE_MODEL_MISMATCH')
        before=dict(self.stats)
        if root.ids is None and root.terminal is None:
            self.expand(root);self.stats['root_expansions']+=1
        if root.terminal is not None:raise ValueError('DECISION_AT_TERMINAL')
        prior=root.prior.copy()
        if not evaluation:prior=(1-self.config['root_noise_mix'])*prior+self.config['root_noise_mix']*rng.dirichlet(np.full(len(prior),self.config['root_dirichlet_alpha']))
        budget=800 if evaluation else self.config['new_simulations']
        return {'model_id':self.model_id,'weight_hash':self.weight_hash,'schema':self.env.schema_hash,'inherited_N':root.n.copy(),'root_prior':prior,'completed':0,'budget':budget,'evaluation':evaluation,'root_observation':root.obs_hash,'stats_before':before}
    def simulate(self,root,decision):
        if decision['model_id']!=self.model_id or decision['schema']!=self.env.schema_hash or decision['weight_hash']!=self.weight_hash:raise ValueError('SEARCH_MODEL_SCHEMA_MISMATCH')
        began=time.perf_counter();excluded_before=sum(self.stats[k] for k in ('rule_seconds','encoding_seconds','network_seconds','cache_seconds'))
        node=root;trail=[];depth=0
        while True:
            if node.terminal is not None:value=node.value;self.stats['terminal_backups']+=1;break
            if node.ids is None:value=self.expand(node);break
            p=decision['root_prior'] if node is root else node.prior
            q=np.divide(node.w,node.n,out=np.zeros_like(node.w),where=node.n>0)
            scores=q+self.config['cpuct']*p*math.sqrt(int(node.n.sum()))/(1+node.n)
            j=int(np.argmax(scores));trail.append((node,j));depth+=1
            if j not in node.children:
                began_rule=time.perf_counter();res=self.env.step(node.state,int(node.ids[j]));self.stats['rule_seconds']+=time.perf_counter()-began_rule
                node.children[j]=Node(res['state'],node.t+1)
            node=node.children[j]
        for parent,j in trail:parent.n[j]+=1;parent.w[j]+=value
        self.stats['new_simulations']+=1;self.stats['max_depth']=max(self.stats['max_depth'],depth);decision['completed']+=1
        excluded_after=sum(self.stats[k] for k in ('rule_seconds','encoding_seconds','network_seconds','cache_seconds'))
        self.stats['traversal_backup_seconds']+=max(0,time.perf_counter()-began-(excluded_after-excluded_before))
        return value
    def finish(self,root,decision,rng):
        budget=decision['budget']
        if decision['completed']!=budget or int((root.n-decision['inherited_N']).sum())!=budget:raise ValueError('INCOMPLETE_SEARCH')
        pi=root.n/root.n.sum()
        j=int(np.argmax(root.n)) if decision['evaluation'] or root.t>=self.config['sample_first_interactions'] else int(rng.choice(len(pi),p=pi))
        aid=int(root.ids[j]);record={'step':root.t,'model_id':self.model_id,'weight_hash':self.weight_hash,'schema':self.env.schema_hash,'observation_hash':root.obs_hash,'state_id':self.env.state_id(root.state),'legal_action_ids':root.ids.tolist(),'root_N':root.n.tolist(),'inherited_N':decision['inherited_N'].tolist(),'pi':pi.tolist(),'selected_action_id':aid,'new_simulations':decision['completed'],'prediction':root.value,'root_prior':decision['root_prior'].tolist(),'evaluation':decision['evaluation'],'search_cost':{k:self.stats[k]-decision['stats_before'].get(k,0) for k in self.stats if k!='max_depth'}}
        child=root.children.get(j)
        if child is None:raise ValueError('SELECTED_UNVISITED_ACTION')
        return child,record
    def assert_counters(self):
        if self.stats['expanded']!=self.stats['forwards']+self.stats['pv_cache_hits']:raise ValueError('FORWARD_CACHE_NONCONSERVATION')
        if self.stats['new_simulations']!=self.stats['expanded']+self.stats['terminal_backups']-self.stats['root_expansions']:raise ValueError('SEARCH_NONCONSERVATION')
