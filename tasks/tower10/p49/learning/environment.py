"""Independent P49 learner identity. Never instantiates or disables P45 guards."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.p49.environment import Environment as Tower, HERE
from tasks.tower10.neural.common import canonical, digest, state_id, dynamic
from tasks.tower10.neural.vendor import topology_engine as reference
from tasks.tower10.engineering_upgrade.actions import interaction_kind
from tasks.tower10.engineering_upgrade.fast import FastMask
from tasks.tower10.engineering_upgrade.topology import InteractionMask
from .observation import Observation


def make_config():
    original=json.loads((HERE.parent/'p45/config.json').read_text())
    original.update(stage='P49',task='p49-handdrawn-learning-v1',engineering='P49-binding-v1')
    original['network'].update(input=[34,5,9],initialization_seed=49001)
    original['search']['evaluation_new_simulations']=100
    original['training']['min_new_positions']=64
    original['reward']={'success':'terminal_hp/H_bound','failure':-1,'gamma':1,'shaping':False,'bound_derivation':'initial HP plus all positive HP pickups'}
    original['evaluation']={'episodes_per_condition':16,'conditions':['direct_policy_sample','MCTS100_no_noise_maxN'],'seeds':list(range(49201,49217)),'enters_training':False}
    original['runtime']['checkpoint_every_search_simulations']=100
    original['budget']={'budget_seconds':3600}
    return original


class Codec:
    def __init__(self, initial):
        self.actions=[{'target':k} for k in sorted(initial['objects'])]
        self.action_map={canonical(a):i for i,a in enumerate(self.actions)}
    def action_id(self, a):return self.action_map[canonical(a)]


class Environment:
    def __init__(self, config=None):
        self.config=make_config() if config is None else config
        source=json.loads((HERE/'initial_topology.json').read_text())
        pinned=json.loads((HERE/'identity.json').read_text())
        if digest(source)!=pinned['initial_sha256'] or digest(source['rules'])!=pinned['rules_sha256']:raise ValueError('P49_FROZEN_MAP_IDENTITY')
        self.tower=Tower(source);self.initial=self.tower.initial
        self.codec=Codec(self.initial)
        self.h_bound=self.initial['hero']['hp']+sum(self.initial['rules']['values'][o['event']['id']]*self.initial['rules']['ratios'][o['position'][0]] for o in self.initial['objects'].values() if o['trigger']=='getItem' and o['event']['id'] in ('redPotion','bluePotion'))
        self.observation=Observation(self.initial,self.h_bound)
        self.fast_mask=FastMask(self.codec,self.initial);self.slow_mask=InteractionMask(self.codec,self.initial)
        self.identity={'initialSHA256':digest(source),'rulesSHA256':digest(source['rules']),'engineSHA256':pinned['engine_sha256']}
        self.schema_description={'task':self.config['task'],'shape':[34,5,9],'channels':self.observation.channels,'actions':self.codec.actions,'padding':[21,713],'map':self.identity,'H_bound':self.h_bound}
        self.schema_hash=digest(self.schema_description)
        self.config_hash=digest({k:self.config[k] for k in ('task','algorithm','search','reward','network','training','evaluation')})
        self.compatibility={'task':self.config['task'],'initial':self.identity['initialSHA256'],'rules':self.identity['rulesSHA256'],'schema':self.schema_hash,'actions':digest(self.codec.actions),'recipe':self.config_hash,'network':'P49-stem34-grid5x9-body2x32-P713-V256-tanh-BN','engine':self.identity['engineSHA256'],'torch':str(torch.__version__)}

    def mask(self,s):return np.pad(self.fast_mask(s),(0,713-len(self.codec.actions)),constant_values=False)
    def reference_mask(self,s):return np.pad(self.slow_mask(s),(0,713-len(self.codec.actions)),constant_values=False)
    def terminal(self,s,mask=None):
        if s['task']['status']=='GOAL':
            assert s['objects'][self.tower.boss]['status']=='resolved' and s['hero']['hp']>0
            return {'end':'GOAL','task_hp':s['hero']['hp'],'z':s['hero']['hp']/self.h_bound,'proof':{'atomic_settled':True,'post_goal_actions':0}}
        mask=self.mask(s) if mask is None else mask
        if mask.any():return None
        if self.reference_mask(s).any():raise ValueError('P49_INCOMPLETE_MASK')
        return {'end':'T1_NO_INTERACTION','task_hp':None,'z':-1,'proof':{'reference_T1_empty':True}}
    def step(self,s,aid):
        if type(aid) is not int or not 0<=aid<len(self.codec.actions):raise ValueError('P49_PADDING_OR_INVALID_ACTION')
        if not self.mask(s)[aid]:raise ValueError('P49_ILLEGAL_ACTION')
        return self.tower.step(s,self.codec.actions[aid])
    def kind(self,s,aid):return interaction_kind(s,self.codec.actions[int(aid)])
    def diagnostics(self,s,ids):return {'legal_count':len(ids),'task':'P49'}
    state_id=staticmethod(state_id)
    dynamic=staticmethod(dynamic)
    @staticmethod
    def observation_hash(x):return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
    def restore(self,d):
        if 'rules' in d or 'graph' in d:raise ValueError('DYNAMIC_INJECT_STATIC')
        return dict(d,graph=self.initial['graph'],rules=self.initial['rules'])
