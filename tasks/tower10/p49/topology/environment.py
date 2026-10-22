"""Search, rules, legal mask and identities consume only object-topology state."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.neural.common import canonical,digest
from .kernel import Session
from .observation import Observation

HERE=Path(__file__).resolve().parent


def make_config():
    cfg=json.loads((HERE.parent.parent/'p45/config.json').read_text())
    cfg.update(stage='P49-object-topology',task='p49-object-topology-learning-v1',engineering='P49-object-topology-v1')
    cfg['topology']={'version':'object-topology-v1','runtime_nodes':'events/objects only','scope':'frozen P49, no corridor event programs'}
    cfg['network']={'blocks':2,'channels':32,'node_features':27,'relations':2,'global_features':7,'policy':21,'value':'tanh','initialization_seed':49001,'architecture':'shared node policy, two relation message blocks, mean-pooled value'}
    cfg['search']['evaluation_new_simulations']=100
    cfg['training']['min_new_positions']=64
    cfg['reward']={'success':'terminal_hp/900','failure':-1,'H_bound':900,'gamma':1,'shaping':False}
    cfg['evaluation']={'episodes_per_condition':16,'conditions':['direct_policy_sample','MCTS100_no_noise_maxN'],'seeds':list(range(49201,49217)),'enters_training':False}
    cfg['runtime']['checkpoint_every_search_simulations']=100
    cfg['budget']={'budget_seconds':3600}
    return cfg


class Codec:
    def __init__(self,initial):
        self.actions=[{'target':k} for k in initial['graph']['order']]
        self.action_map={canonical(a):i for i,a in enumerate(self.actions)}
    def action_id(self,action):return self.action_map[canonical(action)]


class Environment:
    def __init__(self,config=None):
        self.config=make_config() if config is None else config
        initial=json.loads((HERE/'initial.json').read_text());identity=json.loads((HERE/'identity.json').read_text())
        if digest(initial)!=identity['topology_initial_sha256']:raise ValueError('TOPOLOGY_INITIAL_IDENTITY')
        if digest(initial['rules'])!=identity['topology_rules_sha256']:raise ValueError('TOPOLOGY_RULES_IDENTITY')
        self.session=Session(initial);self.initial=self.session.initial;self.codec=Codec(self.initial);self.h_bound=900
        self.observation=Observation(self.initial)
        self.identity={'initialSHA256':digest(initial),'rulesSHA256':digest(initial['rules']),'physical_initial_sha256':identity['physical_initial_sha256']}
        self.schema_description={'task':self.config['task'],'nodes':self.initial['graph']['order'],'node_features':self.observation.channels,'relations':['direct_corridor','destination_resolved'],'global':['hp/900','atk/20','def/20','yellowKey/4','money/100','t/21','goal'],'tensor_shapes':{'nodes':[21,27],'relations':[2,21,21],'hero':[7]},'serialization_envelope':[21,76],'actions':self.codec.actions,'initial':self.identity,'coordinate_input':False}
        self.schema_hash=digest(self.schema_description)
        self.config_hash=digest({k:self.config[k] for k in ['task','algorithm','network','search','training','reward','evaluation']})
        self.compatibility={'task':self.config['task'],'schema':self.schema_hash,'initial':self.identity['initialSHA256'],'rules':self.identity['rulesSHA256'],'recipe':self.config_hash,'actions':digest(self.codec.actions),'network':'object-relations-2x32-P21-meanV256-v1','torch':str(torch.__version__)}

    def mask(self,s):
        mask=np.zeros(len(self.codec.actions),bool)
        for a in self.session.legal(s):mask[self.codec.action_id(a)]=True
        return mask
    reference_mask=mask
    def step(self,s,aid):
        if type(aid) is not int or not 0<=aid<21 or not self.mask(s)[aid]:raise ValueError('TOPOLOGY_ILLEGAL_ACTION')
        return self.session.step(s,self.codec.actions[aid])
    def terminal(self,s,mask=None):
        if s['task']['status']=='GOAL':
            if s['objects'][s['task']['boss']]['status']!='resolved' or s['hero']['hp']<=0:raise ValueError('TOPOLOGY_INCOMPLETE_GOAL')
            return {'end':'GOAL','task_hp':s['hero']['hp'],'z':s['hero']['hp']/900,'proof':{'atomic_settled':True,'post_goal_actions':0,'representation':'objects'}}
        mask=self.mask(s) if mask is None else mask
        if mask.any():return None
        return {'end':'T1_NO_INTERACTION','task_hp':None,'z':-1,'proof':{'topology_legal_empty':True}}
    def kind(self,s,aid):return {'battle':'battle','getItem':'pickup','openDoor':'door'}[s['objects'][self.codec.actions[int(aid)]['target']]['trigger']]
    def diagnostics(self,s,ids):return {'legal_count':len(ids),'representation':'object-topology'}
    @staticmethod
    def dynamic(s):return {k:v for k,v in s.items() if k not in ('rules','graph')}
    def state_id(self,s):return digest(self.dynamic(s))
    @staticmethod
    def observation_hash(obs):return hashlib.sha256(np.asarray(obs,dtype=np.float32).tobytes()).hexdigest()
    def restore(self,d):
        if 'rules' in d or 'graph' in d:raise ValueError('TOPOLOGY_DYNAMIC_STATIC_INJECTION')
        return dict(d,graph=self.initial['graph'],rules=self.initial['rules'])
