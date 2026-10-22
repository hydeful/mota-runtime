"""Independent schemas for same-CNN map and serialized object-topology inputs."""
import json
from pathlib import Path
import numpy as np
from tasks.tower10.neural.common import digest
from tasks.tower10.p49.learning.environment import Environment as MapEnvironment
from tasks.tower10.p49.topology.environment import Environment as TopologyEnvironment
from .observation import Observation,SLICES

HERE=Path(__file__).resolve().parent
WHITELIST={'task','condition','input_encoding','environment_representation'}


def make_config(condition,seed=49001,phase='online'):
    cfg=json.loads((HERE/f'config_{condition}.json').read_text())
    cfg['network']['initialization_seed']=seed;cfg['experiment_phase']=phase
    return cfg


def assert_configs(a,b):
    differences={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if differences!=WHITELIST:raise ValueError('UNCONTROLLED_CONFIGURATION_DIFFERENCES '+repr(differences))
    if sum(1 for _ in differences)!=4:raise ValueError('CONDITION_CONFIG')
    return sorted(differences)


class Environment:
    def __init__(self,config):
        self.config=config;self.condition=config['condition']
        self.inner=(MapEnvironment if self.condition=='A' else TopologyEnvironment)(config)
        self.codec=self.inner.codec;self.initial=self.inner.initial;self.h_bound=900
        if len(self.codec.actions)!=21:raise ValueError('ACTION_DICTIONARY')
        if self.condition=='B' and [x['target'] for x in self.codec.actions]!=[f'e{i:02d}' for i in range(21)]:raise ValueError('OBJECT_ORDER')
        self.observation=self.inner.observation if self.condition=='A' else Observation(self.inner.observation)
        self.identity=self.inner.identity
        self.schema_description={'task':config['task'],'phase':config['experiment_phase'],'condition':self.condition,
          'encoding':config['input_encoding'],'container':[34,5,9],'network':'exact original map CNN',
          'input_schema':self.inner.schema_description,'actions':self.codec.actions,'padding_actions':[21,713]}
        if self.condition=='B':self.schema_description.update(serialized_fields=SLICES,physical_spatial_axes=False)
        self.schema_hash=digest(self.schema_description)
        self.config_hash=digest({k:config[k] for k in ['task','condition','input_encoding','environment_representation','experiment_phase','algorithm','network','search','training','reward','evaluation']})
        self.compatibility=dict(self.inner.compatibility,task=config['task'],schema=self.schema_hash,
          recipe=self.config_hash,network='unchanged-map-CNN-123994',phase=config['experiment_phase'],condition=self.condition)
    def mask(self,s):
        mask=self.inner.mask(s)
        if len(mask)==21:mask=np.pad(mask,(0,692),constant_values=False)
        if mask[21:].any():raise ValueError('PADDING_ACTION')
        return mask
    def reference_mask(self,s):
        mask=self.inner.reference_mask(s)
        return np.pad(mask,(0,692),constant_values=False) if len(mask)==21 else mask
    def step(self,s,aid):return self.inner.step(s,aid)
    def terminal(self,s,mask=None):return self.inner.terminal(s,mask)
    def kind(self,s,aid):return self.inner.kind(s,aid)
    def diagnostics(self,s,ids):return self.inner.diagnostics(s,ids)
    def state_id(self,s):return self.inner.state_id(s)
    def dynamic(self,s):return self.inner.dynamic(s)
    def restore(self,s):return self.inner.restore(s)
    def observation_hash(self,x):return self.inner.observation_hash(x)
