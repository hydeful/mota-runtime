"""Two 32-wide relation-message blocks; shared node policy, invariant value."""
import math
import torch
from torch import nn
from tasks.tower10.p45.network import model_hash,manual_l2
from tasks.tower10.p45.network import gradients as original_gradients
from .observation import NODE_FEATURES,RELATIONS,GLOBAL_FEATURES

audit_sink=None


class RelationBlock(nn.Module):
    def __init__(self):
        super().__init__();self.mix=nn.Sequential(nn.Linear(32*(1+RELATIONS),32),nn.ReLU(),nn.Linear(32,32));self.norm=nn.LayerNorm(32)
    def forward(self,h,relations):
        messages=[]
        for relation in relations.unbind(1):
            normalized=relation/relation.sum(-1,keepdim=True).clamp_min(1)
            messages.append(torch.bmm(normalized,h))
        return torch.relu(self.norm(h+self.mix(torch.cat([h,*messages],dim=-1))))


class Network(nn.Module):
    def __init__(self,config=None):
        super().__init__();config=config or {'blocks':2,'channels':32}
        assert config['blocks']==2 and config['channels']==32
        self.body=nn.ModuleDict({'nodes':nn.Linear(NODE_FEATURES,32),'hero':nn.Linear(GLOBAL_FEATURES,32),'blocks':nn.ModuleList([RelationBlock(),RelationBlock()])})
        self.policy=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
        self.value=nn.Sequential(nn.Linear(64,256),nn.ReLU(),nn.Linear(256,1),nn.Tanh())

    @staticmethod
    def unpack(packed):
        batch,n,_=packed.shape
        nodes=packed[:,:,:NODE_FEATURES]
        relations=packed[:,:,NODE_FEATURES:NODE_FEATURES+RELATIONS*n].reshape(batch,n,RELATIONS,n).permute(0,2,1,3)
        hero=packed[:,:,-GLOBAL_FEATURES:].mean(1)
        return nodes,relations,hero

    def forward(self,packed):return self.relation_forward(*self.unpack(packed))

    def relation_forward(self,nodes,relations,hero):
        context=torch.relu(self.body['hero'](hero));h=torch.relu(self.body['nodes'](nodes)+context[:,None,:])
        for block in self.body['blocks']:h=block(h,relations)
        return self.policy(h).squeeze(-1),self.value(torch.cat([h.mean(1),context],dim=-1)).flatten()


def gradients(model):
    values=original_gradients(model)
    if not all(math.isfinite(v) and v>0 for v in values.values()):raise ValueError('TOPOLOGY_DATA_GRADIENT_PATH '+repr(values))
    if audit_sink:audit_sink({'event':'data_gradients','values':values})
    return values
