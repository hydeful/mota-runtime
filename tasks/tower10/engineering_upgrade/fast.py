"""Pure E variants preserving T1 and frozen network feature meanings."""
import hashlib,math
from collections import OrderedDict
import numpy as np
from tasks.tower10.neural.common import Codec,canonical,dynamic
from .topology import InteractionMask
from .actions import interaction_kind

class FastMask(InteractionMask):
    """Same indexed BFS/path order; omit construction of already-masked IDs."""
    def __call__(self,s):
        mask=np.zeros(len(self.codec.actions),bool)
        if s['task']['status']=='GOAL':return mask
        frontiers=self.paths(s);self.ns['paths']=lambda _:frontiers
        for k in sorted(frontiers):
            o=s['objects'][k]
            cs=[[3],[0,3],[1,3],[2,3]] if k=='o:MT4:6:1' else [[0],[1]] if o['trigger']=='trader' and not s['flags'].get(k[2:].replace(':','@')+'@A') else [[0]] if o['trigger']=='oldman' and o['position'][0]=='MT3' else [None]
            for choice in cs:
                a={'target':k}
                if choice is not None:a['choices']=choice
                if interaction_kind(s,a) is not None and self.ns['refusal'](s,k,choice) is None:
                    mask[self.codec.action_id(a)]=True
        return mask

class FastCodec(Codec):
    """Bounded token-address memo; accumulation order/dtype unchanged."""
    def __init__(self,initial):
        super().__init__(initial);self.tokens=OrderedDict();self.token_limit=32768
    def encode(self,s):
        x=np.zeros(self.n_features,np.float32);h=s['hero'];scales=[1000,100,100,100,100,100,10,12,12]
        x[:len(self.scalars)]=[h.get(k,0)/v for k,v in zip(self.scalars,scales)]
        off=len(self.scalars);x[off+int(h['floorId'][2:])-1]=1;off+=10
        x[off+['up','right','down','left'].index(h['direction'])]=1;off+=4
        for i,k in enumerate(self.slots):
            o=s['objects'].get(k);status=o['status'] if o else 'absent'
            x[off+5*i+['absent','active','hidden','resolved','moved'].index(status)]=1
        off+=5*len(self.slots)
        def emit(token,value):
            address=self.tokens.get(token)
            if address is None:
                n=int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),'little')
                address=(n%512,1 if n>>63 else -1);self.tokens[token]=address
                if len(self.tokens)>self.token_limit:self.tokens.popitem(last=False)
            x[off+address[0]]+=value*address[1]
        def visit(v,path):
            if isinstance(v,dict):
                for k in sorted(v):visit(v[k],path+'/'+str(k))
            elif isinstance(v,list):
                for i,z in enumerate(v):visit(z,path+'/'+str(i))
                if not v:emit(path+'=[]',1.)
            elif type(v) in (int,float):emit(path,math.copysign(math.log1p(abs(v)),v)/10)
            else:emit(path+'='+canonical(v),1.)
        d=dynamic(s)
        d['objects']={k:{j:v for j,v in o.items() if self.initial['objects'].get(k,{}).get(j)!=v} for k,o in s['objects'].items()}
        visit(d,'');x[off:]/=8
        return x
