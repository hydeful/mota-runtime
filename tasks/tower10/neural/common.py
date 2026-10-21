"""Pinned topology context, conservative state identity and V1 encodings."""
from __future__ import annotations
import hashlib, importlib, json, math, zipfile
from pathlib import Path
import numpy as np
from .vendor import topology_engine as reference
from .vendor.kernel import prepare

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'tasks/tower10/datasets/neural_bootstrap_v1'
CLEAN = DATA / 'clean_v1'
REPORT = ROOT / '20_handoff/neural_bootstrap_20261016'
SOURCE_COMMIT = '0b877d49d6793306c85d5ad65459d376fe0b26ed'
ZIP_SHA = '857862e0858d578219b0f6a5e46d1ed22ab96eb9b910ecd925786e82b95e1339'
ENCODING = 'topology-current-v1'
ACTION_ENCODING = 'engine-primitives-v1'

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, x):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def source():
    p = DATA/'source/MOTA_training_inventory_20261016.zip'
    assert sha(p) == ZIP_SHA, 'source archive checksum mismatch'
    with zipfile.ZipFile(p) as z:
        files = {n:z.read(n) for n in z.namelist()}
    for n,h in json.loads(files['SHA256.json']).items():
        assert hashlib.sha256(files[n]).hexdigest() == h, n
    assert hashlib.sha256(files['topology_engine.py']).hexdigest() == sha(reference.__file__)
    s=json.loads(files['shared_initial_topology.json']); identity=json.loads(files['shared_identity.json'])
    assert digest(s['rules']) == identity['rulesSHA256']
    return files,s,identity

def context():
    _,s,identity=source()
    return s,prepare(s),identity

def dynamic(s):
    # Only two counters excluded: read nowhere by the frozen rule functions.
    # All flags, direction, entities, serial, event and object metadata retained.
    h={k:v for k,v in s['hero'].items() if k!='steps'}
    return {k:(h if k=='hero' else v) for k,v in s.items()
            if k not in ('graph','rules','actionCount')}

def state_id(s):
    return digest(dynamic(s))

def family(src):
    if src.startswith('P28_V1_R1/') or src.startswith('d2_p28/runs/V1_R1/'):
        return 'P28_V1_R1'
    return '/'.join(src.split('/')[:3]) if src.startswith('d2_') else src.split('/')[0]

def split_for(f):
    return 'validation' if f=='d2_p24/delivery/V2' else 'test' if f=='d2_p28/delivery/V3' else 'train'

def normalized_action(a):
    # Engine defaults which are invariant for these named interfaces.
    a=dict(a)
    if a.get('target')=='o:MT4:6:1' and a.get('choices') is None:a['choices']=[3]
    return a

class Codec:
    def __init__(self, initial):
        self.slots=sorted(set(initial['objects']) |
                          {'o:'+':'.join(map(str,n['position'])) for n in initial['graph']['nodes'].values()})
        actions=[]
        for k in self.slots:
            actions.append({'target':k})
            obj=initial['objects'].get(k)
            if obj and obj['trigger']=='trader':actions.extend({'target':k,'choices':[c]} for c in (0,1))
            if obj and obj['trigger']=='oldman' and obj['position'][0]=='MT3':actions.append({'target':k,'choices':[0]})
        actions.extend({'target':'o:MT4:6:1','choices':c} for c in ([3],[0,3],[1,3],[2,3]))
        actions.extend({'target':'hero','fly':f} for f in sorted(initial['rules']['fly']['floors']))
        self.actions=sorted({canonical(normalized_action(a)):normalized_action(a) for a in actions}.values(),key=canonical)
        self.action_map={canonical(a):i for i,a in enumerate(self.actions)}
        self.scalars=['hp','atk','def','money','mdef','mana','lv','x','y']
        self.n_features=len(self.scalars)+10+4+5*len(self.slots)+512
        self.initial=initial
        self.description={'encoding':ENCODING,'actions':ACTION_ENCODING,'slots':self.slots,
                          'action_vocabulary':self.actions,'features':self.n_features,
                          'scalars':self.scalars,'hash_bins':512}
        self.hash=digest(self.description)
    def action_id(self,a):
        return self.action_map[canonical(normalized_action(a))]
    def encode(self,s):
        x=np.zeros(self.n_features,np.float32);h=s['hero']
        scales=[1000,100,100,100,100,100,10,12,12]
        x[:len(self.scalars)]=[h.get(k,0)/v for k,v in zip(self.scalars,scales)]
        off=len(self.scalars);x[off+int(h['floorId'][2:])-1]=1;off+=10
        x[off+['up','right','down','left'].index(h['direction'])]=1;off+=4
        for i,k in enumerate(self.slots):
            o=s['objects'].get(k);status=o['status'] if o else 'absent'
            x[off+5*i+['absent','active','hidden','resolved','moved'].index(status)]=1
        off+=5*len(self.slots)
        # Entire dynamic information is represented, including unknown fields.
        # Hashing is a lossy feature representation, never used for state identity.
        def visit(v,path):
            if isinstance(v,dict):
                for k in sorted(v):visit(v[k],path+'/'+str(k))
            elif isinstance(v,list):
                for i,z in enumerate(v):visit(z,path+'/'+str(i))
                if not v:emit(path+'=[]',1.)
            elif type(v) in (int,float):emit(path,math.copysign(math.log1p(abs(v)),v)/10)
            else:emit(path+'='+canonical(v),1.)
        def emit(token,value):
            b=hashlib.blake2b(token.encode(),digest_size=8).digest();n=int.from_bytes(b,'little')
            x[off+n%512]+=value*(1 if n>>63 else -1)
        # Static per-object definitions shared; retain changing object/entity data.
        d=dynamic(s)
        d['objects']={k:{j:v for j,v in o.items() if self.initial['objects'].get(k,{}).get(j)!=v}
                      for k,o in s['objects'].items()}
        visit(d,'')
        x[off:]/=8
        return x
    def legal(self,s):
        acts=reference.legal_actions(s)
        mask=np.zeros(len(self.actions),bool)
        for a in acts:mask[self.action_id(a)]=True
        return mask

class RuleMask:
    """Exact legal_actions with one unchanged BFS result reused within a call.

    Private namespace only, no shared engine globals or transitions modified.
    """
    def __init__(self, codec, initial):
        from .vendor.path_index import PathIndex
        self.codec=codec;self.paths=PathIndex(initial['graph']).paths
        self.ns={'__name__':'pinned_mask'}
        exec(compile(Path(reference.__file__).read_text(),reference.__file__,'exec'),self.ns)
    def __call__(self,s):
        frontiers=self.paths(s)
        self.ns['paths']=lambda _:frontiers
        acts=self.ns['legal_actions'](s)
        mask=np.zeros(len(self.codec.actions),bool)
        for a in acts:mask[self.codec.action_id(a)]=True
        return mask


def validate_clean():
    manifest=json.loads((CLEAN/'SHA256.json').read_text())
    for name,h in manifest.items():
        if sha(CLEAN/name)!=h:raise ValueError('CLEAN_DATA_CHECKSUM '+name)
    return sha(CLEAN/'SHA256.json')
