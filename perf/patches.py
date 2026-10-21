"""Opt-in engineering patches; original22 runtime files remain unchanged."""
import hashlib
import io
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import torch

class PackedState:
    # P46 mechanism retained with the same file+directory fsync boundaries.
    # On load, only built-in pickle.loads is needed; no perf module is imported.
    def __init__(self, state): self.data = pickle.dumps(state, protocol=5)
    def __reduce__(self): return pickle.loads, (self.data,)

def packed_save(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); began=time.perf_counter()
    packed=isinstance(value,dict) and 'state' in value
    if packed: value=dict(value,state=PackedState(value['state']))
    buffer=io.BytesIO(); torch.save(value,buffer,pickle_protocol=5 if packed else 2)
    serialization=time.perf_counter()-began; began=time.perf_counter()
    with tmp.open('wb') as f:
        f.write(buffer.getbuffer()); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'serialization_seconds':serialization,'write_sync_seconds':time.perf_counter()-began,'bytes':buffer.tell()}

def buffered_dump(obj, fp, *args, **kwargs): fp.write(json.dumps(obj,*args,**kwargs))

def direct_hash(x):
    x=np.asarray(x,dtype=np.float32)
    return hashlib.sha256(memoryview(x).cast('B') if x.flags.c_contiguous else x.tobytes()).hexdigest()

class Profile:
    """Exclusive synchronous scopes. Unclassified wall time is not reassigned."""
    def __init__(self): self.stack=[]; self.buckets={}; self.calls={}
    def wrap(self, fn, name):
        def measured(*a, **kw):
            row=[time.perf_counter(),0.]; self.stack.append(row)
            try: return fn(*a,**kw)
            finally:
                elapsed=time.perf_counter()-row[0]; self.stack.pop()
                if self.stack: self.stack[-1][1]+=elapsed
                self.buckets[name]=self.buckets.get(name,0.)+max(0,elapsed-row[1])
                self.calls[name]=self.calls.get(name,0)+1
        return measured

def install(options, env):
    from tasks.tower10.p45 import runtime, parallel
    if options.get('packed'):
        runtime.atomic_torch=packed_save; parallel.atomic_torch=packed_save
        runtime.json.dump=buffered_dump
    if options.get('direct_hash'): env.observation_hash=direct_hash
    profile=Profile()
    for name in ('step','terminal'):
        setattr(env,name,profile.wrap(getattr(env,name),'rules'))
    env.mask=profile.wrap(env.mask,'rules')
    env.observation.encode=profile.wrap(env.observation.encode,'encoding')
    env.observation_hash=profile.wrap(env.observation_hash,'hash')
    return profile

def instrument_runner(runner, profile):
    for name,bucket in [('save','save'),('publish','publication'),('load_episode','sample_read'),
                        ('train','train_other'),('finish_episode','episode_finalize'),
                        ('commit_cohort','cohort_commit')]:
        setattr(runner,name,profile.wrap(getattr(runner,name),bucket))
    runner.model.forward=profile.wrap(runner.model.forward,'network_forward')
