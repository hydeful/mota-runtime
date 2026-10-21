"""Read-only, exact checkpoint comparator. Unknown types fail closed with a path."""
import dataclasses
import math
import random
from collections import OrderedDict
import numpy as np
import torch
from tasks.tower10.p45.search import Node

class Difference(AssertionError):
    pass

def compare(a,b,path='$'):
    def fail(why):raise Difference(path+': '+why)
    if type(a) is not type(b):fail('type '+type(a).__module__+'.'+type(a).__name__+' != '+type(b).__module__+'.'+type(b).__name__)
    if a is None:return
    if isinstance(a,(bool,int,str,bytes)):
        if a!=b:fail('value differs')
    elif isinstance(a,float):
        if a!=b and not (math.isnan(a) and math.isnan(b)):fail('float differs')
    elif isinstance(a,np.generic):
        if a.dtype!=b.dtype or a.tobytes()!=b.tobytes():fail('numpy scalar differs')
    elif isinstance(a,np.ndarray):
        if a.dtype.hasobject:fail('unsupported object array')
        if a.dtype!=b.dtype:fail('array dtype differs')
        if a.shape!=b.shape:fail('array shape differs')
        if a.flags.writeable!=b.flags.writeable:fail('array writeable differs')
        if a.tobytes()!=b.tobytes():fail('array data differs')
    elif isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or a.layout!=b.layout or a.device!=b.device or a.requires_grad!=b.requires_grad:fail('tensor metadata differs')
        if a.layout!=torch.strided:fail('unsupported tensor layout')
        if a.detach().cpu().contiguous().numpy().tobytes()!=b.detach().cpu().contiguous().numpy().tobytes():fail('tensor data differs')
    elif isinstance(a,np.random.Generator):
        compare(a.bit_generator,b.bit_generator,path+'.bit_generator')
    elif isinstance(a,np.random.BitGenerator):
        compare(a.state,b.state,path+'.state')
    elif isinstance(a,np.random.RandomState):compare(a.get_state(),b.get_state(),path+'.get_state')
    elif isinstance(a,random.Random):compare(a.getstate(),b.getstate(),path+'.getstate')
    elif isinstance(a,torch.Generator):
        if a.device!=b.device:fail('generator device differs')
        compare(a.get_state(),b.get_state(),path+'.get_state')
    elif isinstance(a,(dict,OrderedDict)):
        # Keys are strictly scalar; never order by object repr.
        def key_ok(k):return type(k) in (str,int,bool,bytes) or (type(k) is tuple and all(key_ok(x) for x in k))
        if any(not key_ok(k) for k in list(a)+list(b)):fail('unsupported mapping key')
        if a.keys()!=b.keys():fail('mapping keys differ')
        for k in a:compare(a[k],b[k],path+'['+repr(k)+']')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):fail('length differs')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
    elif isinstance(a,(set,frozenset)):
        if any(type(x) not in (int,str,bytes) for x in a|b):fail('unsupported set member')
        if a!=b:fail('set differs')
    elif type(a) is Node:
        expected={f.name for f in dataclasses.fields(Node)}
        if set(vars(a))!=expected or set(vars(b))!=expected:fail('unknown Node field')
        for f in dataclasses.fields(Node):compare(getattr(a,f.name),getattr(b,f.name),path+'.'+f.name)
    else:fail('UNKNOWN_OBJECT_TYPE '+type(a).__module__+'.'+type(a).__qualname__)

def report(a,b):
    try:compare(a,b);return {'status':'pass','exact':True}
    except Difference as exc:return {'status':'fail','first_difference':str(exc)}
