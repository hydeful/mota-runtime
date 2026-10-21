"""Hash-bound, locally validated smoke artifacts; never a production replay source."""
import hashlib
import math
from pathlib import Path
from perf.common import dump,load

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seal(root,provenance):
    paths=[root/'pending.pt',root/'fixture/checkpoint.pt',root/'fixture/status.json',root/'fixture/recipe.json']
    paths+=sorted((root/'fixture').glob('episode-*.json.gz'))+sorted((root/'fixture').glob('*-N0.pt'))
    files={str(p.relative_to(root)):digest(p) for p in paths}
    value={'validated_before_use':True,'provenance':provenance,'files':files,'basis_id':hashlib.sha256(str(sorted(files.items())).encode()).hexdigest()}
    dump(root/'inputs.json',value);return value

def valid(root):
    try:
        x=load(root/'inputs.json');assert isinstance(x,dict) and x['validated_before_use'] is True and isinstance(x['files'],dict) and x['files']
        assert isinstance(x['provenance'],dict) and x['provenance'].get('formal_productivity') is False
        assert x['basis_id']==hashlib.sha256(str(sorted(x['files'].items())).encode()).hexdigest()
        for name,h in x['files'].items():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts
            assert digest(root/p)==h
        return x
    except (OSError,AssertionError,KeyError,TypeError,ValueError):return None

def seal_reference(out,inputs):
    path=out/'cpu-reference.npz'
    x={'basis_id':inputs['basis_id'],'sha256':digest(path),'file':str(path),'CPU_numeric_gate_passed':True}
    dump(out/'reference.json',x);return x

def valid_reference(out,inputs):
    try:
        x=load(out/'reference.json');assert x['basis_id']==inputs['basis_id'] and x['CPU_numeric_gate_passed']
        assert digest(out/'cpu-reference.npz')==x['sha256'];return out/'cpu-reference.npz'
    except (OSError,AssertionError,KeyError,TypeError,ValueError):return None


def candidate_micro(path):
    """Invalid/partial metadata is a failed input, never an exception bypassing fallback."""
    try:
        x=load(path);assert isinstance(x,dict)
        assert x['hash_correctness']['status']=='pass' and x['correctness']['cpu1']['pass'] is True
        assert x['serialization']['original']['status']=='pass'
        for name in ('copy_sha256','memoryview_sha256'):
            values=x['hash'][name];assert isinstance(values,list) and values
            assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values)
        return x
    except (OSError,ValueError,KeyError,TypeError,AssertionError):return {}
