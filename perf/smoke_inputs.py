"""Hash-bound, locally validated smoke artifacts; never a production replay source."""
import hashlib
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
        x=load(root/'inputs.json');assert x and x['validated_before_use'] and x['files']
        for name,h in x['files'].items():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts
            assert digest(root/p)==h
        return x
    except (OSError,AssertionError,KeyError,TypeError):return None

def seal_reference(out,inputs):
    path=out/'cpu-reference.npz'
    x={'basis_id':inputs['basis_id'],'sha256':digest(path),'file':str(path),'CPU_numeric_gate_passed':True}
    dump(out/'reference.json',x);return x

def valid_reference(out,inputs):
    try:
        x=load(out/'reference.json');assert x['basis_id']==inputs['basis_id'] and x['CPU_numeric_gate_passed']
        assert digest(out/'cpu-reference.npz')==x['sha256'];return out/'cpu-reference.npz'
    except (OSError,AssertionError,KeyError,TypeError):return None
