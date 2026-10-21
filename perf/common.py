import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

SOURCE = Path(__file__).resolve().parents[1]
BASELINE = 'f0825385c6bb5914b653679d14635a2b1393cf87'
TOLERANCES = {'inference_atol': 1e-5, 'inference_rtol': 1e-4,
              'sgd_atol': 1e-5, 'sgd_rtol': 1e-3}

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def stamp(value): return dt.datetime.fromisoformat(value).timestamp()
def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def event(path, **value):
    with Path(path).open('a') as f: f.write(json.dumps({'utc':utc(), **value})+'\n')
def verify_source():
    manifest = load(SOURCE/'SOURCE_MANIFEST.json')
    for item in manifest['files']:
        p = SOURCE/item['destination']
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('FROZEN_SOURCE_CHANGED '+str(p))
    return len(manifest['files'])
def configure():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
def budget(config, deadline):
    import copy
    c = copy.deepcopy(config)
    c['budget'] = {'session_start_utc': utc(),
                   'deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(),
                   'budget_seconds': max(0, deadline-time.time()),
                   'authorization': 'P48 performance experiment; subordinate to immutable3600s controller deadline'}
    return c
