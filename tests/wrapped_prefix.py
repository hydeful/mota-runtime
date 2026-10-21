"""Check that deployment timing/storage wrappers preserve the P45 trajectory."""
import argparse, gzip, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--wrapped',required=True);p.add_argument('--output',required=True);a=p.parse_args()
original=Path(a.original);wrapped=Path(a.wrapped)
excluded={'run_id','model_id','search_cost','search_stats','terminal_utc','terminal_monotonic','parent','profile','started_utc','started_monotonic','finished_utc','finished_monotonic'}
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in excluded}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def digest(x):return hashlib.sha256(json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
positions=0;hashes=[]
for path in sorted(original.glob('episode-*.json.gz')):
    with gzip.open(path,'rt') as f:x=json.load(f)
    with gzip.open(wrapped/path.name,'rt') as f:y=json.load(f)
    assert clean(x)==clean(y),path.name
    positions+=len(x['records']);hashes.append({'episode_id':x['episode_id'],'positions':len(x['records']),'semantic_sha256':digest(x),'weight_hash':x['weight_hash'],'end':x['end'],'z':x['z']})
x=json.loads((original/'training-N1.json').read_text());y=json.loads((wrapped/'training-N1.json').read_text())
assert clean(x)==clean(y),'first real training update differs'
result={'pass':True,'positions':positions,'complete_episodes':len(hashes),'real_sgd_steps':x['steps_done'],'training_semantic_sha256':digest(x),'exact_comparison':'Full observations, actions, visits, pi, priors, predictions, terminal states/labels, training sampled pairs, loss/gradient and parameter-group hashes. Excludes only timing/cost fields and renamed run/model identities. Weight hashes are NOT excluded.','episodes':hashes}
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'wrapped_prefix_exact':True,'positions':positions,'episodes':len(hashes),'sgd_steps':x['steps_done']}))
