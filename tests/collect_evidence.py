"""Audit a real run without loading or modifying its model/checkpoint."""
import argparse, datetime, gzip, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--output', required=True)
a=p.parse_args();root=Path(a.run)
def read(p):return json.loads(p.read_text())
m=read(root/'run.json');s=read(root/'state/status.json');c=read(root/'control.json');models=read(root/'state/models.json')
window=read(root/'state/p48-window.json')
start=datetime.datetime.fromisoformat(window['started_utc'])
cutoff=start+datetime.timedelta(seconds=window['requested_window_seconds'])
checkpoints=[json.loads(x) for x in (root/'state/checkpoint-events.jsonl').read_text().splitlines()]
inside=[x for x in checkpoints if start<=datetime.datetime.fromisoformat(x['utc'])<=cutoff]
episodes=[]
for path in sorted((root/'state').glob('episode-*.json.gz')):
    with gzip.open(path,'rt') as f:ep=json.load(f)
    terminal=datetime.datetime.fromisoformat(ep['terminal_utc'])
    saved=[x for x in inside if datetime.datetime.fromisoformat(x['utc'])>=terminal]
    episodes.append({'episode_id':ep['episode_id'],'model_id':ep['model_id'],'weight_hash':ep['weight_hash'],'positions':len(ep['records']),'end':ep['end'],'z':ep['z'],'terminal_utc':ep['terminal_utc'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'terminal_and_later_save_in_window':start<=terminal<=cutoff and bool(saved)})
updated=[e for e in episodes if any(e['model_id']==x['id'] and x['generation']>0 and e['weight_hash']==x['hash'] for x in models)]
for ep in updated:
    ep['model_binding_verified']=True
result={'source_commit':m['source_commit'],'baseline_commit':'63bdd80216edf825d887bc5316eb0ad78e7250ad','environment':'local managed environment; NOT AutoDL','compute_device':'cpu','seed':m['seed'],'authorized_global_budget_seconds':m['budget_seconds'],'deadline_utc':m['deadline_utc'],'invocation':c,'window':window,'durable_at_return':{'committed_complete_episodes':len(s['episodes']),'complete_episode_artifacts':len(episodes),'qualified_training_positions':s['total_positions'],'model_updates':s['generation'],'real_sgd_steps':s['sgd_steps'],'decisions':s['total_decisions'],'elapsed_core_seconds':s['elapsed_seconds']},'strict_window':{'complete_episodes_with_later_checkpoint_in_window':sum(x['terminal_and_later_save_in_window'] for x in episodes),'last_checkpoint_real_sgd_steps':inside[-1]['sgd_steps'] if inside else 0,'qualification_note':'qualified positions/model update counts above are durable-return counts; original checkpoint journal lacks per-save position/generation fields, so no exact parent-ledger cutoff count is inferred'},'model_chain':[{'id':x['id'],'generation':x['generation'],'hash':x['hash'],'parent':x['parent'],'initial_random':x['initial_random'],'positions':x['positions'],'sgd_steps':x['sgd_steps'],'published_utc':x['published_utc']} for x in models],'updated_model_complete_episode_consumption':updated,'episodes':episodes,'limitations':['No AutoDL output or terminal-browser closure evidence','No claim of learning improvement','No one-hour extrapolation is a measurement']}
assert c['core_returncode']==0 and c['state'] in ['slice_finished','budget_finished'],c
assert 599.9<=window['requested_window_seconds']<=600.1,window
assert len(episodes)>=1 and s['total_positions']>=1 and s['generation']>=1 and s['sgd_steps']>=1
assert updated,'no complete subsequent episode consumed an updated model'
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed_local_600':True,**result['durable_at_return'],'updated_model_consuming_episodes':len(updated),'collection_call_seconds':window['collection_call_seconds'],'post_cutoff_cleanup_seconds':window['post_cutoff_cleanup_seconds']}))
