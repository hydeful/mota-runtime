"""16 policy samples, one argmax and one deterministic MCTS100; RNG isolated."""
import gzip,json,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .network import model_hash
from .runtime import load_model,rng_state,restore_rng,structure_hash
from .search import Search,Node,resources


def evaluate(env,path,out,deadline,heartbeat=None):
    state_rng=rng_state();rng_before=structure_hash(state_rng);out=Path(out);out.mkdir(exist_ok=True);rows=[]
    try:
        model,identity=load_model(path,env);assert not model.training
        for condition in env.config['evaluation']['conditions']:
            seeds=env.config['evaluation']['sample_seeds'] if condition=='direct_policy_sample' else [env.config['evaluation']['deterministic_seed']]
            dest=out/(identity['id']+'-'+condition+'.jsonl.gz');assert not dest.exists()
            with gzip.open(dest,'wt') as stream:
                for seed in seeds:
                    began=time.monotonic();state=env.initial;records=[];rng=np.random.default_rng(seed);search=None;root=None;forwards=0
                    if condition=='MCTS100_no_noise_maxN':
                        search=Search(env,model,identity['id'],env.config['search'],OrderedDict());root=Node(state,0)
                    while env.terminal(state) is None:
                        if time.monotonic()>=deadline:raise TimeoutError('SHARED_FORMAL_BUDGET')
                        if search:
                            decision=search.begin(root,rng,evaluation=True);assert decision['budget']==100
                            for _ in range(100):search.simulate(root,decision)
                            root,rec=search.finish(root,decision,rng);aid=rec['selected_action_id']
                        else:
                            obs=env.observation.encode(state,len(records));ids=np.flatnonzero(env.mask(state))
                            with torch.inference_mode():logits,value=model(torch.from_numpy(obs[None]))
                            raw=logits[0,ids].numpy().astype(np.float64);raw-=raw.max();p=np.exp(raw);p/=p.sum()
                            assert np.isfinite(p).all() and np.isfinite(float(value))
                            aid=int(rng.choice(ids,p=p)) if condition=='direct_policy_sample' else int(ids[np.argmax(p)])
                            forwards+=1
                            rec={'step':len(records),'legal_action_ids':ids.tolist(),'network_prior':p.tolist(),
                              'prediction':float(value),'selected_action_id':aid,'new_simulations':0,
                              'inherited_N':[],'added_N':[],'root_N':[],'action_Q':[],
                              'state_id':env.state_id(state),'observation_hash':env.observation_hash(obs)}
                        before=state;state=env.step(state,aid)['state']
                        if search:assert env.state_id(state)==env.state_id(root.state)
                        rec.update(model_id=identity['id'],weight_hash=identity['hash'],schema=env.schema_hash,
                          target=env.codec.actions[aid]['target'],resources_before=resources(before),resources_after=resources(state))
                        rec['resource_delta']={k:rec['resources_after'][k]-v for k,v in rec['resources_before'].items()}
                        records.append(rec);assert len(records)<=21
                    terminal=env.terminal(state)
                    if search:search.assert_counters();forwards=search.stats['forwards']
                    row={'evaluation':True,'condition':condition,'seed':seed,'model_id':identity['id'],
                      'weight_hash':identity['hash'],'schema':env.schema_hash,'outcome':'success' if terminal['end']=='GOAL' else 'legal_failure',
                      'terminal':terminal,'final_hp':state['hero']['hp'] if terminal['end']=='GOAL' else None,
                      'decisions':len(records),'new_simulations':search.stats['new_simulations'] if search else 0,
                      'network_forwards':forwards,'seconds':time.monotonic()-began,'records':records,'error':None}
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
            if heartbeat:heartbeat({'event':'evaluation_condition_complete','model':identity['id'],'condition':condition,'games':len(seeds)})
        assert model_hash(model)==identity['hash']
    finally:
        restore_rng(state_rng);assert structure_hash(rng_state())==rng_before
    return rows
