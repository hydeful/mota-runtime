"""Isolated fixed-seed evaluation. No acceptance answers or replay writes."""
import gzip
import json
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .network import model_hash
from .runtime import load_model, rng_state, restore_rng
from .search import Search, Node


class BudgetStop(Exception):pass


def evaluate(env,path,out,deadline,heartbeat=None):
    saved_rng=rng_state();rows=[];out=Path(out);out.mkdir(exist_ok=True)
    try:
        model,identity=load_model(path,env)
        for condition in env.config['evaluation']['conditions']:
            dest=out/(identity['id']+'-'+condition+'.jsonl.gz')
            if dest.exists():raise ValueError('EVALUATION_ARTIFACT_EXISTS')
            with gzip.open(dest,'wt',encoding='utf-8') as stream:
                for episode_index,seed in enumerate(env.config['evaluation']['seeds']):
                    began=time.monotonic();rng=np.random.default_rng(seed);state=env.initial;records=[];forwards=0;simulations=0;outcome='unfinished';error=None
                    search=None;root=None
                    try:
                        if condition=='MCTS100_no_noise_maxN':
                            search=Search(env,model,identity['id'],env.config['search'],cache=OrderedDict());root=Node(state,0)
                        while env.terminal(state) is None:
                            if time.monotonic()>=deadline:raise BudgetStop('FORMAL_BUDGET')
                            if search:
                                decision=search.begin(root,rng,evaluation=True)
                                if decision['budget']!=100:raise ValueError('WRONG_EVALUATION_BUDGET')
                                for _ in range(100):
                                    if time.monotonic()>=deadline:raise BudgetStop('FORMAL_BUDGET')
                                    search.simulate(root,decision)
                                child,record=search.finish(root,decision,rng)
                                if record['new_simulations']!=100:raise ValueError('WRONG_EVALUATION_COUNT')
                                aid=record['selected_action_id'];root=child
                            else:
                                mask=env.mask(state);ids=np.flatnonzero(mask);obs=env.observation.encode(state,len(records))
                                with torch.inference_mode():logits,value=model(torch.from_numpy(obs[None]))
                                raw=logits[0,ids].numpy().astype(np.float64);raw-=raw.max();p=np.exp(raw);p/=p.sum()
                                if not np.isfinite(p).all() or not np.isfinite(float(value)):raise ValueError('NONFINITE_DIRECT_POLICY')
                                aid=int(rng.choice(ids,p=p));forwards+=1
                                record={'step':len(records),'legal_action_ids':ids.tolist(),'policy':p.tolist(),'prediction':float(value),'selected_action_id':aid,'new_simulations':0,'state_id':env.state_id(state),'observation_hash':env.observation_hash(obs)}
                            # Execute the actual game action too: search tree is not
                            # a substitute for the real environment trajectory.
                            before=state;result=env.step(state,aid);state=result['state']
                            if search and env.state_id(state)!=env.state_id(root.state):raise ValueError('EVALUATION_TREE_REAL_STATE')
                            record.update(target=env.codec.actions[aid]['target'],hp_before=before['hero']['hp'],hp_after=state['hero']['hp'])
                            records.append(record);simulations+=record['new_simulations']
                            if len(records)>21:raise ValueError('P49_NON_PROGRESS')
                        terminal=env.terminal(state);outcome='success' if terminal['end']=='GOAL' else 'legal_failure'
                        if search:search.assert_counters();forwards=search.stats['forwards']
                    except BudgetStop as exc:
                        error=str(exc);outcome='unfinished'
                    except Exception as exc:
                        error=repr(exc);outcome='exception'
                    terminal=env.terminal(state)
                    row={'evaluation':True,'condition':condition,'episode_index':episode_index,'seed':seed,'model_id':identity['id'],'weight_hash':identity['hash'],'schema':env.schema_hash,'outcome':outcome,'terminal':terminal,'final_hp':state['hero']['hp'] if outcome=='success' else None,'decisions':len(records),'new_simulations':search.stats['new_simulations'] if search else simulations,'network_forwards':search.stats['forwards'] if search else forwards,'seconds':time.monotonic()-began,'records':records,'error':error}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
                    if heartbeat:heartbeat({k:v for k,v in row.items() if k!='records'})
                    if outcome in ('exception','unfinished'):raise BudgetStop('EVALUATION_'+outcome)
        if model_hash(model)!=identity['hash']:raise ValueError('EVALUATION_MUTATED_MODEL')
    finally:
        restore_rng(saved_rng)
    return rows
