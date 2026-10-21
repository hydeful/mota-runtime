"""Fail-closed P43 sample and terminal provenance validation."""
import math,re
import numpy as np

class ContractError(ValueError):pass
def require(ok,message):
    if not ok:raise ContractError(message)

def validate_record(r,episode,step,model_id,schema,expected_simulations=100):
    require(r['origin']=='P43_actual_episode','ORIGIN')
    require(r['episode_id']==episode,'EPISODE_BINDING')
    require(type(r['step']) is int and r['step']==step and type(step) is int and step>=0,'STEP')
    require(r['model_id']==model_id,'MODEL_BINDING')
    require(r['schema']==schema,'SCHEMA_BINDING')
    require(r['evaluation'] is False,'EVALUATION_ISOLATION')
    ids=r['legal_action_ids'];n=r['root_N'];inh=r['inherited_N'];pi=r['pi']
    require(len(ids)>0 and len(ids)==len(n)==len(inh)==len(pi),'ACTION_SHAPE')
    require(len(set(ids))==len(ids) and all(type(x) is int and 0<=x<713 for x in ids),'ACTION_IDS')
    require(all(type(x) is int and x>=0 for x in n+inh),'VISITS')
    require(all(a>=b for a,b in zip(n,inh)),'INHERITED')
    require(type(r['new_simulations']) is int and r['new_simulations']==expected_simulations and sum(n)-sum(inh)==expected_simulations,'NEW_SIMULATIONS')
    require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in pi),'PI_FINITE_POSITIVE')
    require(abs(sum(pi)-1)<=1e-6 and max(abs(p-k/sum(n)) for p,k in zip(pi,n))<=1e-6,'PI_N')
    require(type(r['selected_action_id']) is int and r['selected_action_id'] in ids,'SELECTED_ILLEGAL')
    require(n[ids.index(r['selected_action_id'])]>0,'SELECTED_UNVISITED')
    if step>=30:require(r['selected_action_id']==ids[int(np.argmax(n))],'POST30_NOT_MAX_N')
    require(isinstance(r['observation_hash'],str) and re.fullmatch('[0-9a-f]{64}',r['observation_hash']) is not None,'OBSERVATION_HASH')
    require(isinstance(r['state_id'],str) and re.fullmatch('[0-9a-f]{64}',r['state_id']) is not None,'STATE_HASH')
    require(math.isfinite(r['prediction']) and -1<=r['prediction']<=1,'PREDICTION')
    require(len(r['root_prior'])==len(ids) and all(math.isfinite(x) and x>=0 for x in r['root_prior']) and abs(sum(r['root_prior'])-1)<=1e-6,'ROOT_PRIOR')
    return True

def validate_episode(env,episode,reference=False):
    require(episode['origin']=='P43_actual_episode','EPISODE_ORIGIN')
    require(not episode['evaluation'],'EVALUATION_NOT_TRAINING')
    require(episode['config_hash']==env.config_hash,'CONFIG_BINDING')
    require(episode['T']=='T1.0' and episode['A']==env.config['algorithm'],'VERSIONS')
    require(episode['initial_sha']==env.identity['initialSHA256'],'NON_GOLDEN')
    state=env.initial;records=episode['records'];require(len(records)>0,'EMPTY_EPISODE')
    for t,r in enumerate(records):
        require(env.terminal(state) is None,'POST_GOAL_OR_FAILURE_ACTION')
        validate_record(r,episode['episode_id'],t,episode['model_id'],env.schema_hash,env.config['search']['new_simulations'])
        require(r['weight_hash']==episode['weight_hash'],'WEIGHT_BINDING')
        mask=env.reference_mask(state) if reference else env.mask(state)
        require(np.array_equal(np.flatnonzero(mask),r['legal_action_ids']),'LEGAL_SET')
        require(env.state_id(state)==r['state_id'],'STATE_REPLAY')
        require(env.observation_hash(env.observation.encode(state,t))==r['observation_hash'],'OBS_REPLAY')
        require(env.state_id(env.restore(r['state']))==r['state_id'],'STORED_STATE')
        state=env.step(state,r['selected_action_id'])['state']
    terminal=env.terminal(state)
    require(terminal is not None,'PENDING_NOT_TERMINAL')
    require(episode['end']==terminal['end'],'TERMINAL_END')
    require(episode['terminal_state_id']==env.state_id(state),'TERMINAL_REPLAY')
    require(episode['task_hp']==terminal['task_hp'],'TASK_HP')
    require(math.isfinite(episode['z']) and abs(episode['z']-terminal['z'])<=1e-9,'WRONG_EPISODE_Z')
    require(episode['terminal']==terminal,'TERMINAL_PROOF')
    return True
