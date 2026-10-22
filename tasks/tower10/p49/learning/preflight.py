"""Bounded normal-prefix interface checks and one untrained timing game."""
import copy
import datetime
import json
import time
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.neural.vendor import topology_engine as reference
from .environment import Environment, make_config
from .network import model_hash
from .runtime import Runner, atomic_json, load_model, seed_all
from .search import Search, Node


def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)


def run(root):
    setup();root=Path(root);root.mkdir(parents=True,exist_ok=False)
    env=Environment();config=env.config
    now=datetime.datetime.now(datetime.timezone.utc)
    config['budget'].update(session_start_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(hours=1)).isoformat())
    runner=Runner(root/'checkpoint',env,create=True,seed=49001,run_id='P49-interface-NO-TRAIN')
    restored=Runner(root/'checkpoint',env,create=False)
    assert model_hash(runner.model)==model_hash(restored.model)
    model,identity=load_model(root/'checkpoint'/f"{runner.s['model_id']}.pt",env)
    mismatches=[]
    for key in ['task','schema']:
        bad=copy.copy(env);bad.compatibility=dict(env.compatibility);bad.compatibility[key]='intentional-mismatch'
        try:load_model(root/'checkpoint'/f"{runner.s['model_id']}.pt",bad)
        except ValueError as e:mismatches.append(str(e))
        else:raise AssertionError('MISMATCH_ACCEPTED')
    checks=[];s=env.initial
    assert not env.mask(s)[env.codec.action_id({'target':'o:MT1:4:1'})]
    blocked=env.step(s,env.codec.action_id({'target':'o:MT1:5:2'}))['state']
    blocked=env.step(blocked,env.codec.action_id({'target':'o:MT1:8:2'}))['state']
    assert reference.refusal(blocked,'o:MT1:8:1')=='WOULD_DIE'
    assert not env.mask(blocked)[env.codec.action_id({'target':'o:MT1:8:1'})]
    # A normal reachable prefix selected solely to exercise interfaces, never
    # written as training data or used as an action preference by the learner.
    prefix=['o:MT1:3:3','o:MT1:6:4','o:MT1:7:4','o:MT1:8:4','o:MT1:3:2','o:MT1:2:4','o:MT1:1:4','o:MT1:0:4','o:MT1:4:1']
    idx=env.observation.index
    for k in prefix:
        before=s;obs=env.observation.encode(s,s['actionCount'])
        mask=env.mask(s);assert np.array_equal(mask,env.reference_mask(s))
        assert not mask[21:].any();assert obs.shape==(34,5,9)
        for v in s['graph']['nodes'].values():
            _,x,y=v['position'];assert obs[idx['walkable'],y,x]==1
        for o in s['objects'].values():
            _,x,y=o['position'];assert obs[idx['type:'+o['event']['id']],y,x]==1
            assert obs[idx['status:'+o['status']],y,x]==1
        aid=env.codec.action_id({'target':k});assert env.codec.actions[aid]=={'target':k}
        r=env.step(s,aid);ref=env.tower.step(s,{'target':k},independent=True)
        assert r==ref;s=r['state'];new=env.observation.encode(s,s['actionCount']);assert not np.array_equal(obs,new)
        item=before['objects'][k]['event']['id']
        if item=='shield1':
            assert s['hero']['def']==20 and before['hero']['def']==10
            assert reference.damage(s,'greenSlime')==reference.damage(s,'redSlime')==0
        if item=='sword1':assert before['hero']['atk']==10 and s['hero']['atk']==20
        if item=='yellowKey':assert s['hero']['items']['tools'].get('yellowKey',0)==before['hero']['items']['tools'].get('yellowKey',0)+1
        if item=='yellowDoor':assert s['hero']['items']['tools'].get('yellowKey',0)==before['hero']['items']['tools'].get('yellowKey',0)-1
        assert s['objects'][k]['status']=='resolved'
        checks.append({'target':k,'id':item,'hp_before':before['hero']['hp'],'hp_after':s['hero']['hp'],'atk':s['hero']['atk'],'def':s['hero']['def'],'keys':s['hero']['items']['tools'].get('yellowKey',0),'state_id':env.state_id(s)})
    assert env.terminal(s)['z']==s['hero']['hp']/900
    assert not env.mask(s).any()
    try:env.step(s,env.codec.action_id({'target':prefix[-1]}))
    except ValueError:pass
    else:raise AssertionError('POST_GOAL_ACTION')
    assert env.h_bound==900
    # Time a full actual N0+MCTS100 episode. No SGD, no score-based selection,
    # and no artifact from this probe is entered into the formal replay.
    seed_all(49001);root_node=Node(env.initial,0);search=Search(env,model,'preflight-random-N0',env.config['search']);rng=np.random.default_rng(49901)
    began=time.monotonic();decisions=0
    while env.terminal(root_node.state) is None:
        d=search.begin(root_node,rng,evaluation=True)
        assert d['budget']==100
        for _ in range(d['budget']):search.simulate(root_node,d)
        root_node,r=search.finish(root_node,d,rng);assert r['new_simulations']==100;decisions+=1
    search.assert_counters();elapsed=time.monotonic()-began
    result={'passed':True,'H_bound':900,'shape':[34,5,9],'action_count':21,'output':713,'permanently_masked_padding':692,'prefix_checks':checks,'normal_checkpoint_reload':True,'task_schema_mismatch_rejections':mismatches,'timing_probe':{'seconds':elapsed,'decisions':decisions,'new_simulations':search.stats['new_simulations'],'forwards':search.stats['forwards'],'estimated_96_MCTS_evaluation_games_seconds':96*elapsed},'torch':str(torch.__version__),'SGD_steps':0}
    atomic_json(root/'interface.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
