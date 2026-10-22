"""Only after equivalence certification: structured input, reload, timing."""
import copy
import datetime
import json
import time
from pathlib import Path
import numpy as np
import torch
from .environment import Environment,HERE
from .network import model_hash
from .runtime import Runner,load_model,atomic_json,BINDING_IDENTITY
from .search import Search,Node
from .observation import Observation


def run(root):
    certificate=json.loads((HERE/'CERTIFICATION.json').read_text());assert certificate['passed']
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(root);root.mkdir(exist_ok=False)
    env=Environment();now=datetime.datetime.now(datetime.timezone.utc);env.config['budget'].update(session_start_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(hours=1)).isoformat())
    runner=Runner(root/'checkpoint',env,create=True,seed=49001,run_id='P49-topology-preflight-NO-TRAIN')
    loaded=Runner(root/'checkpoint',env,create=False);assert model_hash(loaded.model)==model_hash(runner.model)
    model,identity=load_model(root/'checkpoint'/f"{runner.s['model_id']}.pt",env)
    rejections=[]
    for k in ('task','schema'):
        bad=copy.copy(env);bad.compatibility=dict(env.compatibility);bad.compatibility[k]='map-version-mismatch'
        try:load_model(root/'checkpoint'/f"{runner.s['model_id']}.pt",bad)
        except ValueError as exc:rejections.append(str(exc))
        else:raise ValueError('TOPOLOGY_WRONG_SCHEMA_ACCEPTED')
    # Reorder the actual graph's object index order, not just a printed tensor.
    permutation=np.random.default_rng(49911).permutation(21);reordered=copy.deepcopy(env.initial)
    reordered['graph']['order']=[env.initial['graph']['order'][i] for i in permutation]
    actual_reordered=Observation(reordered).encode(reordered,0)
    base=env.observation.encode(env.initial,0)
    assert np.array_equal(actual_reordered,env.observation.permute(base,permutation))
    with torch.inference_mode():
        p,v=model(torch.from_numpy(base[None]));pp,vv=model(torch.from_numpy(actual_reordered[None]))
    assert float((pp-p[:,permutation]).abs().max())<1e-6 and float((vv-v).abs().max())<1e-6
    root_node=Node(env.initial,0);search=Search(env,model,identity['id'],env.config['search']);rng=np.random.default_rng(49901);began=time.monotonic();records=[]
    while env.terminal(root_node.state) is None:
        d=search.begin(root_node,rng,evaluation=True)
        for _ in range(d['budget']):search.simulate(root_node,d)
        root_node,r=search.finish(root_node,d,rng);records.append(r)
    search.assert_counters();elapsed=time.monotonic()-began
    result={'passed':True,'normal_reload':True,'mismatch_rejections':rejections,'actual_graph_order_permutation':permutation.tolist(),'policy_equivariance_max_abs':float((pp-p[:,permutation]).abs().max()),'value_invariance_max_abs':float((vv-v).abs().max()),'tensor_shapes':{'nodes':[1,21,27],'relations':[1,2,21,21],'hero':[1,7],'policy':[1,21],'value':[1]},'parameters':sum(p.numel() for p in model.parameters()),'runtime_binding':BINDING_IDENTITY,'seconds_per_full_MCTS100_game':elapsed,'decisions':len(records),'new_simulations':search.stats['new_simulations'],'forwards':search.stats['forwards'],'SGD_steps':0,'estimated_MCTS96_games_seconds':96*elapsed}
    atomic_json(root/'interface.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
