"""Independent equivalence traversal. Never imported by collection/training."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.p49.environment import Environment as Physical
from tasks.tower10.neural.vendor import topology_engine as reference
from .environment import Environment,HERE
from .kernel import component_and_frontier
from .network import Network


FORBIDDEN={'paths','_walk','position','_membership','_direct_reachable'}


def spatial_scan(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('x','y','width','height','widths','display_position','reference_slot'):raise ValueError('SPATIAL_STATE_FIELD '+path+'/'+k)
            spatial_scan(v,path+'/'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):spatial_scan(v,path+'/'+str(i))


def run():
    began=time.monotonic();env=Environment();physical=Physical()
    mapping=json.loads((HERE/'display_mapping.json').read_text())
    old_to_new={m['reference_slot']:m['node'] for m in mapping};new_to_old={v:k for k,v in old_to_new.items()}
    spatial_scan(env.initial)
    seen=set();edges=0;aliases=0;checks=0;forbidden=[];first_failure=None
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_filename.endswith('topology_engine.py') and frame.f_code.co_name in FORBIDDEN:
            forbidden.append(frame.f_code.co_name);raise AssertionError('RUNTIME_GRID_FUNCTION '+frame.f_code.co_name)
        if event=='call' and frame.f_code.co_filename.endswith('path_index.py'):raise AssertionError('RUNTIME_GRID_INDEX')
    def projected(s,old=False):
        hero=s['hero']
        return {'hero':{k:hero[k] for k in ['hp','atk','def','money','items','experience']},
                'objects':{old_to_new[k] if old else k:o['status'] for k,o in s['objects'].items()},
                'goal':s['task']['status'],'weapon':s['flags'].get('nowWeapon'),'shield':s['flags'].get('nowShield'),'hatred':s['flags'].get('hatred',0)}
    def check(p,t):
        nonlocal checks
        assert projected(p,True)==projected(t)
        legal_old={old_to_new[a['target']] for a in reference.legal_actions(p)}
        # Profile every adapter legal-set call, and representative transitions.
        sys.setprofile(profile)
        try:legal_new={a['target'] for a in env.session.legal(t)}
        finally:sys.setprofile(None)
        assert legal_old==legal_new,{'physical':legal_old,'topology':legal_new}
        term=env.terminal(t);goal=p['task']['status']=='GOAL'
        assert (term is not None)==(goal or not legal_old)
        if goal:assert term['task_hp']==p['hero']['hp'] and term['z']==p['hero']['hp']/900
        checks+=1
        return sorted(legal_new)
    def visit(p,t):
        nonlocal edges,aliases
        legal=check(p,t)
        mask=sum(1<<i for i,k in enumerate(sorted(p['objects'])) if p['objects'][k]['status']=='resolved')
        component,_=component_and_frontier(t)
        key=(mask,p['hero']['hp'],p['hero']['x'],p['hero']['y'],tuple(sorted(component)))
        if key in seen:aliases+=1;return
        seen.add(key)
        if len(seen)%10000==0:print(json.dumps({'equivalence_states':len(seen),'compared_edges':edges}),flush=True)
        for target in legal:
            old=physical.step(p,{'target':new_to_old[target]})['state']
            if edges<512:sys.setprofile(profile)
            try:new=env.session.step(t,{'target':target})['state']
            finally:sys.setprofile(None)
            edges+=1
            visit(old,new)
    try:visit(physical.initial,env.initial)
    except Exception as exc:
        first_failure=repr(exc);(HERE/'FIRST_FAILURE.json').write_text(json.dumps({'failure':first_failure,'states':len(seen),'edges':edges},indent=2)+'\n');raise
    # Equivariance checks use actual initial and actually reached states only.
    torch.set_num_threads(1);torch.manual_seed(49001);model=Network(env.config['network']).eval();obs=env.observation.encode(env.initial,0)
    permutation=np.random.default_rng(49911).permutation(21);changed=env.observation.permute(obs,permutation)
    with torch.inference_mode():
        policy,value=model(torch.from_numpy(obs[None]));permuted_policy,permuted_value=model(torch.from_numpy(changed[None]))
        nodes,relations,hero=model.unpack(torch.from_numpy(obs[None]));ablation=model.relation_forward(nodes,torch.zeros_like(relations),hero)
    p_error=float((permuted_policy-policy[:,permutation]).abs().max());v_error=float((permuted_value-value).abs().max())
    assert p_error<1e-6 and v_error<1e-6
    influence={'policy_max_abs':float((policy-ablation[0]).abs().max()),'value_abs':float((value-ablation[1]).abs().max())}
    assert influence['policy_max_abs']>1e-6 and influence['value_abs']>1e-6
    n,r,h=env.observation.tensors(env.initial,0)
    (HERE/'tensor_example.json').write_text(json.dumps({'node_shape':list(n.shape),'relation_shape':list(r.shape),'global_shape':list(h.shape),'node_features':n.tolist(),'relations':r.tolist(),'hero':h.tolist(),'permutation':permutation.tolist(),'policy_before':policy.tolist(),'policy_after':permuted_policy.tolist(),'value_before':value.tolist(),'value_after':permuted_value.tolist()},indent=2)+'\n')
    certificate={'passed':True,'states':len(seen),'edges':edges,'equivalence_checks':checks,'memo_aliases_checked':aliases,'seconds':time.monotonic()-began,'runtime_grid_function_calls':forbidden,'object_nodes':21,'runtime_grid_nodes':0,'policy_equivariance_max_abs':p_error,'value_invariance_max_abs':v_error,'relation_ablation_influence':influence,'node_shape':[21,27],'relation_shape':[2,21,21],'global_shape':[7],'frozen_physical_initial':reference.digest(physical.initial),'topology_initial':env.identity['initialSHA256'],'old_evidence_reuse':'old summary only, no persisted states/transitions; reconstruct one joint equivalence traversal without optimal-value computation','scope':'fixed P49; empty corridor event/stair/fly programs asserted; projected decision-state equivalence, not equality of physical replay coordinates or step counters'}
    (HERE/'CERTIFICATION.json').write_text(json.dumps(certificate,indent=2)+'\n');print(json.dumps(certificate),flush=True)


if __name__=='__main__':run()
