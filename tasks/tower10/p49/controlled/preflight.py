"""New adapter boundary only; no SGD and no exhaustive certification repeat."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from tasks.tower10.p49.learning.network import Network as MapNetwork
from tasks.tower10.p45.network import Block,Network as P45Network
from . import runtime
from .network import Network,model_hash
from .environment import Environment,make_config,assert_configs
from .evaluation import evaluate
from .fitting import publication


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg={c:make_config(c,49099,'preflight') for c in ['A','B']};different=assert_configs(cfg['A'],cfg['B'])
    runtime.seed_all(49099);initial=Network(cfg['A']['network']);runtime.INITIAL_MODEL=initial
    assert Network is MapNetwork and Network.forward is P45Network.forward
    assert sum(p.numel() for p in initial.parameters())==123994
    signature=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in initial.named_parameters()]
    signature_b=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in copy.deepcopy(initial).named_parameters()]
    assert signature==signature_b
    envs={c:Environment(cfg[c]) for c in ['A','B']};hashes={};timings={};reject=[]
    for c,env in envs.items():
        before=model_hash(initial);model=copy.deepcopy(initial);identity=publication(model,env,f'preflight-{c}')
        path=root/f'{c}.pt';runtime.atomic_torch(path,{'model':model.state_dict(),'identity':identity})
        loaded,_=runtime.load_model(path,env);assert model_hash(loaded)==before and not loaded.training
        assert all(not m.training for m in loaded.modules() if isinstance(m,torch.nn.BatchNorm2d))
        bad=envs['B' if c=='A' else 'A']
        try:runtime.load_model(path,bad)
        except ValueError:reject.append(c)
        else:raise AssertionError('SCHEMA_NOT_REJECTED')
        # No-gradient timing of exactly one full MCTS100 game, same as formal.
        save=env.config['evaluation'];env.config['evaluation']=dict(save,conditions=['MCTS100_no_noise_maxN'])
        began=time.monotonic();rows=evaluate(env,path,root,time.monotonic()+300);timings[c]=dict(seconds=time.monotonic()-began,decisions=rows[0]['decisions'],forwards=rows[0]['network_forwards'],simulations=rows[0]['new_simulations'])
        env.config['evaluation']=save;hashes[c]=before;assert model_hash(initial)==before
    assert hashes['A']==hashes['B'] and envs['A'].schema_hash!=envs['B'].schema_hash
    result={'passed':True,'parameter_count':123994,'network_class_identical':True,'forward_identical_to_original_P45':True,
      'parameter_signatures_identical':True,'signature':signature,'paired_state_dict_hashes':hashes,
      'BatchNorm_buffers_in_hash':True,'save_reload_equal':True,'cross_schema_rejected':reject,
      'configuration_difference_whitelist':different,'MCTS100_timings':timings,'SGD_steps':0,
      'encoding_checks':'940 real replay states in frozen data MANIFEST, exact roundtrips and same masks/targets',
      'conservative_estimate_seconds':900,'formal_budget_seconds':3600,'protocol_size_unchanged':True}
    runtime.atomic_json(root/'PREFLIGHT.json',result);return result


if __name__=='__main__':print(json.dumps(run('/workspace/p49-controlled-preflight'),indent=2))
