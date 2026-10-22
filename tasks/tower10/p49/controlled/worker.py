"""One isolated condition/process: no RNG calls from other paired conditions."""
import argparse,copy,datetime,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import runtime
from .environment import Environment,make_config
from .network import Network,model_hash
from .evaluation import evaluate
from . import fitting
from .guard import install


def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)


def template(path,config):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    # Allocation on meta performs no second random initialization on CPU.
    with torch.device('meta'):model=Network(config['network'])
    model.load_state_dict(saved['model'],strict=True,assign=True)
    assert model_hash(model)==saved['hash'] and sum(p.numel() for p in model.parameters())==123994
    return model,saved


def online(env,root,seed,deadline,heartbeat,initial_hash):
    runner=runtime.Runner(root/'chain',env,create=True,seed=seed,run_id=f'P49-controlled-online-{seed}-{env.condition}')
    assert model_hash(runner.model)==initial_hash
    runtime.atomic_json(root/'RNG-initial.json',{'rng_hash':runtime.structure_hash(runtime.rng_state()),'seed':seed,
       'actor_noise_seed':'seed*100000+episode_id*2','actor_action_seed':'actor_noise_seed+1',
       'SGD_rng':'process-local numpy RandomState, original Runner uniform replacement',
       'isolation':'one OS process per condition; evaluation saves/restores all training RNG'})
    rounds=[]
    evaluate(env,runner.root/(runner.s['model_id']+'.pt'),root/'evaluation',deadline,heartbeat)
    for generation in range(1,6):
        beforep=runner.s['total_positions'];beforee=len(runner.s['episodes']);befores=runner.s['sgd_steps'];began=time.monotonic()
        heartbeat({'event':'online_collect_start','generation':generation,'seed':seed,'condition':env.condition})
        runner.collect(deadline,until_generation=generation)
        assert runner.s['generation']==generation,'FORMAL_BUDGET_OR_INCOMPLETE_GENERATION'
        t=json.loads((runner.root/f'training-N{generation}.json').read_text())
        assert runner.s['sgd_steps']-befores==t['steps_target']==t['steps_done']
        row={'generation':generation,'new_positions':runner.s['total_positions']-beforep,
          'new_episodes':len(runner.s['episodes'])-beforee,'SGD_steps':runner.s['sgd_steps']-befores,
          'SGD_target':t['steps_target'],'seconds':time.monotonic()-began,'model_id':runner.s['model_id'],
          'weight_hash':runner.s['models'][-1]['hash'],'optimizer_hash':runtime.structure_hash(runner.optimizer.state_dict()),
          'RNG_hash':runtime.structure_hash(runtime.rng_state()),'BatchNorm_buffers':fitting.buffer_state(runner.model)}
        rounds.append(row);runtime.atomic_json(root/'rounds.json',rounds);heartbeat({'event':'online_published',**{k:v for k,v in row.items() if k!='BatchNorm_buffers'}})
        evaluate(env,runner.root/(runner.s['model_id']+'.pt'),root/'evaluation',deadline,heartbeat)
    runner=runtime.Runner(root/'chain',env,create=False);before=len(runner.s['episodes'])
    runner.collect(deadline,max_cohorts=1,no_train=True);fresh=runner.s['episodes'][before:]
    assert len(fresh)==4 and runner.s['generation']==5 and len(runner.s['models'])==6 and runner.s['trainer'] is None
    assert all(ep['model_id']==runner.s['model_id'] and ep['weight_hash']==runner.s['models'][-1]['hash'] for ep in fresh)
    runtime.atomic_json(root/'N5-reflow.json',{'reload':True,'episodes':fresh,'N6_triggered':False,
        'model_id':runner.s['model_id'],'weight_hash':runner.s['models'][-1]['hash']})
    runtime.atomic_json(root/'complete.json',{'complete':True,'generation':5,'rounds':rounds,
       'initial_hash':initial_hash,'final_hash':model_hash(runner.model),'SGD_steps':runner.s['sgd_steps'],
       'schema':env.schema_hash,'reflow_episodes':4})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['fit','online'],required=True)
    parser.add_argument('--condition',choices=['A','B'],required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--initial',required=True);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--start-utc',required=True);parser.add_argument('--deadline-utc',required=True);args=parser.parse_args()
    configure();root=Path(args.output);root.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    deadline=time.monotonic()+datetime.datetime.fromisoformat(args.deadline_utc).timestamp()-time.time()
    denied=[];counts={};progress={'phase':args.phase,'condition':args.condition,'seed':args.seed,'complete':False}
    def heartbeat(event):
        progress.update(last_completed_step=event,last_update_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_seconds=time.monotonic()-began)
        runtime.atomic_json(root/'progress.json',progress);print(json.dumps({'heartbeat':event}),flush=True)
    try:
        if args.condition=='B':denied,counts=install(Path(__file__).resolve().parents[4])
        config=make_config(args.condition,args.seed,args.phase);config['budget'].update(session_start_utc=args.start_utc,deadline_utc=args.deadline_utc)
        env=Environment(config);runtime.atomic_json(root/'config.json',config);runtime.atomic_json(root/'schema.json',env.schema_description)
        initial,identity=template(args.initial,config);runtime.INITIAL_MODEL=initial
        assert identity['phase']==args.phase and identity['seed']==args.seed
        runtime.seed_all(args.seed)
        runtime.atomic_json(root/'initial-consumption.json',{'artifact':str(Path(args.initial).resolve()),'state_dict_hash':identity['hash'],
             'actual_model_hash':model_hash(initial),'parameter_signature':identity['signature'],'phase':identity['phase'],
             'schema':env.schema_hash,'rng_hash':runtime.structure_hash(runtime.rng_state())})
        if args.phase=='fit':fitting.run(env,copy.deepcopy(initial),args.data,root/'fit',args.seed,deadline,heartbeat)
        else:online(env,root,args.seed,deadline,heartbeat,identity['hash'])
        progress['complete']=True;heartbeat({'event':'worker_complete','phase':args.phase,'seed':args.seed,'condition':args.condition})
    except BaseException as exc:
        progress.update(error=repr(exc),traceback=traceback.format_exc());heartbeat({'event':'worker_failed','error':repr(exc)});raise
    finally:
        sys.setprofile(None);progress.update(runtime_no_grid_profile=dict(counts),denied_input_reads=denied,worker_seconds=time.monotonic()-began)
        runtime.atomic_json(root/'progress.json',progress)


if __name__=='__main__':main()
