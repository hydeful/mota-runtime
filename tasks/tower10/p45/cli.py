"""P43 command line. All collection starts at the complete exported Golden."""
import argparse,copy,gzip,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .environment import Environment,install_guard
from .network import Network,model_hash,manual_l2,gradients
from .runtime import Runner,atomic_json,atomic_torch,rng_state,restore_rng,cap_deadline
from .search import Search,Node
from .contracts import validate_episode

def evaluate(root,env,model_path,seconds):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'evaluation.pt'
    model_data=torch.load(model_path,map_location='cpu',weights_only=False);model=Network(env.config['network']);model.load_state_dict(model_data['model']);model.eval();identity=model_data['identity']
    if model_hash(model)!=identity['hash']:raise ValueError('EVAL_MODEL_HASH')
    if identity['compatibility']!=env.compatibility or identity['schema']!=env.schema_hash:raise ValueError('EVAL_MODEL_CONTRACT')
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['model_identity']!=identity or saved['compatibility']!=env.compatibility:raise ValueError('EVAL_VERSION_MISMATCH')
        state=saved
    else:state={'model_identity':identity,'compatibility':env.compatibility,'root':Node(env.initial,0),'records':[],'decision':None,'search_stats':{},'end':None,'evaluation':True,'elapsed_seconds':0.}
    search=Search(env,model,identity['id'],env.config['search']);search.stats.update(state['search_stats']);rng=np.random.default_rng(0);began=time.monotonic();deadline=cap_deadline(began+seconds,env.config['budget']);last=began
    def save():
        nonlocal last
        now=time.monotonic();state['elapsed_seconds']+=now-last;last=now;state['search_stats']=dict(search.stats);atomic_torch(path,state)
        out={'model_identity':identity,'evaluation':True,'end':state['end'],'decisions':len(state['records']),'elapsed_seconds':state['elapsed_seconds'],'search_stats':state['search_stats'],'pending_simulations':state['decision']['completed'] if state['decision'] else 0,'hero':state['root'].state['hero']}
        atomic_json(root/'status.json',out)
    while time.monotonic()<deadline:
        terminal=env.terminal(state['root'].state)
        if terminal is not None:
            state['end']=terminal;save();atomic_json(root/'result.json',{'model_identity':identity,'evaluation':True,'initial_sha':env.identity['initialSHA256'],'records':state['records'],'terminal':terminal,'terminal_state':env.dynamic(state['root'].state),'search_stats':state['search_stats'],'elapsed_seconds':state['elapsed_seconds']});return
        if state['decision'] is None:state['decision']=search.begin(state['root'],rng,evaluation=True)
        while state['decision']['completed']<800 and time.monotonic()<deadline:
            search.simulate(state['root'],state['decision'])
            if state['decision']['completed']%400==0:save()
        if state['decision']['completed']<800:break
        before=state['root'].state;child,record=search.finish(state['root'],state['decision'],rng);record.update(state=env.dynamic(before),diagnostics=env.diagnostics(before,record['legal_action_ids']),origin='P43_independent_evaluation');state['records'].append(record);state['root']=child;state['decision']=None;save()
        print(json.dumps({'event':'evaluation_decision','model_id':identity['id'],'step':record['step'],'action':record['selected_action_id']},ensure_ascii=False),flush=True)
    save()

def acceptance_probe(runner):
    """Only an explicitly marked disposable copy may run off-trigger SGD."""
    root=runner.root;env=runner.env
    if not (root/'ACCEPTANCE_COPY').is_file():raise ValueError('ACCEPTANCE_COPY_MARKER_REQUIRED')
    if not runner.s['replay']:raise ValueError('NO_QUALIFIED_REPLAY')
    a=runner.s['actor'];node=a['root'] if a is not None else Node(env.initial,0)
    if env.terminal(node.state) is not None:node=Node(env.initial,0)
    search=Search(env,runner.model,runner.s['model_id'],env.config['search']);rng=np.random.default_rng(998)
    decision=search.begin(node,rng);search.simulate(node,decision)
    before=model_hash(runner.model);pairs=[runner.s['replay'][int(i)] for i in np.random.randint(0,len(runner.s['replay']),size=runner.s['recipe']['batch'])];xs=[];masks=[];pis=[];zs=[];cache={}
    for eid,t in pairs:
        if eid not in cache:cache[eid]=runner.load_episode(eid);validate_episode(env,cache[eid])
        ep=cache[eid];r=ep['records'][t];xs.append(env.observation.encode(env.restore(r['state']),t));mask=np.zeros(713,bool);mask[r['legal_action_ids']]=True;masks.append(mask);pi=np.zeros(713,np.float32);pi[r['legal_action_ids']]=r['pi'];pis.append(pi);zs.append(ep['z'])
    runner.model.train();runner.optimizer.zero_grad(set_to_none=True);logits,v=runner.model(torch.from_numpy(np.stack(xs)));mask=torch.from_numpy(np.stack(masks));pi=torch.from_numpy(np.stack(pis));z=torch.tensor(zs,dtype=torch.float32);ce=-(pi*torch.log_softmax(logits.masked_fill(~mask,-1e9),1)).sum(1).mean();mse=(v-z).square().mean();(ce+mse).backward(retain_graph=True);g=gradients(runner.model);(1e-4*manual_l2(runner.model)).backward();runner.optimizer.step();runner.model.eval();after=model_hash(runner.model)
    if before==after or not all(x>0 and np.isfinite(x) for x in g.values()):raise ValueError('ACCEPTANCE_SGD_NOT_EFFECTIVE')
    result={'acceptance_copy_only':True,'formal_generation_advanced':False,'new_search_simulations':search.stats['new_simulations'],'new_real_sgd_steps':1,'batch':len(pairs),'sampled_positions':pairs,'before_hash':before,'after_hash':after,'data_gradients':g,'ce':float(ce.detach()),'mse':float(mse.detach()),'search_model_id':runner.s['model_id']}
    atomic_torch(root/'acceptance-probe.pt',{'model':runner.model.state_dict(),'optimizer':runner.optimizer.state_dict(),'rng':rng_state(),'result':result});atomic_json(root/'acceptance-probe.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','collect','train','freeze-recipe','evaluate','check','acceptance-probe','fingerprint','status','stop','resume']);p.add_argument('--run',required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--seed',type=int,default=4201);p.add_argument('--cohorts',type=int,default=0);p.add_argument('--no-train',action='store_true');p.add_argument('--min-new',type=int);p.add_argument('--reason');p.add_argument('--model');p.add_argument('--output');p.add_argument('--steps',type=int);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--run-id',default='P45-probe-V0');p.add_argument('--until-decisions',type=int);p.add_argument('--until-simulations',type=int);p.add_argument('--until-sgd-steps',type=int);p.add_argument('--until-generation',type=int)
    p.add_argument('--config');p.add_argument('--mode',choices=['serial','parallel'],default='parallel');p.add_argument('--worker-until-simulations',type=int);p.add_argument('--observe-recovery',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(args.run)
    if args.command=='stop':
        if not (root/'checkpoint.pt').exists():raise ValueError('MISSING_RUN')
        (root/'STOP').touch();return
    if args.command=='status':
        print((root/'status.json').read_text());return
    if args.command=='resume':
        (root/'STOP').unlink(missing_ok=True)
    denied=install_guard(args.run)
    if args.command=='init':config=json.loads(Path(args.config).read_text()) if args.config else None
    else:config=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)['state']['full_config']
    env=Environment(config=config)
    if args.command=='evaluate':
        if not args.model or not args.output:p.error('evaluate requires --model and --output')
        evaluate(args.output,env,Path(args.run)/args.model,args.seconds);return
    from .parallel import MultiActorRunner
    mode=args.mode if args.command=='init' else torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)['state'].get('execution_mode','serial')
    cls=MultiActorRunner if mode=='parallel' else Runner
    runner=cls(args.run,env,create=args.command=='init',seed=args.seed,run_id=args.run_id)
    if args.command=='init':runner.s['execution_mode']=mode;runner.save()
    if args.command in ('collect','resume'):
        if mode=='parallel' and any(x is not None for x in (args.until_decisions,args.until_simulations,args.until_generation)):p.error('parallel mode uses --cohorts or --worker-until-simulations; global serial limits are unsupported')
        if mode=='parallel':runner.collect_parallel(time.monotonic()+args.seconds,args.cohorts,args.no_train,args.worker_until_simulations,args.observe_recovery)
        else:runner.collect(time.monotonic()+args.seconds,args.cohorts,args.no_train,args.until_decisions,args.until_simulations,args.until_generation)
    elif args.command=='train':
        if args.until_sgd_steps is not None and args.until_sgd_steps<=runner.s['sgd_steps']:runner.save()
        else:runner.train(time.monotonic()+args.seconds,(args.until_sgd_steps-runner.s['sgd_steps']) if args.until_sgd_steps is not None else args.steps)
    elif args.command=='freeze-recipe':
        if runner.s['sgd_steps'] or runner.s['generation'] or runner.s['trainer'] is not None:raise ValueError('RECIPE_ALREADY_TRAINED')
        if args.min_new is None or args.min_new!=env.config['training']['min_new_positions'] or not args.reason:p.error('P43 frozen config requires matching --min-new and --reason; changing recipe requires new explicit config identity')
        before=copy.deepcopy(runner.s['recipe']);runner.s['recipe']['min_new_positions']=args.min_new
        runner.s['recipe_freeze_reason']={'before':before,'after':copy.deepcopy(runner.s['recipe']),'reason':args.reason,'observed_terminal_positions':runner.s['total_positions'],'episodes':len(runner.s['episodes']),'wall_elapsed':runner.s['elapsed_seconds']};atomic_json(runner.root/'recipe.json',runner.s['recipe']);atomic_json(runner.root/'recipe-freeze.json',runner.s['recipe_freeze_reason']);runner.save()
    elif args.command=='check':
        results=[]
        for ep in runner.s['episodes']:validate_episode(env,runner.load_episode(ep['episode_id']),reference=True);results.append(ep['episode_id'])
        atomic_json(runner.root/'episode-verification.json',{'all_pass':True,'episodes':results,'positions':runner.s['total_positions']})
    elif args.command=='acceptance-probe':acceptance_probe(runner)
    elif args.command=='fingerprint':
        result=runner.parallel_fingerprint() if mode=='parallel' else runner.fingerprint();atomic_json(runner.root/'logical-fingerprint.json',result);print(json.dumps(result),flush=True)
    if denied:raise ValueError('UNEXPECTED_LEGACY_READS '+repr(denied))
if __name__=='__main__':main()
