"""Fixed-experience diagnostic; original CE+MSE/manual-L2/SGD formula."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from .network import Network,model_hash,manual_l2,gradients
from .runtime import structure_hash,rng_state,atomic_json,atomic_torch
from .evaluation import evaluate

CHECKPOINTS=[0,10,27,37,47,56]


def buffer_state(model):return {k:v.detach().cpu().tolist() for k,v in model.named_buffers()}
def group_hashes(model):
    return {group:structure_hash({n:p.detach() for n,p in model.named_parameters() if n.startswith(group)}) for group in ['body','policy','value']}


def publication(model,env,mid,parent=None):
    return {'id':mid,'hash':model_hash(model),'schema':env.schema_hash,'data_contract':env.config_hash,
            'compatibility':env.compatibility,'parent':parent}


def metric(model,x,mask,pi,z,indices,constant):
    old=model.training;before=model_hash(model);model.eval();predictions=[];ces=[];kls=[];mses=[]
    with torch.inference_mode():
        for start in range(0,len(indices),64):
            ix=indices[start:start+64];logits,value=model(x[ix]);lp=torch.log_softmax(logits.masked_fill(~mask[ix],-1e9),1)
            ce=-(pi[ix]*lp).sum(1);kl=(pi[ix]*(torch.log(pi[ix].clamp_min(1e-30))-lp)).sum(1)
            predictions.extend(value.tolist());ces.extend(ce.tolist());kls.extend(kl.tolist());mses.extend((value-z[ix]).square().tolist())
    assert model_hash(model)==before;model.train(old)
    v=np.asarray(predictions);targets=z[indices].numpy();assert np.isfinite(v).all()
    return {'n':len(indices),'CE':float(np.mean(ces)),'KL':float(np.mean(kls)),
      'value_MSE':float(np.mean(mses)),'constant_value':constant,'constant_MSE':float(np.mean((targets-constant)**2)),
      'prediction_mean':float(v.mean()),'prediction_std':float(v.std()),'prediction_min':float(v.min()),'prediction_max':float(v.max()),
      'target_mean':float(targets.mean()),'target_std':float(targets.std()),'bias':float(np.mean(v-targets)),
      'prediction_target_covariance':float(np.mean((v-v.mean())*(targets-targets.mean())))},v


def run(env,model,data_root,root,seed,deadline,heartbeat):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);data_root=Path(data_root)
    data=np.load(data_root/'paired.npz');batches=np.load(data_root/f'batches-{seed}.npy')
    x=torch.from_numpy(data[env.condition]);mask=torch.from_numpy(data['masks']);pi=torch.from_numpy(data['pi']);z=torch.from_numpy(data['z'])
    training=data['training'];held=data['held'];assert batches.shape==(56,64) and not np.isin(batches,held).any()
    constant=json.loads((data_root/'MANIFEST.json').read_text())['constant_value']
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=0)
    initial_parameters={n:p.detach().clone() for n,p in model.named_parameters()};initial_hash=model_hash(model)
    ledger=[];metrics=[];previous=None;midbase=f'P49-controlled-fit-{seed}-{env.condition}'
    def snapshot(step):
        nonlocal previous
        mid=f'{midbase}-S{step}';ident=publication(model,env,mid,previous);previous=mid
        atomic_torch(root/f'{mid}.pt',{'model':model.state_dict(),'identity':ident})
        # Full optimizer/RNG state kept at every requested diagnostic checkpoint.
        atomic_torch(root/f'checkpoint-S{step}.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':rng_state(),'step':step,'schema':env.schema_hash,'batch_indices':batches,'identity':ident})
        train,vtrain=metric(model,x,mask,pi,z,training,constant);hold,vheld=metric(model,x,mask,pi,z,held,constant)
        np.savez_compressed(root/f'predictions-S{step}.npz',train=vtrain,held=vheld,training_indices=training,held_indices=held)
        row={'step':step,'model_id':mid,'weight_hash':model_hash(model),'initial_hash':initial_hash,
          'train':train,'held':hold,'data_gradients':ledger[-1]['data_gradients'] if ledger else None,
          'parameter_group_hashes':group_hashes(model),'parameter_delta_L2':{
            g:math.sqrt(sum(float((p.detach()-initial_parameters[n]).square().sum()) for n,p in model.named_parameters() if n.startswith(g))) for g in ['body','policy','value']},
          'BatchNorm_buffers':buffer_state(model),'optimizer_hash':structure_hash(optimizer.state_dict()),'rng_hash':structure_hash(rng_state())}
        metrics.append(row);atomic_json(root/'metrics.json',metrics)
        heartbeat({'event':'fit_checkpoint','seed':seed,'condition':env.condition,'step':step})
        if step in (0,56):evaluate(env,root/f'{mid}.pt',root/'evaluation',deadline,heartbeat)
    snapshot(0)
    for step,ix in enumerate(batches,1):
        if time.monotonic()>=deadline:raise TimeoutError('SHARED_FORMAL_BUDGET')
        model.train();assert all(m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
        optimizer.zero_grad(set_to_none=True);logits,value=model(x[ix]);lp=torch.log_softmax(logits.masked_fill(~mask[ix],-1e9),dim=1)
        ce=-(pi[ix]*lp).sum(1).mean();mse=(value-z[ix]).square().mean();loss=ce+mse
        assert all(torch.isfinite(a).all() for a in [x[ix],pi[ix],z[ix],logits,value,loss])
        loss.backward(retain_graph=True);g=gradients(model)
        assert all(math.isfinite(v) and v>0 for v in g.values())
        regularizer=1e-4*manual_l2(model);regularizer.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before=group_hashes(model);optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
        row={'step':step,'sample_indices':ix.tolist(),'CE':float(ce.detach()),'MSE':float(mse.detach()),
           'manual_L2':float(regularizer.detach()),'data_gradients':g,'before_groups':before,'after_groups':group_hashes(model),
           'optimizer_hash':structure_hash(optimizer.state_dict()),'rng_hash':structure_hash(rng_state()),'finite':True}
        ledger.append(row);atomic_json(root/'training.json',ledger)
        if step==1:
            atomic_json(root/'first-formal-step.json',dict(row,input_shape=list(x[ix].shape),policy_shape=list(pi[ix].shape),
              padding_target_mass=float(pi[ix,21:].sum()),value_label_range=[float(z[ix].min()),float(z[ix].max())],
              train_mode=True,BatchNorm_buffers=buffer_state(model),note='first actual SGD, not discarded diagnostic'))
        if step in CHECKPOINTS:snapshot(step)
    model.eval();atomic_json(root/'complete.json',{'complete':True,'steps':56,'seed':seed,'condition':env.condition,
      'initial_hash':initial_hash,'final_hash':model_hash(model),'initial_rng':'copied paired initialization artifact; worker seed stream isolated',
      'schema':env.schema_hash,'training_positions':801,'held_positions':139})
