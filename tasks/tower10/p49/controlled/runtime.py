"""Run the byte-identical P45 Runner with P49 relative dependency bindings.

No source replacement and no process-global monkeypatch. Collection, replay,
SGD, publishing and checkpoint guards are the original implementations.
"""
import copyreg
import hashlib
import importlib.util
import json
import inspect
import math
import sys
import time
import types
from pathlib import Path
import torch
from tasks.tower10.neural.vendor.kernel import _Tracked
from . import network
import copy
INITIAL_MODEL=None
def construct(config):
    if INITIAL_MODEL is None:raise ValueError("MISSING_PAIRED_INITIAL_TEMPLATE")
    return copy.deepcopy(INITIAL_MODEL)

copyreg.pickle(_Tracked,lambda value:(dict,(dict(value),)))
SOURCE=Path(__file__).resolve().parents[2]/'p45/runtime.py'
name=__package__+'._bound_p45_runtime'
core=types.ModuleType(name);core.__file__=str(SOURCE);core.__package__=__package__
sys.modules[name]=core
exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),core.__dict__)
core.Network=construct
atomic_json=core.atomic_json
structure_hash=core.structure_hash
atomic_torch=core.atomic_torch
seed_all=core.seed_all
rng_state=core.rng_state
restore_rng=core.restore_rng


class Runner(core.Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.first_step_path=self.root/'first-formal-step.json'
        self.audit_rows=[]
        self.model.register_forward_hook(self._forward_audit)
        self.optimizer.register_step_pre_hook(self._before_step)
        self.optimizer.register_step_post_hook(self._after_step)

    def _audit(self,row):
        if self.s['sgd_steps']==0 and not self.first_step_path.exists():
            self.audit_rows.append(row)

    def _forward_audit(self,module,args,out):
        if not module.training:return
        if not torch.isfinite(args[0]).all() or not all(torch.isfinite(v).all() for v in out):raise ValueError('P49_NONFINITE_TRAIN_FORWARD')
        self._audit({'event':'forward','batch':list(args[0].shape),'policy_shape':list(out[0].shape),'value_shape':list(out[1].shape),'finite':True})

    def _before_step(self,optimizer,args,kwargs):
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()):raise ValueError('P49_NONFINITE_OR_MISSING_GRADIENT')
        if self.s['sgd_steps']==0 and not self.first_step_path.exists():
            frame=inspect.currentframe()
            while frame is not None and frame.f_code is not core.Runner.train.__code__:frame=frame.f_back
            if frame is None:raise ValueError('P49_FIRST_STEP_CONTEXT')
            data=frame.f_locals;pi=data['pi'];z=data['z']
            if not torch.isfinite(pi).all() or not torch.isfinite(z).all() or not torch.allclose(pi.sum(1),torch.ones(len(pi))) or pi[:,21:].any():raise ValueError('P49_FIRST_STEP_TARGET')
            losses={k:float(data[k].detach()) for k in ['ce','mse','regularizer']}
            if not all(math.isfinite(v) for v in losses.values()):raise ValueError('P49_FIRST_STEP_LOSS')
            self._audit({'event':'targets_and_losses','finite':True,'z':z.tolist(),'policy_shape':list(pi.shape),'policy_row_sums':pi.sum(1).tolist(),'padding_target_mass':float(pi[:,21:].sum()),'losses':losses})
            del frame
        self._audit({'event':'before_optimizer','finite':True,'groups':self.parameter_hashes()})

    def _after_step(self,optimizer,args,kwargs):
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('P49_NONFINITE_PARAMETER')
        if self.s['sgd_steps']==0 and not self.first_step_path.exists():
            self.audit_rows.append({'event':'after_optimizer','finite':True,'groups':self.parameter_hashes()})
            atomic_json(self.first_step_path,{'generation':0,'parent':self.s['model_id'],'rows':self.audit_rows,'note':'first real SGD update; data gradients measured before manual L2; not a discarded diagnostic update'})

    def _data_gradients(self,model):
        values=network.gradients(model);self._audit({"event":"data_gradients","values":values});return values

    def load_episode(self,eid):
        ep=super().load_episode(eid)
        if not math.isfinite(ep['z']):raise ValueError('P49_NONFINITE_VALUE_TARGET')
        if ep['end']=='GOAL' and ep['z']!=ep['task_hp']/self.env.h_bound:raise ValueError('P49_WRONG_HP_LABEL')
        if not all(math.isfinite(v) for r in ep['records'] for v in r['pi']):raise ValueError('P49_NONFINITE_POLICY_TARGET')
        return ep

    def train(self,*args,**kwargs):
        core.gradients=lambda model: self._data_gradients(model)
        began=time.monotonic()
        try:
            result=super().train(*args,**kwargs)
            if self.first_step_path.exists() and self.s['sgd_steps']:
                # Loss values are retained by the original train ledger.
                trainer=self.s['trainer']
                if trainer is None:
                    trainer=json.loads((self.root/f"training-N{self.s['generation']}.json").read_text())
                if trainer['losses'] and not all(math.isfinite(v) for row in trainer['losses'] for k,v in row.items() if k!='sample_pairs'):raise ValueError('P49_NONFINITE_LOSS')
            return result
        finally:
            core.gradients=network.gradients
            with (self.root/'phase-times.jsonl').open('a') as f:
                f.write(json.dumps({'phase':'train','generation_after':self.s['generation'],'seconds':time.monotonic()-began})+'\n')


def load_model(path,env):
    saved=torch.load(path,map_location='cpu',weights_only=False);identity=saved['identity']
    if identity['schema']!=env.schema_hash or identity['data_contract']!=env.config_hash or identity['compatibility']!=env.compatibility:raise ValueError('P49_MODEL_TASK_SCHEMA_MISMATCH')
    model=construct(env.config['network']);model.load_state_dict(saved['model'],strict=True);model.eval()
    if network.model_hash(model)!=identity['hash']:raise ValueError('P49_MODEL_WEIGHT_HASH')
    return model,identity
