"""Small installation/control/serialization tests; never starts the1h matrix."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch
from perf.common import SOURCE, configure, dump, load, utc, verify_source
from perf.patches import direct_hash, packed_save, Profile

def run(out):
    out.mkdir(parents=True,exist_ok=False);configure();count=verify_source();checks={}
    rng=np.random.default_rng(48)
    for shape in [(550,13,13),(2,4,4)]:
        x=rng.standard_normal(shape).astype(np.float32)
        for v in [x,x[...,::2]]:assert direct_hash(v)==hashlib.sha256(v.tobytes()).hexdigest()
    checks['hash_contiguous_and_strided']=True
    arrays=[np.arange(9,dtype=np.float32),np.arange(5,dtype=np.int64)]
    arrays[1].flags.writeable=False
    payload={'state':{'arrays':arrays,'tuple':(1,'x'),'nested':{'a':[1,2,3]}},'model':{'w':torch.arange(3.)},'optimizer':{'momentum':torch.ones(3)}}
    packed_save(out/'packed.pt',payload)
    # Fresh process imports only torch/numpy, not perf. Class-free restore is vital.
    code='import torch,sys; x=torch.load(sys.argv[1],weights_only=False); assert x["state"]["arrays"][0].flags.writeable; assert not x["state"]["arrays"][1].flags.writeable; assert "perf.patches" not in sys.modules; print("original reader compatible")'
    p=subprocess.run([sys.executable,'-c',code,str(out/'packed.pt')],text=True,capture_output=True,check=True)
    checks['packed_builtin_reader_preserves_numpy_writeability']=p.stdout.strip()
    # Measure wrapper overhead only as a calibration, never subtract it from results.
    def nothing():pass
    profile=Profile();wrapped=profile.wrap(nothing,'empty')
    begin=time.perf_counter()
    for _ in range(100000):nothing()
    raw=time.perf_counter()-begin;begin=time.perf_counter()
    for _ in range(100000):wrapped()
    calibrated=max(0,time.perf_counter()-begin-raw)/100000
    checks['timing_wrapper_seconds_per_empty_call']=calibrated
    from perf import control
    control.HOME=out/'home';root=control.HOME/'experiments/test';root.mkdir(parents=True)
    start=utc();deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=45)
    state={'name':'test','status':'prepared','code_commit':control.revision(),'baseline_commit':control.BASELINE,'budget_seconds':45,
           'started_utc':start,'deadline_utc':deadline.isoformat(),'attempts':[],'decisions':[],'active':None}
    dump(root/'state.json',state);dump(root/'environment.json',{})
    with control.lock(root) as held:
        try:control.lock(root)
        except RuntimeError:checks['duplicate_lock_rejected']=True
        else:raise AssertionError('duplicate lock accepted')
        trialdir=root/'runs/disposable';trialdir.mkdir(parents=True)
        code='import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); deadline=time.monotonic()+10\nwhile time.monotonic()<deadline and not (p/"STOP").exists():time.sleep(.05)\n(p/"saved.marker").write_text("stopped")'
        record=control.task(root,state,'bounded-stop',['-c',code,str(trialdir)],sys.executable,.5,held,{'run':str(trialdir)})
        assert record['status']=='incomplete' and (trialdir/'saved.marker').exists()
    checks['actual_child_safe_stop_and_resource_sampling']=True
    initial_deadline=state['deadline_utc']
    control.controller(root) # Insufficient budget must export, not reset or start600s.
    after=load(root/'state.json');assert after['deadline_utc']==initial_deadline
    assert 'INSUFFICIENT_BUDGET' in after['error'] and (root/'results.zip').exists()
    control.controller(root);assert load(root/'state.json')['deadline_utc']==initial_deadline
    checks['restart_preserves_absolute_deadline']=True
    with zipfile.ZipFile(root/'results.zip') as z:
        assert 'summary.csv' in z.namelist() and 'SUMMARY.txt' in z.namelist()
        assert not any(x.endswith(('.pt','.npz')) for x in z.namelist())
    checks['automatic_results_zip_csv_summary_no_weights']=True
    # A one-hour matrix is deliberately not run by installation tests.
    result={'pass':True,'frozen_files_unchanged':count,'checks':checks,'CUDA_tested':False,'AutoDL_tested':False,'full3600s_matrix_tested':False}
    dump(out/'SELFTEST.json',result);print('P48小规模自检通过：源码/哈希/保存兼容/锁/真实子进程停止/持久预算/结果包；CUDA和云端矩阵待验证。')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(Path(a.out))
