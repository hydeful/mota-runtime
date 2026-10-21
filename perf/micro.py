"""Fixed-model/current-run probes; output is diagnostic, never training input."""
import argparse
import copy
import gzip
import io
import json
import statistics
from pathlib import Path
import time
import numpy as np
import torch
from perf.common import configure, dump, load, TOLERANCES
from perf.patches import direct_hash, packed_save
from tasks.tower10.p45.environment import Environment
from tasks.tower10.p45.network import Network, manual_l2, gradients
from tasks.tower10.p45.runtime import atomic_torch, structure_hash
BATCHES=(1,2,3,4,16,64)

def run(baseline,out,gpu=False,reference_file=None):
    configure();out.mkdir(parents=True,exist_ok=True);env=Environment()
    saved=torch.load(baseline/'checkpoint.pt',map_location='cpu',weights_only=False)
    first=saved['state']['models'][0]
    state=torch.load(baseline/(first['id']+'.pt'),map_location='cpu',weights_only=False)['model']
    records=[]
    for p in sorted(baseline.glob('episode-*.json.gz')):
        with gzip.open(p,'rt') as f:ep=json.load(f)
        for r in ep['records']:records.append((r,ep['z']))
        if len(records)>=64:break
    if not records:raise ValueError('NO_CURRENT_BASELINE_TERMINAL_DATA')
    rows=[records[i%len(records)] for i in range(64)]
    began=time.perf_counter();xs=np.stack([env.observation.encode(env.restore(r['state']),r['step']) for r,z in rows]);encode_seconds=time.perf_counter()-began
    mask=np.zeros((64,713),bool);pi=np.zeros((64,713),np.float32)
    for i,(r,z) in enumerate(rows):mask[i,r['legal_action_ids']]=True;pi[i,r['legal_action_ids']]=r['pi']
    z=torch.tensor([z for _,z in rows],dtype=torch.float32);x=torch.from_numpy(xs)
    masks=torch.from_numpy(mask);targets=torch.from_numpy(pi)
    for obs in list(xs)+[xs[0,:,::2,::2]]:
        assert env.observation_hash(obs)==direct_hash(obs)
    hashes={}
    for name,fn in [('copy_sha256',env.observation_hash),('memoryview_sha256',direct_hash)]:
        values=[]
        for _ in range(3):
            began=time.perf_counter()
            for obs in xs:fn(obs)
            values.append(time.perf_counter()-began)
        hashes[name]=values
    # Restore the complete actual state, including NumPy writeability, tensors,
    # optimizer and RNG; an original P45 reader must need no perf class.
    io_result={}
    for name,fn in [('original',atomic_torch),('packed',packed_save)]:
        timings=[]
        for _ in range(2):
            began=time.perf_counter();fn(out/(name+'.pt'),saved);timings.append(time.perf_counter()-began)
        restored=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)
        assert structure_hash(restored)==structure_hash(saved),'SERIALIZATION_SEMANTIC_DIFFERENCE'
        io_result[name]={'seconds':timings,'median':statistics.median(timings),'bytes':(out/(name+'.pt')).stat().st_size}
    json_io={}
    value=load(baseline/'status.json')
    for name,fn in [('stream',lambda f:json.dump(value,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)),('buffer',lambda f:f.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)))]:
        began=time.perf_counter()
        with gzip.open(out/(name+'.json.gz'),'wt',encoding='utf-8') as f:fn(f)
        json_io[name]=time.perf_counter()-began
    with gzip.open(out/'stream.json.gz','rt') as a,gzip.open(out/'buffer.json.gz','rt') as b:assert a.read()==b.read()
    def forward(net,batch):
        net.eval()
        with torch.inference_mode():return tuple(t.detach().cpu() for t in net(x[:batch]))
    def step(net):
        opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=0)
        net.train();opt.zero_grad(set_to_none=True);logits,v=net(x)
        ce=-(targets*torch.log_softmax(logits.masked_fill(~masks,-1e9),1)).sum(1).mean();mse=(v-z).square().mean()
        (ce+mse).backward(retain_graph=True);g=gradients(net);(1e-4*manual_l2(net)).backward();opt.step()
        if not all(np.isfinite(v) and v>0 for v in g.values()):raise ValueError('BAD_GRADIENT')
        return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()},g,float(ce.detach()+mse.detach())
    refnet=Network();refnet.load_state_dict(state)
    outputs={batch:forward(refnet,batch) for batch in BATCHES}
    refnet.load_state_dict(state);reference,gref,lref=step(refnet)
    if gpu:
        if reference_file is None:raise ValueError('CPU214_REFERENCE_REQUIRED_FOR_GPU_GATE')
        golden=np.load(reference_file,allow_pickle=False)
        outputs={batch:(torch.from_numpy(golden[f'logits{batch}']),torch.from_numpy(golden[f'value{batch}'])) for batch in BATCHES}
        reference={k:torch.from_numpy(golden['updated::'+k]) for k in reference}
        gref={k:float(golden['gradient::'+k]) for k in gref};lref=float(golden['loss'])
    else:
        golden={'updated::'+k:v.numpy() for k,v in reference.items()}
        for batch,values in outputs.items():golden['logits'+str(batch)]=values[0].numpy();golden['value'+str(batch)]=values[1].numpy()
        golden.update({'gradient::'+k:np.asarray(v) for k,v in gref.items()});golden['loss']=np.asarray(lref)
        np.savez_compressed(out/'cpu-reference.npz',**golden)
    nn={};correct={}
    configurations=[('cpu1',1),('cpu2',2),('cpu4',4)]
    if gpu:configurations.append(('cuda',1))
    for name,threads in configurations:
        try:
            torch.set_num_threads(threads)
            if name=='cuda':
                if not torch.cuda.is_available():raise RuntimeError('CUDA_NOT_AVAILABLE_IN_SEPARATE_ENV')
                from perf.devices import DeviceNetwork
                factory=DeviceNetwork
            else:factory=Network
            net=factory();net.load_state_dict(state)
            if name=='cuda':torch.cuda.reset_peak_memory_stats()
            timings={}
            for batch in BATCHES:
                observed=forward(net,batch)
                for actual,ref in zip(observed,outputs[batch]):torch.testing.assert_close(actual,ref,atol=TOLERANCES['inference_atol'],rtol=TOLERANCES['inference_rtol'])
                samples=[]
                for _ in range(3):
                    began=time.perf_counter();forward(net,batch);samples.append(time.perf_counter()-began)
                timings[str(batch)]=statistics.median(samples)
            net.load_state_dict(state);began=time.perf_counter();updated,g,loss=step(net);sgd=time.perf_counter()-began
            for k in reference:
                if reference[k].is_floating_point():torch.testing.assert_close(updated[k],reference[k],atol=TOLERANCES['sgd_atol'],rtol=TOLERANCES['sgd_rtol'])
                elif not torch.equal(updated[k],reference[k]):raise ValueError('BN_INTEGER_STATE')
            for k in g:np.testing.assert_allclose(g[k],gref[k],atol=TOLERANCES['sgd_atol'],rtol=TOLERANCES['sgd_rtol'])
            np.testing.assert_allclose(loss,lref,atol=TOLERANCES['sgd_atol'],rtol=TOLERANCES['sgd_rtol'])
            correct[name]={'pass':True,'forward_batches':list(BATCHES),'real_batch64_step':True,'loss':loss,'data_gradients':g}
            nn[name]={'forward_seconds':timings,'sgd_step_seconds':sgd,'threads':threads,'device':'cuda' if name=='cuda' else 'cpu','cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if name=='cuda' else None,'training_split':'CUDA network forward/backward and optimizer, CPU target/loss and encoding' if name=='cuda' else 'all CPU'}
        except Exception as exc:correct[name]={'pass':False,'error':repr(exc)}
    result={'current_baseline':str(baseline),'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,'isolated_probe_not_a_learning_run':True,'tolerances_preregistered':TOLERANCES,'cpu_gpu_numeric_basis':'GPU compares to savedCPU2.14.1 golden outputs and real one-step update, including across Torch versions. float32, deterministic kernels, TF32 disabled; output abs1e-5/rel1e-4, one-step loss/grad/parameters abs1e-5/rel1e-3; no relaxation after observing results','encode64_seconds':encode_seconds,'hash':hashes,'serialization':io_result,'json_gzip_seconds':json_io,'nn':nn,'correctness':correct,'layout_cache':'NOT RETRIED: P46 full-loop0.891/0.953/0.922, only58% hits and key/allocation overhead. No new mechanism justifies retry.','note':'All launch/CUDA warmup/micro failures count against controller budget. Isolated medians cannot be multiplied into a full-loop speedup.'}
    dump(out/'micro.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--reference');a=p.parse_args();run(Path(a.baseline),Path(a.out),a.gpu,a.reference)
