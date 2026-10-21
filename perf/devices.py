"""CPU-visible GPU network and version-bound shared batched inference service."""
from collections import OrderedDict
import hashlib
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
import queue
import secrets
import threading
import time
import numpy as np
import torch
from tasks.tower10.p45.network import Network, model_hash

class DeviceNetwork(Network):
    def __init__(self, config=None):
        super().__init__(config)
        self.to('cuda')
    def forward(self, x):
        # CPU loss/target code remains unchanged; autograd crosses this transfer.
        logits,value=super().forward(x.to('cuda'))
        return logits.cpu(),value.cpu()

def _server(pipe, auth, device, batch_wait, test_shutdown_stall=False):
    from perf.common import configure
    configure()
    listener=Listener(('127.0.0.1',0),authkey=auth)
    pipe.send(listener.address); pipe.close()
    requests=queue.Queue(); commands=queue.Queue(); stopping=threading.Event()
    counters={'requests':0,'kernel_batches':0,'kernel_rows':0,'cache_hits':0,
              'invalidations':0,'evictions':0,'h2d_seconds':0.,'forward_seconds':0.,
              'd2h_seconds':0.,'batch_fill_wait_seconds':0.,'max_batch':0,
              'hash_seconds':0.,'request_bytes':0,'deduplicated_rows':0,'max_request_batch':0}
    def reader(conn):
        try:
            while True:
                kind,payload=conn.recv()
                reply=queue.Queue(maxsize=1)
                (requests if kind=='infer' else commands).put((kind,payload,reply))
                result=reply.get();conn.send(result)
                if kind=='shutdown':break
        except (EOFError,OSError):pass
        finally:conn.close()
    def accept():
        while not stopping.is_set():
            try:c=listener.accept()
            except OSError:break
            threading.Thread(target=reader,args=(c,),daemon=True).start()
    threading.Thread(target=accept,daemon=True).start()
    net=None; current=None; cache=OrderedDict()
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    while not stopping.is_set():
        try:
            kind,payload,reply=commands.get_nowait()
            if kind=='publish':
                state,identity=payload
                candidate=Network();candidate.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in state.items()})
                if model_hash(candidate)!=identity:raise ValueError('RPC_PUBLISHED_HASH')
                candidate.to(device).eval();sync();net=candidate
                if current!=identity:cache.clear();counters['invalidations']+=1
                current=identity;reply.put(('ok',None))
            elif kind=='stats':reply.put(('ok',dict(counters,device=device,weight_hash=current)))
            elif kind=='shutdown':
                if test_shutdown_stall:time.sleep(60) # explicit smoke fault only
                extra={'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_name':torch.cuda.get_device_name()} if device=='cuda' and torch.cuda.is_initialized() else {}
                reply.put(('ok',dict(counters,device=device,**extra)));stopping.set()
            continue
        except queue.Empty:pass
        except Exception as exc:
            reply.put(('error',repr(exc)));continue
        try:first=requests.get(timeout=.002)
        except queue.Empty:continue
        batch=[first];began=time.perf_counter();until=time.perf_counter()+batch_wait
        while len(batch)<16:
            try:batch.append(requests.get(timeout=max(0,until-time.perf_counter())))
            except queue.Empty:break
        counters['batch_fill_wait_seconds']+=time.perf_counter()-began
        pending=[];keys=[];locations={};counters['max_request_batch']=max(counters['max_request_batch'],len(batch))
        for _,(identity,schema,obs),reply in batch:
            counters['requests']+=1
            if identity!=current or net is None:
                reply.put(('error','RPC_STALE_MODEL'));continue
            began=time.perf_counter();key=(identity,schema,hashlib.sha256(memoryview(obs).cast('B')).hexdigest())
            counters['hash_seconds']+=time.perf_counter()-began;counters['request_bytes']+=obs.nbytes
            if key in cache:
                counters['cache_hits']+=1;cache.move_to_end(key);reply.put(('ok',cache[key]))
            elif key in locations:
                pending[locations[key]][1].append(reply);counters['deduplicated_rows']+=1
            else:locations[key]=len(pending);pending.append((obs,[reply]));keys.append(key)
        if not pending:continue
        try:
            began=time.perf_counter();x=torch.from_numpy(np.stack([x for x,_ in pending])).to(device);sync()
            counters['h2d_seconds']+=time.perf_counter()-began;began=time.perf_counter()
            with torch.inference_mode():logits,values=net(x)
            sync();counters['forward_seconds']+=time.perf_counter()-began;began=time.perf_counter()
            logits=logits.cpu().numpy();values=values.cpu().numpy();sync()
            counters['d2h_seconds']+=time.perf_counter()-began
            counters['kernel_batches']+=1;counters['kernel_rows']+=len(pending)
            counters['max_batch']=max(counters['max_batch'],len(pending))
            for i,(_,replies) in enumerate(pending):
                result=(logits[i].copy(),float(values[i]));cache[keys[i]]=result
                if len(cache)>20000:cache.popitem(last=False);counters['evictions']+=1
                for reply in replies:reply.put(('ok',result))
        except Exception as exc:
            for _,replies in pending:
                for reply in replies:reply.put(('error',repr(exc)))
    listener.close()

class Service:
    def __init__(self,device,batch_wait=.0002,*,close_timeout=2.,_test_shutdown_stall=False):
        self.auth=secrets.token_bytes(32);self.connection=None;self.process=None
        self.close_timeout=close_timeout;self.close_result=None
        recv,send=mp.get_context('spawn').Pipe(duplex=False)
        try:
            self.process=mp.get_context('spawn').Process(target=_server,args=(send,self.auth,device,batch_wait,_test_shutdown_stall))
            self.process.start();send.close()
            if not recv.poll(20):raise RuntimeError('RPC_SERVER_START_TIMEOUT')
            self.address=recv.recv()
            self.connection=Client(self.address,authkey=self.auth)
        except BaseException as exc:
            cleanup=self.close(shutdown=False)
            exc.add_note('Service startup cleanup: '+str(cleanup))
            raise
        finally:recv.close();send.close()
    def call(self,kind,payload=None,timeout=30):
        self.connection.send((kind,payload))
        if not self.connection.poll(timeout):raise TimeoutError('RPC_CONTROL_TIMEOUT '+kind)
        state,result=self.connection.recv()
        if state!='ok':raise RuntimeError(result)
        return result
    def publish(self,model):
        return self.call('publish',({k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},model_hash(model)))
    def close(self,shutdown=True):
        # Always return cleanup evidence; never mask the caller's original error.
        if self.close_result is not None:return self.close_result
        result={};errors=[]
        try:
            if shutdown and self.connection is not None:
                result=self.call('shutdown',timeout=self.close_timeout)
                if not isinstance(result,dict):result={};raise RuntimeError('INVALID_SHUTDOWN_REPLY')
        except BaseException as exc:errors.append({'phase':'shutdown','error':repr(exc)})
        finally:
            if self.connection is not None:
                try:self.connection.close()
                except Exception as exc:errors.append({'phase':'connection_close','error':repr(exc)})
            if self.process is not None and self.process.pid is not None:
                try:
                    self.process.join(.4)
                    if self.process.is_alive():self.process.terminate();self.process.join(.4)
                    if self.process.is_alive():self.process.kill();self.process.join(.4)
                except Exception as exc:errors.append({'phase':'process_reap','error':repr(exc)})
        try:alive=self.process is not None and self.process.pid is not None and self.process.is_alive()
        except Exception as exc:alive=True;errors.append({'phase':'liveness_check','error':repr(exc)})
        self.close_result=dict(result,cleanup_errors=errors,cleanup_complete=not alive,service_pid=self.process.pid if self.process is not None else None)
        return self.close_result


def cleanup_record(path,value):
    """Persist cleanup even on failure; a write error cannot replace an active root error."""
    import sys
    from perf.common import dump
    primary=sys.exc_info()[1]
    try:dump(path,value)
    except Exception as exc:
        if primary is not None:primary.add_note('Cleanup evidence write failed: '+repr(exc))
        else:raise

class RemoteForward:
    def __init__(self,address,auth,weight_hash,schema):
        self.conn=Client(address,authkey=auth);self.weight_hash=weight_hash;self.schema=schema
        self.wait_seconds=0.;self.requests=0;self.send_seconds=0.;self.response_wait_seconds=0.
    def __call__(self,x):
        if x.shape[0]!=1:raise ValueError('ACTOR_RPC_REQUIRES_ONE_OBSERVATION')
        began=time.perf_counter();self.conn.send(('infer',(self.weight_hash,self.schema,x[0].numpy())))
        sent=time.perf_counter();self.send_seconds+=sent-began
        if not self.conn.poll(30):raise RuntimeError('RPC_INFER_TIMEOUT')
        state,result=self.conn.recv();self.wait_seconds+=time.perf_counter()-began;self.response_wait_seconds+=time.perf_counter()-sent;self.requests+=1
        if state!='ok':raise RuntimeError(result)
        logits,v=result
        return torch.from_numpy(logits[None]),torch.tensor([v],dtype=torch.float32)
    def close(self):
        try:self.conn.close();return None
        except Exception as exc:return repr(exc)
