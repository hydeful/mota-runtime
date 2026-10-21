"""Exercise real detached sessions, safe stop/resume and non-reset budgets."""
import argparse, json, os, pathlib, subprocess, time

p=argparse.ArgumentParser()
p.add_argument('--command', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
name='lifecycle-'+str(os.getpid())
events=[]
def call(*args, good=True):
    r=subprocess.run([a.command,*args],text=True,capture_output=True)
    events.append({'args':args,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    assert (r.returncode==0)==good,events[-1]
    return r.stdout
def s():return json.loads(call('status',name))
def wait_stopped():
    limit=time.monotonic()+30
    while time.monotonic()<limit:
        x=s()
        if x['state']!='running':return x
        time.sleep(.2)
    raise RuntimeError('safe stop did not complete within 30 seconds')
call('new',name,'--seed','4803','--budget','120')
initial=s()
call('new',name,'--seed','4803','--budget','120',good=False)
call('start',name,'--seconds','60')
call('start',name,'--seconds','60',good=False)
# A separate controlling process exits; the detached worker must still advance.
limit=time.monotonic()+20
while time.monotonic()<limit:
    before=s()
    if before['decisions']>0:break
    time.sleep(.2)
assert before['decisions']>0 and before['state']=='running'
call('stop',name)
stopped=wait_stopped()
assert stopped['state']=='stopped',stopped
call('resume',name,'--seconds','20')
limit=time.monotonic()+20
while time.monotonic()<limit:
    resumed=s()
    if resumed['decisions']>stopped['decisions']:break
    time.sleep(.2)
assert resumed['decisions']>stopped['decisions'],resumed
call('stop',name)
final=wait_stopped()
assert initial['deadline_utc']==stopped['deadline_utc']==final['deadline_utc']
assert final['last_durable_core_status']['resume_count']>stopped['last_durable_core_status']['resume_count']
assert final['real_sgd_steps']>=stopped['real_sgd_steps']
call('diagnose',name)
receipt={'pass':True,'scope':'detached worker survives initiating command exit; safe stop, same-run restore, no budget reset, duplicate new/start rejection; browser closure requires AutoDL user evidence','initial':initial,'stopped':stopped,'resumed':resumed,'final':final,'events':events}
(out/'LIFECYCLE.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pass':True,'name':name,'before_decisions':stopped['decisions'],'after_decisions':final['decisions'],'deadline_unchanged':True}))
