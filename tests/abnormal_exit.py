import subprocess,time,json,pathlib,os,signal
import argparse
p=argparse.ArgumentParser();p.add_argument('--command',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cmd=a.command;name='abnormal-'+str(os.getpid())
def call(command,*args):return subprocess.check_output([cmd,command,name,*args],text=True)
def s():return json.loads(call('status'))
call('new','--seed','4811','--budget','60');call('start','--seconds','30')
for _ in range(100):
 before=s()
 if before['decisions']>0:break
 time.sleep(.1)
children=subprocess.check_output(['ps','-o','pid=','--ppid',str(before['invocation']['pid'])],text=True).split();assert len(children)==1,children
os.kill(int(children[0]),signal.SIGKILL)
for _ in range(40):
 failed=s()
 if failed['state']=='abnormal_exit':break
 time.sleep(.1)
assert failed['state']=='abnormal_exit' and failed['invocation']['core_returncode']==-9,failed
call('resume','--seconds','15')
for _ in range(100):
 resumed=s()
 if resumed['decisions']>before['decisions']:break
 time.sleep(.1)
assert resumed['decisions']>before['decisions']
call('stop')
for _ in range(40):
 final=s()
 if final['state']=='stopped':break
 time.sleep(.1)
assert final['state']=='stopped' and before['deadline_utc']==final['deadline_utc']
pathlib.Path(a.output).write_text(json.dumps({'pass':True,'first_test_failures':['/proc/PID/task/PID/children unavailable; initial disposable slice ended before manual retry','Retry demonstrated crash/restore but test omitted name on stop; explicit safe stop sent. Neither incomplete attempt claimed a full pass. Final harness always supplies the name.'],'scope':'Actual SIGKILL of a disposable collector, supervisor recorded -9/abnormal_exit; restored same run, advanced, safe-stopped; deadline unchanged','before':before,'failed':failed,'after':final},indent=2)+'\n');print('ABNORMAL_EXIT_AND_RESTORE_VERIFIED')
