"""After the 600-second audit, resume the trained run within its original budget."""
import argparse, json, pathlib, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--command',required=True);p.add_argument('--name',required=True);p.add_argument('--output',required=True);a=p.parse_args()
def call(command,*args):return subprocess.check_output([a.command,command,a.name,*args],text=True)
def status():return json.loads(call('status'))
before=status();assert before['state'] in ['slice_finished','budget_finished']
assert before['remaining_wall_budget_seconds']>5,'No authorized budget left; do not extend'
call('resume','--seconds','30')
until=time.monotonic()+20
while time.monotonic()<until:
    active=status()
    if active['decisions']>before['decisions']:break
    time.sleep(.2)
assert active['decisions']>before['decisions'],active
call('stop')
until=time.monotonic()+20
while time.monotonic()<until:
    after=status()
    if after['state']!='running':break
    time.sleep(.2)
assert after['state']=='stopped',after
assert before['deadline_utc']==after['deadline_utc']
assert after['real_sgd_steps']>=before['real_sgd_steps']>0
assert after['model_updates']>=before['model_updates']>0
assert after['last_durable_core_status']['resume_count']>before['last_durable_core_status']['resume_count']
result={'pass':True,'note':'Resumed the same actually trained 600s run; original P45 restore checked model hash, optimizer recipe/state, replay/recipe and full RNG. No new budget granted. This extra run was capped at30s and stopped after observing a new decision.','before':before,'after':after}
pathlib.Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'trained_run_resume':True,'before_decisions':before['decisions'],'after_decisions':after['decisions'],'deadline_unchanged':True}))
