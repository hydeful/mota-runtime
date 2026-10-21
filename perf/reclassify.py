"""Offline sidecar review. Reads prior evidence; never starts work or changes budgets."""
import argparse
import hashlib
import json
from pathlib import Path
from perf.acceptance import case_assessment


def read(path,default):
    return json.loads(path.read_text()) if path.exists() else default

def review(root):
    state=read(root/'state.json',{})
    coverage=read(root/'coverage.json',{}).get('paths',state.get('coverage',[]))
    originals={r['path']:r for r in coverage};rows=[]
    for rec in state.get('attempts',[]):
        name=rec['label'];out=root/'cases'/name;case=read(out/'case.json',{})
        if not case and rec.get('result',{}).get('pass') is True:case={'status':'pass','official_worker_result':rec['result']}
        if not case and name.startswith('dispatch-'):case={'status':'pass' if rec.get('exit_code')==0 else 'fail','logic_only':True}
        if not case and name not in originals:continue
        adapted=dict(rec)
        adapted['legacy_expected_timeout_observed']=originals.get(name,{}).get('expected_timeout_observed')
        orphan=read(out/'orphan.json',{}) if name=='orphan_parent' else {}
        pids=[orphan['child_pid']] if type(orphan.get('child_pid')) is int else None
        assessment=case_assessment(name,case,adapted,(out/'timeout-entered.json').exists(),pids)
        rows.append({'path':name,'original_coverage':originals.get(name),'original_task_record':rec,
                     'original_case_result':case,'assessment':assessment})
    observed={r['path'] for r in rows}
    return {'classification_only':True,'experiments_started':False,'old_code_commit':state.get('code_commit'),
            'elapsed_seconds_unchanged':state.get('execution_elapsed_seconds'),
            'original_deadline_unchanged':state.get('deadline_utc'),'paths':rows,
            'coverage_not_recomputed':[r for n,r in originals.items() if n not in observed],
            'readiness':'This sidecar only reviews task/case acceptance. It does not reconstruct missing functional evidence, dependency chains or authorize the one-hour experiment.'}

def write_review(root,target):
    # No overwrite, no modification of state/cases/coverage/results.zip.
    protected=[root/'state.json',root/'coverage.json']+list((root/'cases').glob('*/*.json'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    report=review(root);report['source_sha256']=before
    for name,digest in before.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    with target.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(write_review(a.root,a.output))
