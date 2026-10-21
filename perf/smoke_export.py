"""Small, bounded diagnostics. No terminal samples/checkpoints in default ZIP."""
import fcntl
import hashlib
import json
import os
import time
import zipfile
from perf.common import dump,load,stamp,utc

MAX_BYTES=5_000_000

def export(root,state,finalizing=False):
    with (root/'export.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        started=time.time();payload={};omitted=[];size=0
        def add(name,data):
            nonlocal size
            if len(data)>150_000 or size+len(data)>3_500_000:omitted.append(name);return
            payload[name]=data;size+=len(data)
        def js(name,value):add(name,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
        for name in ('coverage.json','coverage.csv','COVERAGE.txt','environment.json','plan.json','scheduler.json','real-worker-summary.json','controller-error.txt'):
            p=root/name
            if p.is_file():add(name,p.read_bytes())
        info={k:state.get(k) for k in ('name','code_commit','budget_seconds','started_utc','deadline_utc','status','installation','finished_utc','execution_elapsed_seconds','selection','decisions')}
        info['attempts']=[{k:r.get(k) for k in ('label','directory','status','exit_code','wall_seconds','all_related_cpu_seconds','sampled_peak_sum_rss_bytes','resource_sampling_seconds','gpu_sampling_seconds','primary_error','cleanup','python','trial')} for r in state['attempts']]
        js('run-info.json',info)
        for p in sorted((root/'cases').glob('*/*.json')):
            # Case summaries/negative controls only; no recursively copied training ledgers.
            if p.name in ('case.json','details.json','readonly-negative.json','fixture-provenance.json','service-close.json','hash.json','json-check.json','serialization.json','reference.json','input-consumption.json'):
                add(str(p.relative_to(root)),p.read_bytes())
        for p in sorted((root/'attempts').glob('*/*')):
            if p.name not in ('console.log','resources.jsonl'):continue
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));data=f.read(8192)
            add(str(p.relative_to(root))+'.tail',data)
        from perf.common import SOURCE
        add('frozen-P45-config.json',(SOURCE/'tasks/tower10/p45/config.json').read_bytes())
        manifest={'payload_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},'omitted':omitted,'excluded':'terminal samples, model/optimizer/checkpoint files, repeated artifacts; full originals remain in this smoke directory','max_archive_bytes':MAX_BYTES,'log_tail_bytes_per_file':8192}
        tmp=root/'results.zip.tmp';target=root/'results.zip'
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,data in payload.items():z.writestr(name,data)
            z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None
            assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest['payload_sha256'].items())
        if finalizing and not state.get('finished_utc'):
            state['finished_utc']=utc();state['execution_elapsed_seconds']=time.time()-stamp(state['started_utc'])
            state['remaining_absolute_budget_at_finish']=max(0,stamp(state['deadline_utc'])-time.time())
            dump(root/'state.json',state)
        verification={'status':'pass','payload_CRC_and_SHA256_verified':True,'covered_files':len(payload),'manifest_sha256':hashlib.sha256(json.dumps(manifest,ensure_ascii=False,indent=2).encode()).hexdigest(),'execution_finished_utc':state.get('finished_utc'),'execution_elapsed_seconds_fixed':state.get('execution_elapsed_seconds'),'remaining_absolute_budget_at_finish':state.get('remaining_absolute_budget_at_finish'),'export_started_utc':utc(),'execution_late_past_deadline_seconds':max(0,stamp(state['finished_utc'])-stamp(state['deadline_utc'])) if state.get('finished_utc') else None,'this_export_past_absolute_deadline_seconds':max(0,time.time()-stamp(state['deadline_utc'])),'verification_scope':'Payload CRC+SHA verified before this marker was appended; completed ZIP including marker CRC verified after sealing. Frozen execution elapsed includes payload packaging/checks; final tiny marker+central-directory seal time is separately recorded in export.json.'}
        with zipfile.ZipFile(tmp,'a',zipfile.ZIP_DEFLATED) as z:z.writestr('export-verification.json',json.dumps(verification,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(tmp) as z:assert z.testzip() is None and 'coverage.json' in z.namelist() and 'export-verification.json' in z.namelist()
        assert tmp.stat().st_size<=MAX_BYTES,'SMALL_DIAGNOSTIC_PACKAGE_TOO_LARGE'
        os.replace(tmp,target)
        dump(root/'export-verification.json',verification)
        dump(root/'export.json',{'path':str(target),'bytes':target.stat().st_size,'this_export_seconds':time.time()-started,'execution_elapsed_seconds_fixed':state.get('execution_elapsed_seconds'),'execution_finished_utc':state.get('finished_utc'),'export_finished_utc':utc(),'execution_late_past_deadline_seconds':max(0,stamp(state['finished_utc'])-stamp(state['deadline_utc'])) if state.get('finished_utc') else None,'this_export_past_absolute_deadline_seconds':max(0,time.time()-stamp(state['deadline_utc']))})
        return target


def export_full(root):
    """Explicit opt-in local archive; default small results.zip remains untouched."""
    target=root/'full-results.zip';tmp=root/'full-results.zip.tmp'
    with (root/'export.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.is_symlink() or p.suffix in ('.zip','.tmp','.lock'):continue
                z.write(p,p.relative_to(root))
        with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
        os.replace(tmp,target)
    return target
