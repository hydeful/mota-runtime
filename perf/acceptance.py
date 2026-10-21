"""Pure acceptance classification; raw cases/cleanup are never rewritten."""
TERMINAL={'zombie','dead'}
LIVE={'running','sleeping','disk-sleep','stopped','tracing-stop','waking','wake-kill','idle','locked','waiting','parked'}

def cleanup_assessment(cleanup):
    c=cleanup if isinstance(cleanup,dict) else {}
    before=c.get('before_cleanup');remaining=c.get('remaining');errors=c.get('errors')
    known_before=isinstance(before,list);known_remaining=isinstance(remaining,list)
    rows=before if known_before else []
    live=[r for r in rows if isinstance(r,dict) and r.get('status') in LIVE]
    terminal=[r for r in rows if isinstance(r,dict) and r.get('status') in TERMINAL]
    unknown=[r for r in rows if not isinstance(r,dict) or r.get('status') not in LIVE|TERMINAL]
    unreadable=c.get('unreadable_process_count')
    scope_known=type(unreadable) is int and unreadable==0
    live_left=[r for r in (remaining if known_remaining else []) if isinstance(r,dict) and r.get('status') in LIVE]
    live_status='fail' if live or live_left else ('pass' if known_before and known_remaining and not unknown and not remaining and scope_known else 'unknown')
    if c.get('complete') is False or (known_remaining and remaining) or (isinstance(errors,list) and errors):reaping='fail'
    elif c.get('complete') is True and known_remaining and not remaining and isinstance(errors,list) and not errors:reaping='pass'
    else:reaping='unknown'
    return {'live_process_status':live_status,'reaping_status':reaping,
            'live_before':live,'terminal_before':terminal,'unknown_before':unknown,
            'live_remaining':live_left,'observation_scope_confirmed':scope_known,
            'terminal_records_successfully_reaped':len(terminal) if reaping=='pass' else None,
            'issues':([ 'unexpected live descendants after direct-child exit' ] if live or live_left else [])+
                     ([ 'process scope/status cannot be confirmed' ] if live_status=='unknown' else [])+
                     ([ 'reaping failed or cannot be confirmed' ] if reaping!='pass' else [])}

def task_assessment(exit_code,termination_requested,primary_error,cleanup):
    x=cleanup_assessment(cleanup)
    execution='pass' if type(exit_code) is int and exit_code==0 and not termination_requested and not primary_error else ('unknown' if exit_code is None else 'fail')
    return dict(x,execution_status=execution,
                accepted=execution=='pass' and x['live_process_status']=='pass' and x['reaping_status']=='pass')

def case_assessment(name,case,record,timeout_entered=False,expected_live_pids=None):
    case=case if isinstance(case,dict) else {};raw=case.get('status','not_run')
    x=task_assessment(record.get('exit_code'),record.get('termination_requested',False),record.get('primary_error'),record.get('cleanup'))
    expected='not_applicable';kind='functional_check';rc=record.get('exit_code')
    if name=='orphan_parent':
        kind='controlled_fault_verification'
        root=record.get('primary_error',{}).get('error')
        expected_pids=set(expected_live_pids or [])
        observed_pids={r.get('pid') for r in x['live_before']}
        expected='pass' if expected_pids and observed_pids==expected_pids and rc==17 and not x['unknown_before'] and x['observation_scope_confirmed'] and x['reaping_status']=='pass' and root in (None,'CHILD_EXIT_17') else ('unknown' if not expected_pids else 'fail')
    elif name=='shutdown_timeout':
        kind='controlled_fault_verification'
        closed=case.get('cleanup',{})
        expected='pass' if raw=='pass' and case.get('controlled_actual_service_shutdown_stall') is True and case.get('primary_error_preserved') and closed.get('cleanup_complete') is True and 'RPC_CONTROL_TIMEOUT shutdown' in str(closed.get('cleanup_errors',[])) else 'fail'
    elif name=='worker_failure':
        kind='controlled_fault_verification'
        expected='pass' if raw=='pass' and case.get('controlled_fault')=='real parallel job missing snapshot' and case.get('primary_error_preserved') and case.get('finally_checkpoint_saved') is True and case.get('service_cleanup',{}).get('cleanup_complete') is True else 'fail'
    elif name=='controlled_failure':
        kind='controlled_fault_verification'
        expected='pass' if raw=='fail' and rc==1 and 'EXPECTED_INJECTED_SUBITEM_FAILURE' in case.get('error','') else 'fail'
    elif name=='controlled_timeout':
        kind='controlled_fault_verification'
        timeout_reason=record.get('termination_reason')=='timeout' or (record.get('termination_reason') is None and record.get('legacy_expected_timeout_observed') is True)
        expected='pass' if timeout_entered and type(rc) is int and rc<0 and timeout_reason else 'fail'
    accepted=x['accepted'] and raw=='pass' and expected in ('pass','not_applicable')
    if name=='orphan_parent':status='pass' if expected=='pass' else 'fail'
    elif raw=='pass':status='pass' if accepted else 'fail'
    elif name=='controlled_timeout' and expected=='pass':status='timeout'
    else:status=raw if raw in ('fail','timeout','not_run','device_unavailable') else 'fail'
    unexpected='pass' if name=='orphan_parent' and expected=='pass' else x['live_process_status']
    return dict(x,status=status,functional_status=raw if raw in ('pass','fail','timeout','not_run','device_unavailable') else 'unknown',unexpected_live_process_status=unexpected,expected_fault_status=expected,acceptance_kind=kind,
                termination_evidence='recorded' if 'termination_requested' in record else 'not_recorded_legacy',
                original_task_status=record.get('status'),raw_case_status=raw)
