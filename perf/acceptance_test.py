"""Classification regression tests; no search, SGD, GPU or cloud execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from perf.acceptance import case_assessment,task_assessment
from perf.reclassify import write_review

def clean(before=None,remaining=None,complete=True,errors=None,unreadable=0):
    return {'before_cleanup':before or [],'remaining':remaining or [],'complete':complete,
            'errors':errors or [],'unreadable_process_count':unreadable}
def record(cleanup,exit_code=0):return {'status':'incomplete','exit_code':exit_code,'cleanup':cleanup}

class AcceptanceTests(unittest.TestCase):
    def test_zombie_reaped_case_pass_not_rewritten(self):
        case={'status':'pass','witness':'raw remains intact'};rec=record(clean([{'pid':11,'status':'zombie'}]))
        before=copy.deepcopy((case,rec));x=case_assessment('cpu_parallel',case,rec)
        self.assertEqual((x['status'],x['functional_status'],x['live_process_status'],x['reaping_status']),('pass','pass','pass','pass'))
        self.assertEqual(x['terminal_records_successfully_reaped'],1);self.assertEqual((case,rec),before)
        for name in ('cpu_parallel','service','gpu_service'):self.assertEqual(case_assessment(name,case,rec)['status'],'pass')
        self.assertTrue(task_assessment(0,False,None,rec['cleanup'])['accepted'])
    def test_live_leak_cleaned_still_problem(self):
        x=case_assessment('service',{'status':'pass'},record(clean([{'pid':11,'status':'sleeping'}])))
        self.assertEqual((x['status'],x['functional_status'],x['unexpected_live_process_status'],x['reaping_status']),('fail','pass','fail','pass'))
    def test_live_remaining_and_zombie_remaining(self):
        for status in ('sleeping','zombie'):
            with self.subTest(status=status):
                x=case_assessment('service',{'status':'pass'},record(clean(remaining=[{'pid':11,'status':status}],complete=False)))
                self.assertEqual(x['status'],'fail');self.assertEqual(x['reaping_status'],'fail')
                self.assertEqual(x['live_process_status'],'pass' if status=='zombie' else 'fail')
    def test_cleanup_error_not_hidden_by_empty_remaining(self):
        for c in (clean(complete=False),clean(errors=['wait failed'])):
            x=case_assessment('service',{'status':'pass'},record(c))
            self.assertEqual(x['functional_status'],'pass');self.assertEqual(x['status'],'fail');self.assertEqual(x['reaping_status'],'fail')
    def test_unknown_is_not_pass(self):
        for c in ({},clean(unreadable=1),clean(before=[{'pid':11,'gone':True}]),dict(clean(),remaining='scan unavailable')):
            x=case_assessment('service',{'status':'pass'},record(c));self.assertEqual(x['status'],'fail')
            self.assertTrue('unknown' in (x['live_process_status'],x['reaping_status']))
    def test_expected_orphan_only_exact_owned_pid(self):
        rec=record(clean([{'pid':11,'status':'sleeping'}]),17);rec['primary_error']={'error':'CHILD_EXIT_17'}
        x=case_assessment('orphan_parent',{},rec,expected_live_pids=[11])
        self.assertEqual(x['functional_status'],'not_run');self.assertEqual(x['live_process_status'],'fail')
        self.assertEqual(x['unexpected_live_process_status'],'pass');self.assertEqual(x['expected_fault_status'],'pass');self.assertEqual(x['status'],'pass')
        for allowed in (None,[12]):self.assertNotEqual(case_assessment('orphan_parent',{},rec,expected_live_pids=allowed)['status'],'pass')
        rec['cleanup']['before_cleanup'].append({'pid':12,'status':'sleeping'})
        self.assertEqual(case_assessment('orphan_parent',{},rec,expected_live_pids=[11])['status'],'fail')
    def test_expected_shutdown_stall_zombie_reaped(self):
        case={'status':'pass','controlled_actual_service_shutdown_stall':True,'primary_error_preserved':'INJECTED_PRIMARY_WORKER_ERROR','cleanup':{'cleanup_complete':True,'cleanup_errors':[{'error':'RPC_CONTROL_TIMEOUT shutdown'}]}}
        x=case_assessment('shutdown_timeout',case,record(clean([{'pid':11,'status':'zombie'}])))
        self.assertEqual((x['status'],x['functional_status'],x['expected_fault_status']),('pass','pass','pass'))
    def test_expected_worker_fault_zombie_reaped(self):
        case={'status':'pass','controlled_fault':'real parallel job missing snapshot','primary_error_preserved':'FileNotFoundError','finally_checkpoint_saved':True,'service_cleanup':{'cleanup_complete':True}}
        self.assertEqual(case_assessment('worker_failure',case,record(clean([{'pid':11,'status':'zombie'}])))['status'],'pass')
    def test_expected_failure_and_timeout_remain_raw_failure(self):
        rec=record(clean(),1);case={'status':'fail','error':'EXPECTED_INJECTED_SUBITEM_FAILURE'}
        x=case_assessment('controlled_failure',case,rec);self.assertEqual((x['status'],x['expected_fault_status']),('fail','pass'))
        rec=record(clean(),-15);rec.update(termination_requested=True,termination_reason='timeout')
        x=case_assessment('controlled_timeout',{},rec,True);self.assertEqual((x['status'],x['expected_fault_status']),('timeout','pass'))
        rec['termination_reason']='user_stop';self.assertNotEqual(case_assessment('controlled_timeout',{},rec,True)['expected_fault_status'],'pass')
    def test_stop_and_primary_errors_not_pass(self):
        self.assertFalse(task_assessment(0,True,None,clean())['accepted'])
        self.assertFalse(task_assessment(0,False,{'error':'monitor'},clean())['accepted'])
    def test_offline_sidecar_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cases/service').mkdir(parents=True)
            raw={'status':'pass'};state={'code_commit':'old','deadline_utc':'unchanged','execution_elapsed_seconds':159.57,'attempts':[dict(record(clean([{'pid':11,'status':'zombie'}])),label='service')]}
            for p,value in [(root/'state.json',state),(root/'coverage.json',{'paths':[{'path':'service','status':'fail'}]}),(root/'cases/service/case.json',raw)]:p.write_text(json.dumps(value))
            before={str(p):p.read_bytes() for p in root.rglob('*.json')}
            target=root/'sidecar.json';write_review(root,target);x=json.loads(target.read_text())
            self.assertTrue(x['classification_only']);self.assertFalse(x['experiments_started']);self.assertEqual(x['paths'][0]['assessment']['status'],'pass')
            self.assertEqual(x['elapsed_seconds_unchanged'],159.57)
            for p,data in before.items():self.assertEqual(Path(p).read_bytes(),data)
            with self.assertRaises(FileExistsError):write_review(root,target)
if __name__=='__main__':unittest.main()
