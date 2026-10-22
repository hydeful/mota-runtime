"""Block physical-map input/old answers and monitor forbidden runtime helpers."""
import sys
from pathlib import Path


def install(repo,run_root):
    repo=Path(repo).resolve();run_root=Path(run_root).resolve();here=Path(__file__).resolve().parent
    allowed={repo/p for p in ['tasks/tower10/p45/runtime.py','tasks/tower10/p45/config.json','tasks/tower10/p45/network.py','tasks/tower10/p45/search.py','tasks/tower10/p45/contracts.py','tasks/tower10/neural/common.py','tasks/tower10/neural/vendor/kernel.py','tasks/tower10/neural/vendor/topology_engine.py','tasks/tower10/p49/learning/evaluation.py']}
    allowed.update({here/'initial.json',here/'identity.json'})
    denied=[];counts={'forbidden_grid_calls':0,'profiled_python_calls':0}
    def file_guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if path.is_relative_to(run_root):return
        if not path.is_relative_to(repo):return
        if path.suffix=='.pyc' and path.parent.name=='__pycache__':path=path.parent.parent/(path.name.split('.')[0]+'.py')
        if path in allowed:return
        if path.parent==here and path.suffix=='.py' and path.name not in ('build.py','certify.py','report.py','preflight.py'):return
        denied.append(str(path));raise PermissionError('TOPOLOGY_FORBIDDEN_PHYSICAL_OR_OLD_INPUT '+str(path))
    def profile(frame,event,arg):
        if event!='call':return
        counts['profiled_python_calls']+=1
        filename=frame.f_code.co_filename;name=frame.f_code.co_name
        if filename.endswith('path_index.py') or (filename.endswith('topology_engine.py') and name in {'paths','_walk','_membership','position','_direct_reachable'}):
            counts['forbidden_grid_calls']+=1;raise RuntimeError('TOPOLOGY_RUNTIME_GRID_DEPENDENCE '+name)
    sys.addaudithook(file_guard);sys.setprofile(profile)
    return denied,counts
