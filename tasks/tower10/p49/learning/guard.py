"""Keep acceptance answers, old models and unrelated behavior files out of actors."""
import sys
from pathlib import Path


def install(root,run_root):
    root=Path(root).resolve();run_root=Path(run_root).resolve()
    allowed={root/p for p in [
        'tasks/tower10/p49/initial_topology.json','tasks/tower10/p49/identity.json',
        'tasks/tower10/p49/environment.py','tasks/tower10/p49/__init__.py',
        'tasks/tower10/p45/runtime.py','tasks/tower10/p45/config.json','tasks/tower10/p45/network.py',
        'tasks/tower10/p45/search.py','tasks/tower10/p45/contracts.py','tasks/tower10/p45/__init__.py',
        'tasks/tower10/neural/common.py','tasks/tower10/neural/vendor/kernel.py',
        'tasks/tower10/neural/vendor/path_index.py','tasks/tower10/neural/vendor/topology_engine.py',
        'tasks/tower10/engineering_upgrade/actions.py','tasks/tower10/engineering_upgrade/topology.py','tasks/tower10/engineering_upgrade/fast.py']}
    learner=root/'tasks/tower10/p49/learning'
    denied=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if not path.is_relative_to(root):return
        if path.is_relative_to(run_root):return
        if path.suffix=='.pyc' and path.parent.name=='__pycache__':path=path.parent.parent/(path.name.split('.')[0]+'.py')
        if path in allowed or (path.is_relative_to(learner) and path.suffix=='.py' and path.name!='report.py'):return
        denied.append(str(path));raise PermissionError('P49_OLD_BEHAVIOR_OR_ACCEPTANCE_BLOCKED '+str(path))
    sys.addaudithook(guard)
    return denied
