"""B worker forbids physical initial/mapping and grid traversal at runtime."""
import sys
from pathlib import Path


def install(repo):
    repo=Path(repo).resolve();denied=[];counts={'forbidden_grid_calls':0,'profiled_calls':0}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if not path.is_relative_to(repo):return
        rel=str(path.relative_to(repo))
        if path.suffix in ('.py','.pyc'):return
        forbidden=(rel.startswith('tasks/tower10/p49/acceptance/') or
          rel in ['tasks/tower10/p49/initial_topology.json','tasks/tower10/p49/map.json','tasks/tower10/p49/identity.json','tasks/tower10/p49/topology/display_mapping.json'] or
          '/learning/results/' in rel or '/topology/results/' in rel)
        if forbidden:denied.append(rel);raise PermissionError('CONTROLLED_B_FORBIDDEN_INPUT '+rel)
    def profile(frame,event,arg):
        if event!='call':return
        counts['profiled_calls']+=1;filename=frame.f_code.co_filename;name=frame.f_code.co_name
        if filename.endswith('path_index.py') or (filename.endswith('topology_engine.py') and name in {'paths','_walk','_membership','position','_direct_reachable'}):
            counts['forbidden_grid_calls']+=1;raise RuntimeError('CONTROLLED_B_GRID_DEPENDENCE '+name)
    sys.addaudithook(audit);sys.setprofile(profile)
    return denied,counts
