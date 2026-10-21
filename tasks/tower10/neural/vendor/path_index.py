"""Prepared graph index preserving original BFS edge and path order.

No transition-result cache. Index belongs to one immutable graph only.
"""
from collections import deque

class PathIndex:
    def __init__(self, graph):
        self.graph=graph
        self.out={p:tuple((edge['target'], 'o:'+':'.join(map(str,graph['nodes'][edge['target']]['position'])), edge['path'][1:]) for edge in edges) for p,edges in graph['out'].items()}

    def paths(self,state):
        hero=state['hero']; start='v:'+str(hero['floorId'])+':'+str(hero['x'])+':'+str(hero['y'])
        queue=deque([(start,[])]); seen={start};found={}
        # Membership alone varies; topology strings and corridor tails are static.
        objects=state['objects']; barriers=state['barriers']
        while queue:
            p,trail=queue.popleft()
            for q,k,tail in self.out.get(p,()):
                if k in barriers: continue
                path=trail+tail
                obj=objects.get(k)
                if obj and obj['status']=='active':
                    if q!=start: found.setdefault(k,path)
                    continue
                if q not in seen:seen.add(q);queue.append((q,path))
        return found
