"""Initialization-only contraction. Never imported by the actor or learner."""
import copy
import json
from collections import deque
from pathlib import Path
from tasks.tower10.neural.vendor import topology_engine as reference

HERE=Path(__file__).resolve().parent


def build():
    source=json.loads((HERE.parent/'initial_topology.json').read_text())
    pinned=json.loads((HERE.parent/'identity.json').read_text())
    assert reference.digest(source)==pinned['initial_sha256']
    assert not source['rules']['eventPrograms'] and not source['rules']['stairs'] and not source['rules']['fly']['floors']
    assert not source['barriers']
    assert all(o['trigger'] in ('battle','openDoor','getItem') for o in source['objects'].values())
    order=sorted(source['objects']);ids={key:f'e{i:02d}' for i,key in enumerate(order)}
    by_vertex={reference.vertex(o['position']):key for key,o in source['objects'].items()}
    out=source['graph']['out']
    def frontier(start):
        todo=deque([start]);seen={start};found=set()
        while todo:
            v=todo.popleft()
            for edge in out.get(v,[]):
                q=edge['target']
                if q in seen:continue
                seen.add(q)
                if q in by_vertex:found.add(ids[by_vertex[q]])
                else:todo.append(q)
        return sorted(found)
    adjacency={ids[k]:frontier(reference.vertex(source['objects'][k]['position'])) for k in order}
    for i,neighbors in adjacency.items():
        if i in neighbors:neighbors.remove(i)
        for j in neighbors:assert i in adjacency[j]
    graph={'order':[ids[k] for k in order],'adjacency':adjacency,'entry':frontier(reference.vertex(reference.position(source)))}
    objects={};entities={};mapping=[]
    for k in order:
        event_id=ids[k];o=copy.deepcopy(source['objects'][k]);entity='entity:'+event_id
        mapping.append({'node':event_id,'reference_slot':k,'display_position':o['position']})
        # Only the constant floor context is needed by the original item-ratio
        # primitive. No x/y, map dimensions or grid vertices survive contraction.
        o.update(id=event_id,entity=entity,position=['MT1'])
        objects[event_id]=o
        entities[entity]={'status':'active','slot':event_id,'enemy':o['event']['id']}
    rules={k:copy.deepcopy(source['rules'][k]) for k in ['enemies','items','values','ratios','eventPrograms','stairs','fly','templates']}
    rules['version']='P49-object-topology-v1/P48-rule-primitives'
    hero={k:copy.deepcopy(source['hero'][k]) for k in ['hp','atk','def','money','items','experience','floorId']};hero['anchors']=[]
    state={'schema':'P49-object-topology-v1','hero':hero,'flags':{},'graph':graph,'rules':rules,
           'objects':objects,'entities':entities,'barriers':{},'events':{},'serial':0,'actionCount':0,
           'task':{'status':'ACTIVE','id':'p49-object-topology-learning-v1','boss':ids[source['task']['boss']]}}
    identity={'physical_initial_sha256':reference.digest(source),'physical_rules_sha256':reference.digest(source['rules']),
              'topology_initial_sha256':reference.digest(state),'topology_rules_sha256':reference.digest(rules),
              'scope':'single frozen P49 task; no corridor-triggered events, stairs or fly; unsupported event programs fail at build',
              'contraction':'object-to-object direct corridors; initial entry is a frontier vector, never a graph/decision node',
              'object_nodes':len(objects),'runtime_grid_vertices':0,'relations':sum(map(len,adjacency.values()))}
    for name,data in [('initial.json',state),('identity.json',identity),('display_mapping.json',mapping)]:
        (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(identity,ensure_ascii=False),flush=True)


if __name__=='__main__':build()
