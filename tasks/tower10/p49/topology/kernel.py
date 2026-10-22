"""Object-topology storage binding around unchanged P48 rule primitives.

Original battle/item/door/lifecycle transaction code runs in a private kernel
namespace. Only routing, location bookkeeping and disabled membership change.
This adapter explicitly rejects nonempty corridor event programs.
"""
from collections import deque
from tasks.tower10.neural.vendor.kernel import _engine,_State,_plain


def component_and_frontier(state):
    graph=state['graph'];objects=state['objects'];frontier={};seen=set();todo=deque()
    anchors=state['hero']['anchors']
    if anchors:
        for k in anchors:todo.append((k,[]));seen.add(k)
    else:
        for k in graph['entry']:
            if objects[k]['status']=='active':frontier[k]=[k]
            else:todo.append((k,[k]));seen.add(k)
    while todo:
        k,path=todo.popleft()
        for q in graph['adjacency'][k]:
            if q in state['barriers']:continue
            if objects[q]['status']=='active':frontier.setdefault(q,path+[q]);continue
            if q not in seen:seen.add(q);todo.append((q,path+[q]))
    return seen,frontier


def paths(state):return component_and_frontier(state)[1]


def relocate(state,path,no_pass):
    # The interaction removes the target object in the same original atomic
    # transition. Its now-open component equals the pre-door noPass endpoint's
    # component. Physical walking/direction/steps are external replay metadata.
    state['hero']['anchors']=[path[-1]]
    return ['object:'+k for k in path]


def membership(state,key,enabled):
    disabled=set(state['flags'].get('__disabled_events__',[]))
    if enabled:disabled.discard(key)
    else:disabled.add(key)
    state['flags']['__disabled_events__']=sorted(disabled)


class Session:
    def __init__(self,initial):
        assert not initial['rules']['eventPrograms'] and not initial['rules']['stairs'] and not initial['rules']['fly']['floors']
        self.initial=_plain(initial)
        self.engine=_engine()
        self.engine.update(paths=paths,_walk=relocate,_membership=membership)

    def legal(self,state):return self.engine['legal_actions'](state)

    def step(self,state,action):
        result=self.engine['step'](_State(state),action)
        if result['result']['accepted']:
            s=result['state'];boss=s['task']['boss']
            if s['objects'][boss]['status']=='resolved':
                s['task']=dict(s['task'],status='GOAL');result['result']['goal']=True
                result['delta']['task']['status']={'before':'ACTIVE','after':'GOAL'}
        return result
