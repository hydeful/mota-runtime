"""Accelerated P28 topology transitions without changing frozen game semantics.

``step`` is a fully independent mutable-output compatibility interface.
``prepare(initial).step`` is a branchable fast session: its graph/rules and
object event/position metadata are session-owned READ ONLY. All returned session structures are read only to callers. Transition writes
copy their affected records transactionally; use ``export`` before arbitrary external mutation.
There is no answer cache, JIT, native extension, or process-global monkeypatch.
"""
from __future__ import annotations
import copy
from contextvars import ContextVar
import hashlib
from pathlib import Path
from . import topology_engine as original

SOURCE_SHA256 = 'f602d445d832904eb54a030bc0bf8ebf4e9b9d76ea8f2860fccd34d8b07355d6'
if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError('P29.5 requires the pinned P28 topology engine')


def _clone(state):
    """Clone mutable transition fields; share only engine-read-only metadata."""
    out = dict(state)
    for key in ('hero','flags'):
        out[key] = copy.deepcopy(state[key])
    for key in ('objects','entities'):
        out[key] = _Tracked(state[key])
    for key in ('barriers','events','task'):
        out[key] = dict(state[key])
    return out



class _Tracked(dict):
    """Per-transition changed-key candidates, bounded by objects/entities.

    All nested writes in the pinned engine first pass through own()/spawn's
    map assignment. New entities are inserted by setdefault, overridden here.
    The set is fresh on each clone; it is neither a history nor an answer cache.
    """
    def __init__(self, source):
        super().__init__(source)
        self.touched = set()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.touched.add(key)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


def _delta(before, after):
    out = {}
    for key in ('hero','flags','objects','entities','barriers','events','task'):
        a,b=before[key],after[key]
        candidates=b.touched if isinstance(b,_Tracked) else set(a)|set(b)
        changes={}
        for k in sorted(candidates):
            va,vb=a.get(k),b.get(k)
            if va is vb:
                continue
            if va != vb:
                changes[k]={'before':copy.deepcopy(va),'after':copy.deepcopy(vb)}
        out[key]=changes
    return out


class _State(dict):
    pass


class _Copy:
    @staticmethod
    def deepcopy(value, memo=None):
        if isinstance(value,_State):
            return _clone(value)
        return copy.deepcopy(value,memo)


def _engine():
    # Separate globals let original helpers call each other with the optimized
    # copy implementation; the imported reference engine is never modified.
    source = Path(original.__file__).read_text()
    namespace = {'__name':__name__+'.semantics'}
    exec(compile(source,original.__file__,'exec'),namespace)
    namespace['copy'] = _Copy
    namespace['_delta'] = _delta
    remove=namespace['_remove'];spawn=namespace['_spawn']
    ambush=namespace['_ambush'];boss=namespace['_boss']
    def own(s,k):
        o=s['objects'].get(k)
        if o:
            s['objects'][k]=dict(o)
            eid=o['entity']
            if eid in s['entities']:s['entities'][eid]=dict(s['entities'][eid])
    def remove_owned(s,k):
        own(s,k)
        return remove(s,k)
    def spawn_owned(s,k,template,eid=None):
        effective=eid or 'born:'+k+':'+str(s['serial'])
        if effective in s['entities']:s['entities'][effective]=dict(s['entities'][effective])
        return spawn(s,k,template,eid)
    def ambush_owned(s):
        for move in s['rules']['ambush']['moves']:
            own(s,namespace['slot'](move['source']))
            eid=move['futureEntity']
            s['entities'][eid]=dict(s['entities'][eid])
        return ambush(s)
    def boss_owned(s):
        own(s,'o:MT10:6:11')
        return boss(s)
    namespace.update(_remove=remove_owned,_spawn=spawn_owned,_ambush=ambush_owned,_boss=boss_owned)
    return namespace

_ENGINE = _engine()


class Prepared:
    def __init__(self, initial):
        # Own all incoming metadata. Caller mutations cannot alter this session.
        self.initial = copy.deepcopy(initial)
        # Frozen dict subclasses from P28 intentionally refuse deepcopy: force
        # ordinary JSON-like containers for export and independent ownership.
        self.initial['graph'] = _plain(initial['graph'])
        self.initial['rules'] = _plain(initial['rules'])
        self.graph = self.initial['graph']
        self.rules = self.initial['rules']
        from .path_index import PathIndex
        self._engine = _engine()
        indexed_paths = PathIndex(self.graph).paths
        pending_paths = ContextVar('topology_step_paths', default=None)
        def paths(state):
            cell = pending_paths.get()
            if cell is None:
                return indexed_paths(state)
            if not cell:
                cell.append(indexed_paths(state))
            return cell[0]
        transition = self._engine['step']
        def transactional(state, action):
            # Pinned source calls paths only in refusal and immediately after
            # cloning, before _walk mutates anything. They see identical world
            # occupancy and position. Reuse is strictly this call, never across
            # transitions; ContextVar also isolates threads and nested calls.
            token = pending_paths.set([])
            try:
                return transition(state, action)
            finally:
                pending_paths.reset(token)
        self._engine.update(paths=paths, step=transactional)

    def step(self, state, action):
        # Accept P28 restore(initial,dynamic) as well as session state. Static
        # assets must be semantically equal; a different context is rejected.
        if state['graph'] is not self.graph or state['rules'] is not self.rules:
            if state['graph'] != self.graph or state['rules'] != self.rules:
                raise ValueError('prepared static context mismatch')
            state = dict(state,graph=self.graph,rules=self.rules)
        return self._engine['step'](_State(state),action)

    def export(self, state):
        """Materialize an entirely mutable independent complete topology state."""
        return _plain(state)

    def compile_actions(self, actions, state=None):
        state = self.initial if state is None else state
        route=[];records=[]
        for index, action in enumerate(actions):
            result=self.step(state,action)
            if not result['result']['accepted']:
                raise ValueError({'index':index,'action':action,'result':result['result'],'stateHash':original.digest(state)})
            state=result['state'];route.extend(result['routeSegment']);records.append(result['result'])
        return {'state':state,'route':route,'records':records}


def _plain(value):
    if type(value) in (str,int,float,bool,type(None)):return value
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,list):return [_plain(v) for v in value]
    if isinstance(value,tuple):return tuple(_plain(v) for v in value)
    return copy.deepcopy(value)


def prepare(initial_state):
    return Prepared(initial_state)


def step(state, action):
    # Full compatibility includes arbitrary mutation of returned static/event
    # structures; exporting is therefore required and measured explicitly.
    result = _ENGINE['step'](_State(state),action)
    result['state'] = _plain(result['state'])
    return result

execute=step
