"""Map/task adapter over the unchanged P48 reference and prepared kernel.

Only map/configuration and the registered goal differ. No battle, reachability,
inventory, event sequencing or object lifecycle implementation is replaced.
"""
import copy
import hashlib
import json
from pathlib import Path

from tasks.tower10.neural.vendor import topology_engine as reference
from tasks.tower10.neural.vendor.kernel import prepare
from tasks.tower10.engineering_upgrade.actions import progress_candidates, require_progress

HERE = Path(__file__).resolve().parent
SOURCE_COMMIT = '0f732f99d9c0db093258fa807a1dbd4e879fc111'
ENGINE_SHA = 'f602d445d832904eb54a030bc0bf8ebf4e9b9d76ea8f2860fccd34d8b07355d6'


def build(config=None):
    cfg = json.loads((HERE / 'map.json').read_text()) if config is None else copy.deepcopy(config)
    if hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() != ENGINE_SHA:
        raise ValueError('P48_ENGINE_IDENTITY_MISMATCH')
    # Read static rule data only, never trajectories, old models or answers.
    assets = HERE.parent / 'p45/assets'
    source_bytes = (assets/'shared_initial_topology.json').read_bytes()
    expected = json.loads((assets/'SHA256.json').read_text())['shared_initial_topology.json']
    if hashlib.sha256(source_bytes).hexdigest() != expected:
        raise ValueError('P48_STATIC_RULE_DATA_IDENTITY_MISMATCH')
    source = json.loads(source_bytes)
    rows = cfg['rows']; width = len(rows[0])
    assert width == 9 and len(rows) == 5 and all(len(row) == width for row in rows)
    assert sum(row.count('@') for row in rows) == sum(row.count('M') for row in rows) == 1
    floor = 'MT1'; objects = {}; entities = {}; nodes = {}; edges = {}
    templates = {o['event']['id']: o for o in source['rules']['templates'].values()}
    templates.update({o['event']['id']: o for o in source['objects'].values()})
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol == '#':
                continue
            pos = [floor, x, y]; v = reference.vertex(pos)
            nodes[v] = {'position': pos, 'role': 'anchor'}; edges[v] = []
            if symbol == '@':
                start = pos
            eid = cfg['enemies'].get(symbol) or cfg['items'].get(symbol) or ('yellowDoor' if symbol == 'Y' else None)
            if not eid:
                assert symbol in '.@X'
                continue
            obj = copy.deepcopy(templates[eid]); key = reference.slot(pos); entity = 'initial:' + key
            obj.update(id=key, position=pos, entity=entity, status='active', excluded=False)
            objects[key] = obj
            entities[entity] = {'slot': key, 'status': 'active', 'enemy': eid}
            if symbol == 'M':
                boss = key
    for v, node in nodes.items():
        floor, x, y = node['position']
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            dest = [floor, x + dx, y + dy]; target = reference.vertex(dest)
            if target in nodes:
                edges[v].append({'target': target, 'path': [node['position'], dest]})
    r = source['rules']
    rules = {
        'version': 'P49-map-v1/P48-rules', 'widths': {floor: width},
        'enemies': {k: copy.deepcopy(r['enemies'][k]) for k in sorted(set(cfg['enemies'].values()))},
        'items': {k: copy.deepcopy(r['items'][k]) for k in sorted(set(cfg['items'].values()))},
        'values': copy.deepcopy(r['values']), 'ratios': {floor: 1},
        'templates': {}, 'stairs': [], 'eventPrograms': [],
        'fly': {'floors': {}},
    }
    # Fresh hero/flags: no P48 starting inventory or historical progress.
    hero = {k: copy.deepcopy(v) for k, v in source['hero'].items()}
    hero.update(hp=cfg['hero']['hp'], atk=cfg['hero']['atk'], **{'def': cfg['hero']['def']},
                floorId=floor, x=start[1], y=start[2], direction='up', money=0, steps=0,
                items={'tools': {}, 'constants': {}, 'equips': {}}, equipment=[], followers=[])
    if cfg['hero']['yellowKey']:
        hero['items']['tools']['yellowKey'] = cfg['hero']['yellowKey']
    state = {'schema': 'P49-topology-state-v1', 'hero': hero, 'flags': {},
             'graph': {'nodes': nodes, 'out': edges}, 'rules': rules,
             'objects': objects, 'entities': entities, 'barriers': {}, 'events': {},
             'serial': 0, 'actionCount': 0, 'task': {'status': 'ACTIVE', 'id': cfg['id'], 'boss': boss}}
    return state


class Environment:
    def __init__(self, initial=None):
        self.raw = prepare(build() if initial is None else initial)
        self.initial = self.raw.initial
        self.boss = self.initial['task']['boss']

    def legal(self, state):
        return progress_candidates(state, reference.legal_actions(state))

    def terminal(self, state):
        if state['task']['status'] == 'GOAL':
            assert state['objects'][self.boss]['status'] == 'resolved' and state['hero']['hp'] > 0
            return {'end': 'GOAL', 'score': state['hero']['hp']}
        if not self.legal(state):
            return {'end': 'T1_NO_INTERACTION', 'score': None}
        return None

    def step(self, state, action, *, independent=False):
        result = reference.step(state, action) if independent else self.raw.step(state, action)
        if result['result']['accepted']:
            require_progress(state, action, result)
        # Explicit task adapter, after all original transition/automatic events.
        if result['result']['accepted'] and result['state']['objects'][self.boss]['status'] == 'resolved':
            s = result['state']; before = copy.deepcopy(s['task'])
            s['task'] = dict(s['task'], status='GOAL')
            result['result']['goal'] = True
            result['delta']['task']['status'] = {'before': before['status'], 'after': 'GOAL'}
        return result


def view(state):
    """Serializable view for both the Python HTTP UI and browser Python UI."""
    env = Environment.__new__(Environment); env.boss = state['task']['boss']
    actions = reference.legal_actions(state)
    frontier = reference.paths(state) if state['task']['status'] != 'GOAL' else {}
    objects = []
    for k, obj in state['objects'].items():
        o = {'key': k, 'x': obj['position'][1], 'y': obj['position'][2],
             'id': obj['event']['id'], 'status': obj['status'], 'trigger': obj['trigger'],
             'reachable': k in frontier, 'reason': reference.refusal(state, k)}
        if obj['trigger'] == 'battle':
            o['enemy'] = state['rules']['enemies'][o['id']]
            o['damage'] = reference.damage(state, o['id'])
        objects.append(o)
    return {'hero': state['hero'], 'objects': objects, 'actions': actions,
            'terminal': env.terminal(state), 'actionCount': state['actionCount']}
