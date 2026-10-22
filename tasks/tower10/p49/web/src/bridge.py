"""Browser adapter: all gameplay calls the unmodified original Python engine."""
import copy
import json
import topology_engine as engine

initial = json.load(open('/initial.json'))
state = copy.deepcopy(initial)
history = []
records = []


def snapshot():
    actions = engine.legal_actions(state)
    terminal = None
    if state['task']['status'] == 'GOAL':
        terminal = {'end': 'GOAL', 'score': state['hero']['hp']}
    elif not actions:
        terminal = {'end': 'T1_NO_INTERACTION', 'score': None}
    frontier = engine.paths(state) if terminal is None else {}
    objects = []
    for key, o in state['objects'].items():
        v = {'key': key, 'x': o['position'][1], 'y': o['position'][2],
             'id': o['event']['id'], 'trigger': o['trigger'], 'status': o['status'],
             'reachable': key in frontier, 'reason': engine.refusal(state, key)}
        if o['trigger'] == 'battle':
            v['enemy'] = state['rules']['enemies'][v['id']]
            v['damage'] = engine.damage(state, v['id'])
        objects.append(v)
    return json.dumps({'hero': state['hero'], 'objects': objects, 'actions': actions,
                       'terminal': terminal, 'actionCount': state['actionCount'], 'records': records}, ensure_ascii=False)


def interact(key):
    global state
    before = state
    result = engine.step(state, {'target': key})
    if not result['result']['accepted']:
        raise ValueError(result['result']['reason'])
    state = result['state']
    if state['objects'][state['task']['boss']]['status'] == 'resolved':
        state['task']['status'] = 'GOAL'
    history.append(before)
    records.append({'key': key, 'id': before['objects'][key]['event']['id'],
                    'hp_before': before['hero']['hp'], 'hp_after': state['hero']['hp'],
                    'state_sha256': engine.digest(state), 'route': result['routeSegment']})
    return snapshot()


def restart():
    global state
    state = copy.deepcopy(initial); history.clear(); records.clear()
    return snapshot()


def undo():
    global state
    if history:
        state = history.pop(); records.pop()
    return snapshot()
