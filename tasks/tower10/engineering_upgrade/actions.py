"""Generation boundary: actual interactions, never stand-alone locomotion.

This restricts candidates, not the frozen game's transition semantics. Walking
and normal stairs remain inside the engine's route to an interaction target.
Every chosen transition must also pass require_progress before being recorded.
"""
POLICY_VERSION = 'interaction-progress-v1'
SHOP = 'o:MT4:6:1'
EVENT_TARGETS = frozenset({'o:MT10:6:2', 'o:MT10:6:5'})


def interaction_kind(state, action):
    if state['task']['status'] == 'GOAL':
        return None
    if not isinstance(action, dict) or set(action) - {'target', 'choices'}:
        return None
    target = action.get('target')
    obj = state['objects'].get(target)
    if not obj or obj['status'] != 'active' or obj.get('excluded'):
        return None
    trigger = obj['trigger']
    choices = action.get('choices')
    if target == SHOP:
        # A menu exit alone is not an interaction. Preserve actual purchases,
        # including valid multi-purchase protocol strings passed to step().
        if (isinstance(choices, list) and len(choices) > 1
                and choices[-1] == 3
                and all(type(c) is int and c in (0, 1, 2) for c in choices[:-1])):
            return 'shop_purchase'
        return None
    if trigger in ('battle', 'getItem', 'openDoor'):
        return {'battle': 'battle', 'getItem': 'pickup', 'openDoor': 'door'}[trigger]
    if trigger == 'trader':
        bought = target[2:].replace(':', '@') + '@A'
        if not state['flags'].get(bought):
            return 'key_trade' if choices == [0] else None
        return 'one_time_npc'
    if trigger == 'oldman':
        return 'one_time_npc'
    if trigger == 'action' and target in EVENT_TARGETS:
        return 'special_event'
    # Fail closed on unknown/pass/placeholder events and changeFloor targets.
    return None


def progress_candidates(state, legal_actions):
    """Filter the frozen legal list without adding movement or changing rules."""
    return [a for a in legal_actions if interaction_kind(state, a) is not None]


def require_progress(before, action, result):
    """Fail closed before appending an action, even if candidate filtering errs.

    Position, direction, steps, visited floors, display flags and generic event
    receipt counters are intentionally insufficient evidence of an interaction.
    """
    kind = interaction_kind(before, action)
    if kind is None:
        raise ValueError('NON_INTERACTION_ACTION: ' + repr(action))
    if not result['result']['accepted']:
        raise ValueError('REJECTED_INTERACTION: ' + repr(result['result']))
    after = result['state']
    target = action['target']
    old, new = before['objects'][target], after['objects'].get(target)
    consumed = new is None or new['status'] != old['status'] or new['entity'] != old['entity']
    if kind == 'shop_purchase':
        a, b = before['hero'], after['hero']
        progressed = (b['money'] < a['money']
                      and any(b[k] > a[k] for k in ('hp', 'atk', 'def'))
                      and after['flags'].get('times1', 0) > before['flags'].get('times1', 0))
    elif kind == 'key_trade':
        a, b = before['hero'], after['hero']
        progressed = b['money'] < a['money'] and b['items'] != a['items']
    elif kind == 'special_event':
        # Both supported special events remove their own one-time trigger;
        # ambush also moves/spawns monsters and doors in the same transaction.
        progressed = consumed
    else:
        progressed = consumed
    if not progressed:
        raise ValueError('NO_INTERACTION_PROGRESS: ' + repr(action))
    return kind
