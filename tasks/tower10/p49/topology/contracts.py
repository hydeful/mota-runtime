"""Original visit/terminal/RNG contracts, with exact object-action dimension."""
from tasks.tower10.p45.contracts import validate_record as original_record,validate_episode as original_episode


def validate_record(*args,**kwargs):
    original_record(*args,**kwargs)
    if any(a>=21 for a in args[0]['legal_action_ids']):raise ValueError('TOPOLOGY_ACTION_RANGE')
    return True


def validate_episode(env,episode,reference=False):
    original_episode(env,episode,reference=reference)
    if episode['schema']!=env.schema_hash:raise ValueError('TOPOLOGY_EPISODE_SCHEMA')
    for r in episode['records']:
        if any(a>=21 for a in r['legal_action_ids']):raise ValueError('TOPOLOGY_ACTION_RANGE')
    return True
