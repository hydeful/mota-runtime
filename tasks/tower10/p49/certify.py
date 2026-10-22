"""Independent map acceptance only. No output is read by any learning runner."""
import argparse
import json
import math
import random
import time
from pathlib import Path
from .environment import Environment, HERE, reference


def replay(env, actions):
    fast = env.initial; slow = env.initial; records = []
    for a in actions:
        assert env.terminal(fast) is None, 'POST_TERMINAL_ACTION'
        assert env.legal(fast) == reference.legal_actions(slow)
        old_hash = reference.digest(fast)
        f = env.step(fast, a); r = env.step(slow, a, independent=True)
        assert reference.digest(fast) == old_hash, 'INPUT_MUTATION'
        assert f == r, 'FULL_TRANSITION_DIFFERENCE'
        assert f['result']['accepted']
        records.append({'action': a, 'hp_before': fast['hero']['hp'],
                        'hp_after': f['state']['hero']['hp'], 'state_sha256': reference.digest(f['state'])})
        fast = f['state']; slow = r['state']
    assert env.terminal(fast) is not None
    assert not env.legal(fast)
    rejected = env.step(fast, {'target': env.boss})
    assert not rejected['result']['accepted'] and rejected['state'] == fast
    return {'terminal': env.terminal(fast), 'records': records}


def wilson(k, n):
    z = 1.959963984540054; p = k/n; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0, center-half), min(1, center+half)]


def sample(n=30000, seeds=(49101, 49102, 49103)):
    env = Environment(); batches = []; witness = None; failures = None
    for seed in seeds:
        rng = random.Random(seed); count = n//len(seeds); wins = 0; scores = []; lengths = []
        t = time.monotonic()
        for i in range(count):
            state = env.initial; actions = []
            while (end := env.terminal(state)) is None:
                legal = env.legal(state)
                # One action per object in this map, hence target-uniform too.
                assert len({a['target'] for a in legal}) == len(legal)
                a = rng.choice(legal); actions.append(a)
                state = env.step(state, a)['state']
                assert len(actions) <= len(env.initial['objects']), 'NON_PROGRESS_LOOP'
            lengths.append(len(actions))
            if end['end'] == 'GOAL':
                wins += 1; scores.append(end['score'])
                if witness is None:
                    witness = replay(env, actions)
            elif failures is None:
                failures = replay(env, actions)
            if i < 20:
                replay(env, actions)
            if (i+1) % 1000 == 0:
                print(json.dumps({'seed': seed, 'done': i+1, 'wins': wins, 'elapsed': round(time.monotonic()-t, 2)}), flush=True)
        batches.append({'seed': seed, 'games': count, 'wins': wins,
                        'min_goal_hp': min(scores) if scores else None,
                        'max_goal_hp': max(scores) if scores else None,
                        'mean_goal_hp': sum(scores)/len(scores) if scores else None,
                        'min_length': min(lengths), 'max_length': max(lengths)})
    total = sum(x['games'] for x in batches); wins = sum(x['wins'] for x in batches)
    return {'kind': 'uniform_legal_effective_interactions_no_MCTS_no_training',
            'initial_sha256': reference.digest(env.initial), 'rules_sha256': reference.digest(env.initial['rules']),
            'games': total, 'wins': wins, 'rate': wins/total, 'wilson95': wilson(wins, total),
            'batches': batches, 'first_success_independent_replay': witness,
            'first_failure_independent_replay': failures,
            'reference_replay_scope': 'first 20 full episodes per seed and first overall success/failure'}


def exhaustive():
    env = Environment(); slots = sorted(env.initial['objects']); bits = {k: 1 << i for i, k in enumerate(slots)}
    assert not env.initial['rules']['eventPrograms'] and not env.initial['rules']['stairs']
    assert all(o['trigger'] in ('getItem', 'battle', 'openDoor') for o in env.initial['objects'].values())
    memo = {}; choices = {}; count = 0
    def key(s):
        h = s['hero']
        # Resolved set determines atk/def, keys, money, entities and task.
        # HP and location remain order dependent. Direction/steps and flags
        # are bookkeeping only here: no event predicates/stairs/fly exist.
        mask = sum(bits[k] for k, o in s['objects'].items() if o['status'] == 'resolved')
        return mask, h['hp'], h['x'], h['y']
    def solve(s):
        nonlocal count
        k = key(s)
        if k in memo:
            return memo[k]
        count += 1
        if count % 10000 == 0:
            print(json.dumps({'exhaustive_states_started': count, 'memo_complete': len(memo)}), flush=True)
        end = env.terminal(s)
        if end:
            hp = end['score']; value = (int(hp is not None), hp, hp, 1, int(hp is not None))
        else:
            actions = env.legal(s); children = []
            for a in actions:
                r = env.step(s, a); assert r['result']['accepted']
                assert key(r['state'])[0].bit_count() == k[0].bit_count()+1
                children.append((a, solve(r['state'])))
            viable = [(a, v) for a, v in children if v[1] is not None]
            minimum = min(viable, key=lambda av: av[1][1]) if viable else None
            maximum = max(viable, key=lambda av: av[1][2]) if viable else None
            choices[k] = (minimum[0] if minimum else None, maximum[0] if maximum else None)
            # First field is the probability under target-uniform random play.
            # min/max range is across all successful complete trajectories.
            value = (math.fsum(v[0] for _, v in children)/len(children),
                     minimum[1][1] if minimum else None, maximum[1][2] if maximum else None,
                     sum(v[3] for _, v in children), sum(v[4] for _, v in children))
        memo[k] = value
        return value
    t = time.monotonic(); result = solve(env.initial)
    routes = []
    for choice in (0, 1):
        s = env.initial; route = []
        while env.terminal(s) is None:
            a = choices[key(s)][choice]
            if a is None:
                break
            route.append(a); s = env.step(s, a)['state']
        routes.append(replay(env, route))
    return {'kind': 'exhaustive_DAG_acceptance_only_no_training',
            'initial_sha256': reference.digest(env.initial), 'rules_sha256': reference.digest(env.initial['rules']),
            'states': len(memo), 'seconds': time.monotonic()-t,
            'uniform_random_success_probability': result[0],
            'min_success_hp': result[1], 'max_success_hp': result[2],
            'complete_action_sequences': result[3], 'successful_action_sequences': result[4],
            'all_legal_sequences_win': result[3] == result[4],
            'minimum_hp_reference_replay': routes[0], 'maximum_hp_reference_replay': routes[1],
            'reduction_contract': 'resolved bitset, HP, x, y; no event predicates; remaining resources derived from resolved set'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['sample', 'exhaustive'], default='sample')
    p.add_argument('--games', type=int, default=30000); args = p.parse_args()
    report = sample(args.games) if args.mode == 'sample' else exhaustive()
    dest = HERE/'acceptance'; dest.mkdir(exist_ok=True)
    path = dest/(args.mode+'.json'); path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if 'replay' not in k}, ensure_ascii=False), flush=True)
