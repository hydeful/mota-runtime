"""Compare original P45 and extracted P45, through real trigger-driven SGD."""
import argparse, datetime, json, os, pathlib, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--original', required=True)
p.add_argument('--extracted', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = pathlib.Path(a.output).resolve()
out.mkdir(exist_ok=False, parents=True)
config = json.loads((pathlib.Path(a.original)/'tasks/tower10/p45/config.json').read_text())
now = datetime.datetime.now(datetime.timezone.utc)
config['budget'] = {'session_start_utc': now.isoformat(), 'deadline_utc': (now+datetime.timedelta(seconds=3600)).isoformat(), 'budget_seconds': 3600, 'authorization': 'P48 fixed-workload comparison only; stop after first update plus four complete episodes'}
cp = out/'config.json'
cp.write_text(json.dumps(config))
results = {}
for label, source in [('original', a.original), ('extracted', a.extracted)]:
    root = out/label
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(source).resolve()), PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def call(command, *extra):
        with (out/(label+'.log')).open('a') as f:
            subprocess.run([sys.executable, '-B', '-m', 'tasks.tower10.p45_v0', command, '--run', str(root), *map(str, extra)], env=env, cwd=source, stdout=f, stderr=subprocess.STDOUT, check=True)
    call('init', '--seed', 4801, '--run-id', 'P48-fixed', '--config', cp)
    call('collect', '--seconds', 1600, '--until-generation', 1)
    s = json.loads((root/'status.json').read_text())
    assert s['generation'] == 1 and s['sgd_steps'] > 0, s
    call('collect', '--seconds', 300, '--cohorts', 1, '--no-train')
    call('fingerprint')
    results[label] = json.loads((root/'logical-fingerprint.json').read_text())
    (out/(label+'-summary.json')).write_text(json.dumps({'fingerprint': results[label], 'status': json.loads((root/'status.json').read_text())}, indent=2))
    print(label+' complete', flush=True)
assert results['original'] == results['extracted'], {'original':results['original'], 'extracted':results['extracted']}
receipt = {'pass': True, 'seed':4801, 'scope':'first real trigger-driven model update, then four complete episodes consuming N1; original P45 versus byte-identical extraction', 'comparison':'exact logical fingerprints: actions, search targets/pi/visits, terminal labels, model/optimizer/RNG/replay and counters; only measured costs/timestamps excluded by original fingerprint', 'fingerprint':results['original']}
(out/'COMPARISON.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
