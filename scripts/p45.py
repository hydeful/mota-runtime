#!/usr/bin/env python3
"""Linux deployment control. All learning remains in the byte-identical P45 core."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get('P45_HOME', '/root/autodl-tmp/p45')).resolve()

def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default

def write(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)

def revision():
    return subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()

def env():
    return dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

def core(command, root, *args, **kwargs):
    return subprocess.run([sys.executable, '-B', '-m', 'tasks.tower10.p45_v0', command, '--run', str(root), *map(str, args)], cwd=SOURCE, env=env(), check=True, **kwargs)

def lock(root):
    f = (root / 'control.lock').open('a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise RuntimeError('ALREADY_RUNNING: same run is locked')
    return f

def running(root):
    try:
        f = lock(root)
        f.close()
        return False
    except RuntimeError:
        return True

def organize(root):
    """Keep P45's canonical paths, place immutable artifacts in separate folders."""
    for folder, pattern in [('models', root.name+'-N*.pt'), ('data', 'episode-*.json.gz'), ('data', 'training-N*.json')]:
        dest = root / folder
        dest.mkdir(exist_ok=True)
        for src in (root / 'state').glob(pattern):
            if src.is_symlink():
                continue
            target = dest / src.name
            if target.exists():
                raise RuntimeError('ARTIFACT_DESTINATION_EXISTS ' + str(target))
            # Hard-link the complete immutable publication, then atomically install
            # a relative link. The P45 reader always sees complete original bytes.
            os.link(src, target)
            temp = src.with_name(src.name + '.p48-link')
            temp.symlink_to(os.path.relpath(target, src.parent))
            os.replace(temp, src)

def check(root=None):
    import platform
    import shutil
    import torch
    import numpy as np
    for x in read(SOURCE / 'SOURCE_MANIFEST.json')['files']:
        p = SOURCE / x['destination']
        if hashlib.sha256(p.read_bytes()).hexdigest() != x['sha256']:
            raise RuntimeError('SOURCE_HASH_MISMATCH ' + str(p))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from tasks.tower10.p45.environment import Environment
    from tasks.tower10.p45.network import Network
    e = Environment()
    n = Network(e.config['network'])
    with torch.no_grad():
        logits, v = n.eval()(torch.from_numpy(e.observation.encode(e.initial, 0)[None]))
    if logits.shape != (1, 713) or not torch.isfinite(logits).all() or not torch.isfinite(v).all():
        raise RuntimeError('NETWORK_SELF_CHECK')
    def txt(p):
        q = Path(p)
        return q.read_text().strip() if q.exists() else None
    cpu = [s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')]
    result = {'check': 'PASS', 'source_commit': revision(), 'source': str(SOURCE), 'home': str(HOME), 'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__, 'os': platform.platform(), 'os_release': txt('/etc/os-release'), 'cpu_model': cpu[0] if cpu else None, 'visible_cpus': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0)), 'cgroup': txt('/proc/self/cgroup'), 'cpu_max': txt('/sys/fs/cgroup/cpu.max'), 'cpu_quota_v1': txt('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), 'cpu_period_v1': txt('/sys/fs/cgroup/cpu/cpu.cfs_period_us'), 'memory_max': txt('/sys/fs/cgroup/memory.max'), 'memory_limit_v1': txt('/sys/fs/cgroup/memory/memory.limit_in_bytes'), 'meminfo': txt('/proc/meminfo'), 'disk_bytes': shutil.disk_usage(HOME)._asdict(), 'mount': subprocess.run(['findmnt', '-T', str(HOME), '-J'], capture_output=True, text=True).stdout, 'torch_cuda_build': torch.version.cuda, 'gpu_visible': torch.cuda.is_available(), 'actual_compute_device': str(next(n.parameters()).device), 'search_device': 'cpu', 'inference_device': 'cpu', 'sgd_device': 'cpu', 'gpu': subprocess.run(['sh', '-c', 'command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'], capture_output=True, text=True).stdout}
    write(HOME / 'environment.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def status(root):
    m = read(root / 'run.json')
    c = read(root / 'control.json', {})
    s = read(root / 'state/status.json', {})
    active = running(root)
    state = 'running' if active else c.get('state', 'new')
    if not active and state in ('starting', 'running'):
        state = 'abnormal_exit'
    end = dt.datetime.fromisoformat(m['deadline_utc']).timestamp()
    models = read(root / 'state/models.json', [])
    log = root / 'logs/console.log'
    consumed = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get('event') == 'decision' and '-N0' not in x.get('model_id', '-N0'):
                consumed.append({k: x[k] for k in ('episode_id', 'step', 'model_id', 'weight_hash')})
    result = {'state': state, 'source_commit': m['source_commit'], 'actual_compute_device': 'cpu', 'seed': m['seed'], 'budget_seconds': m['budget_seconds'], 'deadline_utc': m['deadline_utc'], 'remaining_wall_budget_seconds': max(0, end - time.time()), 'elapsed_core_seconds': s.get('elapsed_seconds', 0), 'completed_episodes': len(s.get('episodes', [])), 'qualified_training_positions': s.get('total_positions', 0), 'model_updates': s.get('generation', 0), 'real_sgd_steps': s.get('sgd_steps', 0), 'decisions': s.get('total_decisions', 0), 'model_id': s.get('model_id'), 'model_hash': models[-1]['hash'] if models else None, 'updated_model_consumption_records': len(consumed), 'first_updated_model_consumption': consumed[:1], 'invocation': c, 'log': str(log), 'checkpoint': str(root / 'state/checkpoint.pt'), 'models': str(root / 'models'), 'models_ledger': str(root / 'state/models.json'), 'training_data': str(root / 'data'), 'last_durable_core_status': s}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def worker(root, seconds, resume):
    with lock(root) as held:
        m = read(root / 'run.json')
        if m['source_commit'] != revision():
            raise RuntimeError('SOURCE_COMMIT_CHANGED: use the original installation')
        deadline = dt.datetime.fromisoformat(m['deadline_utc']).timestamp()
        if deadline <= time.time():
            raise RuntimeError('BUDGET_EXHAUSTED: resume never resets budget')
        if (root / 'state/STOP').exists() and not resume:
            raise RuntimeError('STOPPED: use resume')
        if resume:
            (root / 'state/STOP').unlink(missing_ok=True)
        began = time.time()
        c = {'state': 'running', 'pid': os.getpid(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'requested_seconds': seconds, 'core_returncode': None}
        write(root / 'control.json', c)
        def stop(sig, frame):
            (root / 'state/STOP').touch()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        remaining = max(0, min(seconds, deadline - time.time()))
        # The child inherits the flock, so supervisor death cannot permit a second writer.
        p = subprocess.Popen([sys.executable, '-B', '-m', 'tasks.tower10.p45_v0', 'collect', '--run', str(root / 'state'), '--seconds', str(remaining)], cwd=SOURCE, env=env(), pass_fds=(held.fileno(),))
        while p.poll() is None:
            organize(root)
            time.sleep(.5)
        rc = p.wait()
        organize(root)
        ended = time.time()
        c.update(core_returncode=rc, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), actual_invocation_seconds=ended-began, post_budget_cleanup_seconds=max(0, ended-min(began+seconds, deadline)), state='abnormal_exit' if rc else ('stopped' if (root / 'state/STOP').exists() else ('budget_finished' if ended >= deadline else 'slice_finished')))
        write(root / 'control.json', c)
        if rc:
            raise SystemExit(rc)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['check', 'new', 'start', 'status', 'stop', 'resume', 'diagnose', '_worker'])
    p.add_argument('name', nargs='?')
    p.add_argument('--seed', type=int)
    p.add_argument('--budget', type=float)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--resume-worker', action='store_true')
    a = p.parse_args()
    HOME.mkdir(parents=True, exist_ok=True)
    if a.command == 'check':
        check(); return
    if not a.name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', a.name):
        p.error('a simple run name is required')
    root = HOME / 'runs' / a.name
    if a.command == 'new':
        if a.seed is None or a.budget is None or not 0 < a.budget <= 31536000:
            p.error('new requires --seed and a positive --budget (seconds)')
        root.mkdir(parents=True, exist_ok=False)
        (root / 'logs').mkdir()
        config = read(SOURCE / 'tasks/tower10/p45/config.json')
        now = dt.datetime.now(dt.timezone.utc)
        deadline = now + dt.timedelta(seconds=a.budget)
        config['budget'] = {'session_start_utc': now.isoformat(), 'deadline_utc': deadline.isoformat(), 'budget_seconds': a.budget, 'task_source_commit': '63bdd80216edf825d887bc5316eb0ad78e7250ad', 'authorization': 'P48 explicit new run budget'}
        write(root / 'config.json', config)
        write(root / 'run.json', {'name': a.name, 'seed': a.seed, 'budget_seconds': a.budget, 'deadline_utc': deadline.isoformat(), 'source_commit': revision(), 'initialization': 'fresh random network; no historical models or samples'})
        core('init', root / 'state', '--seed', a.seed, '--run-id', a.name, '--config', root / 'config.json')
        organize(root)
        write(root / 'control.json', {'state': 'new'})
        print('NEW ' + str(root)); return
    if not (root / 'run.json').is_file():
        raise RuntimeError('MISSING_RUN ' + str(root))
    if a.command == '_worker':
        worker(root, a.seconds, a.resume_worker); return
    if a.command in ('start', 'resume'):
        if not 0 < a.seconds <= 31536000:
            p.error('--seconds must be positive')
        if running(root):
            raise RuntimeError('ALREADY_RUNNING')
        cmd = [sys.executable, str(Path(__file__).resolve()), '_worker', a.name, '--seconds', str(a.seconds)]
        if a.command == 'resume':
            cmd.append('--resume-worker')
        with (root / 'logs/console.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env())
        # Handshake confirms the detached worker actually acquired its lock.
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError('START_FAILED: see ' + str(root / 'logs/console.log'))
            if running(root):
                print('STARTED pid=' + str(child.pid) + ' log=' + str(root / 'logs/console.log')); return
            time.sleep(.05)
        raise RuntimeError('START_HANDSHAKE_TIMEOUT: inspect logs before retrying')
    if a.command == 'stop':
        (root / 'state/STOP').touch()
        print('STOP_REQUESTED: saving at search/SGD boundary; use status to confirm stopped'); return
    if a.command == 'status':
        status(root); return
    if a.command == 'diagnose':
        result = status(root)
        target = HOME / (a.name + '-diagnostics.zip')
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('summary.json', json.dumps(result, ensure_ascii=False, indent=2))
            for path in [HOME / 'environment.json', root / 'run.json', root / 'control.json', root / 'config.json', root / 'state/status.json', root / 'state/models.json']:
                if path.exists():
                    z.write(path, path.name)
            log = root / 'logs/console.log'
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0, log.stat().st_size-2000000))
                    z.writestr('console-tail.log', f.read())
        print('DIAGNOSTICS ' + str(target))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e), file=sys.stderr)
        sys.exit(1)
