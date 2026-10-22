"""Build a static UI that executes the byte-identical Python engine in Pyodide."""
import hashlib
import json
import shutil
import io
import tarfile
import urllib.request
from pathlib import Path
from .environment import HERE, build, reference, ENGINE_SHA, SOURCE_COMMIT

PYODIDE_VERSION = '0.27.7'
PYODIDE_ARCHIVE_SHA256 = '89d5d952675eb08e11286e52c7238f1248ce71726e896db9703e73bef23dc343'
RUNTIME_FILES = ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json', 'package.json']


def ensure_runtime(dest):
    target = dest/'vendor/pyodide'; manifest = target/'SHA256.json'
    if manifest.exists():
        checks = json.loads(manifest.read_text())
        if all((target/n).is_file() and hashlib.sha256((target/n).read_bytes()).hexdigest()==h for n,h in checks.items()) and set(checks)==set(RUNTIME_FILES):
            return
    print('Fetching pinned Pyodide 0.27.7 runtime (first build only)…', flush=True)
    url = f'https://registry.npmjs.org/pyodide/-/pyodide-{PYODIDE_VERSION}.tgz'
    with urllib.request.urlopen(url, timeout=60) as response:
        blob = response.read()
    if hashlib.sha256(blob).hexdigest() != PYODIDE_ARCHIVE_SHA256:
        raise ValueError('PYODIDE_ARCHIVE_IDENTITY_MISMATCH')
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for name in RUNTIME_FILES:
            (target/name).write_bytes(archive.extractfile('package/'+name).read())
    manifest.write_text(json.dumps({n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in RUNTIME_FILES}, indent=2)+'\n')


def build_web(dest=None):
    dest = Path(dest) if dest else HERE/'web/dist'
    dest.mkdir(parents=True, exist_ok=True)
    for p in (HERE/'web/src').iterdir():
        if p.is_file():
            shutil.copyfile(p, dest/p.name)
    state = build()
    engine = Path(reference.__file__).read_bytes()
    assert hashlib.sha256(engine).hexdigest() == ENGINE_SHA
    (dest/'topology_engine.py').write_bytes(engine)
    (dest/'initial.json').write_text(json.dumps(state, ensure_ascii=False, separators=(',', ':'))+'\n')
    shutil.copyfile(HERE/'map.json', dest/'map.json')
    identity = {'map': 'p49-handdrawn-v1', 'source_commit': SOURCE_COMMIT,
                'engine_sha256': ENGINE_SHA, 'initial_sha256': reference.digest(state),
                'rules_sha256': reference.digest(state['rules']),
                'map_sha256': reference.digest(json.loads((HERE/'map.json').read_text())),
                'task_adapter': 'p49-boss-resolved-after-original-step-v1',
                'adapter_sha256': hashlib.sha256((HERE/'environment.py').read_bytes()).hexdigest(),
                'browser_adapter_sha256': hashlib.sha256((HERE/'web/src/bridge.py').read_bytes()).hexdigest(),
                'training_started': False}
    (dest/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (HERE/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (HERE/'initial_topology.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    ensure_runtime(dest)
    print(json.dumps({'directory': str(dest), 'identity': identity}), flush=True)
    return dest


if __name__ == '__main__':
    import sys
    build_web(sys.argv[1] if len(sys.argv) > 1 else None)
