"""No training: verify publication identities, parse JSON and compile Python."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'PUBLIC_MANIFEST.json').read_text())
    checked = 0
    for row in manifest['files']:
        path = ROOT / row['path']
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == row['git_blob_sha'], row['path']
        if path.suffix == '.py':
            ast.parse(raw.decode(), filename=row['path'])
        elif path.suffix == '.json':
            json.loads(raw)
        checked += 1
    print(json.dumps({'verified_files': checked, 'training_started': False}))


if __name__ == '__main__':
    main()
