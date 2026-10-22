"""Restore the private full evidence bundle, including weights, without network."""
import base64
import hashlib
import json
import sys
import zipfile
import io
from pathlib import Path

here=Path(__file__).resolve().parent/'artifacts'
manifest=json.loads((here/'ARCHIVE.json').read_text())
pieces=[]
for part in manifest['parts']:
    raw=(here/part['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=part['sha256']:raise ValueError('PART_HASH '+part['path'])
    pieces.append(raw.decode().strip())
archive=base64.b64decode(''.join(pieces),validate=True)
if hashlib.sha256(archive).hexdigest()!=manifest['archive_sha256']:raise ValueError('ARCHIVE_HASH')
dest=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()/'p49-restored-evidence'
dest.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    for member in z.infolist():
        target=(dest/member.filename).resolve()
        if not target.is_relative_to(dest):raise ValueError('UNSAFE_ARCHIVE_PATH')
    z.extractall(dest)
identities=json.loads((dest/'ARTIFACTS.json').read_text())
for relative,expected in identities['files'].items():
    if hashlib.sha256((dest/relative).read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('FILE_HASH '+relative)
print(json.dumps({'restored':str(dest),'archive_sha256':manifest['archive_sha256'],'verified_files':len(identities['files'])}))
