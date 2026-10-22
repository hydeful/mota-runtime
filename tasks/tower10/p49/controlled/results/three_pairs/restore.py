"""Restore private binary-part evidence and verify every SHA, without network."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
here=Path(__file__).resolve().parent/'artifacts'
m=json.loads((here/'ARCHIVE.json').read_text());parts=[]
for p in m['parts']:
    raw=(here/p['path']).read_bytes()
    if len(raw)!=p['bytes'] or hashlib.sha256(raw).hexdigest()!=p['sha256']:raise ValueError('PART_HASH '+p['path'])
    parts.append(raw)
raw=b''.join(parts)
if len(raw)!=m['archive_bytes'] or hashlib.sha256(raw).hexdigest()!=m['archive_sha256']:raise ValueError('ARCHIVE_HASH')
dest=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()/'p49-controlled-restored'
dest.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for p in z.infolist():
        if not (dest/p.filename).resolve().is_relative_to(dest):raise ValueError('ARCHIVE_PATH')
    z.extractall(dest)
manifest=json.loads((dest/'ARTIFACTS.json').read_text())
for rel,expected in manifest['files'].items():
    data=(dest/rel).read_bytes()
    if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('FILE_HASH '+rel)
print(json.dumps({'restored':str(dest),'archive_sha256':m['archive_sha256'],'verified_files':len(manifest['files'])}))
