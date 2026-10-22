"""Verify the complete distributable project snapshot with streaming SHA-256."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())
failures=[]
for name,record in manifest['files'].items():
    path=root/name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        failures.append('unsafe-path:'+name);continue
    if not path.is_file():failures.append('missing:'+name);continue
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
    if sha.hexdigest()!=record['sha256'] or path.stat().st_size!=record['bytes']:
        failures.append('changed:'+name)
print(json.dumps(dict(status='PASS' if not failures else 'FAIL',files=len(manifest['files']),failures=failures),indent=2))
if failures:raise SystemExit(1)
