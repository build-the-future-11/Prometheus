"""Package final manuscript rendering only; consumes frozen results, runs no study."""
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / 'delivery/piml-paper-release-20261021-r1'
target.mkdir(parents=True, exist_ok=False)
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for p in sorted((root / 'paper').rglob('*')):
        if p.is_file() and (p.suffix in {'.tex', '.sty', '.json'} or (p.parent.name == 'figures' and p.suffix == '.pdf')):
            archive.add(p, arcname=str(p.relative_to(root / 'paper')), recursive=False)
payload = buffer.getvalue()
digest = hashlib.sha256(payload).hexdigest()
script = r'''import base64,hashlib,io,json,tarfile,subprocess,urllib.request
from pathlib import Path
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==DIGEST
root=Path('/kaggle/working/final-paper');root.mkdir()
def unpack(data,destination):
 with tarfile.open(fileobj=io.BytesIO(data)) as t:
  for m in t.getmembers():
   if m.issym() or m.islnk() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('unsafe archive')
  t.extractall(destination,filter='data')
unpack(payload,root)
url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
data=urllib.request.urlopen(url,timeout=90).read()
assert hashlib.sha256(data).hexdigest()=='8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'
binary=Path('/tmp/piml-tectonic');binary.mkdir();unpack(data,binary)
exe=next(binary.rglob('tectonic'));exe.chmod(0o755)
p=subprocess.run([str(exe),'--keep-logs',str(root/'main.tex')],capture_output=True,text=True,timeout=300)
(root/'compile.log').write_text(p.stdout+p.stderr)
print(p.stdout,p.stderr,flush=True)
if p.returncode:raise RuntimeError('paper compilation failed')
(root/'render_receipt.json').write_text(json.dumps(dict(status='COMPLETE',source_archive_sha256=DIGEST,scope='Manuscript-only rendering; no scientific execution'),indent=2))
(root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},indent=2))
with tarfile.open('/kaggle/working/piml-final-paper.tar.gz','w:gz') as t:t.add(root,arcname='final-paper')
print('FINAL_PAPER_COMPLETE',flush=True)
'''.replace('PAYLOAD', repr(base64.b64encode(payload).decode())).replace('DIGEST', repr(digest))
(target / 'render_job.py').write_text(script)
(target / 'kernel-metadata.json').write_text(json.dumps(dict(id='ryangomez010/piml-paper-release-20261021-r1', title='piml-paper-release-20261021-r1', code_file='render_job.py', language='python', kernel_type='script', is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), indent=2)+'\n')
(target / 'package_receipt.json').write_text(json.dumps(dict(source_archive_sha256=digest),indent=2)+'\n')
print(target)
