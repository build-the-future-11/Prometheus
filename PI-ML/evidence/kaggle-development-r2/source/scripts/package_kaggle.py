"""Make a private self-contained Kaggle script without credentials or data uploads."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['development','confirmatory'])
    p.add_argument('slug')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    selected=[]
    for folder in ['src','tests','configs','scripts','research']:
        selected.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.md'})
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for file in sorted(selected):
            archive.add(file,arcname=str(file.relative_to(root)),recursive=False)
    payload=buf.getvalue()
    digest=hashlib.sha256(payload).hexdigest()
    target=root/'delivery'/args.slug
    target.mkdir(parents=True,exist_ok=False)
    script='''import os, sys, json, base64, hashlib, tarfile, io, subprocess, traceback, time
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==DIGEST
root=Path('/kaggle/working/piml-source')
root.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
    for member in archive.getmembers():
        if member.issym() or member.islnk() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
            raise RuntimeError('unsafe archive member')
    archive.extractall(root)
os.chdir(root)
os.environ['PYTHONPATH']=str(root/'src')
out=Path('/kaggle/working/results')
out.mkdir(exist_ok=False)
def command(arguments,timeout):
    print('RUN',arguments,flush=True)
    proc=subprocess.run([sys.executable,*arguments],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    label=str(len(list(out.glob('command-*.log'))))
    (out/('command-'+label+'.log')).write_text(proc.stdout)
    print(proc.stdout[-5000:],flush=True)
    if proc.returncode:
        raise RuntimeError('command failed: '+str(arguments))
started=time.time()
try:
    command(['-m','unittest','discover','-s','tests','-v'],180)
    if PHASE=='development':
        command(['scripts/development_tolerance_sweep.py'],300)
        import shutil
        shutil.copy('runs/development/S1-tolerance-audit.json',out/'s1-tolerance-audit.json')
        command(['-m','piml.s2','run','configs/s2_development.json',str(out/'s2-development')],1500)
    else:
        command(['-m','piml','run','configs/confirmatory_v1.json',str(out/'s1-confirmatory')],300)
        command(['-m','piml.s2','run','configs/s2_confirmatory.json',str(out/'s2-confirmatory'),'--freeze','research/S2_FREEZE.json'],2100)
    status='COMPLETE'
except BaseException:
    status='FAILED'
    (out/'hosted_failure.txt').write_text(traceback.format_exc())
finally:
    (out/'hosted_receipt.json').write_text(json.dumps(dict(status=status,phase=PHASE,source_archive_sha256=DIGEST,wall_seconds=time.time()-started,python=sys.version),indent=2))
    with tarfile.open('/kaggle/working/piml-evidence.tar.gz','w:gz') as archive:
        archive.add(out,arcname='results')
        archive.add(root,arcname='source')
    print('HOSTED_STATUS',status,flush=True)
if status!='COMPLETE':
    raise RuntimeError('retained failed hosted attempt')
'''
    script=script.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('DIGEST',repr(digest)).replace('PHASE',repr(args.phase))
    (target/'piml_job.py').write_text(script)
    metadata=dict(id='ryangomez010/'+args.slug,title=args.slug,code_file='piml_job.py',language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[])
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (target/'source.tar.gz').write_bytes(payload)
    (target/'package_receipt.json').write_text(json.dumps(dict(phase=args.phase,source_sha256=digest,source_files=len(selected),kernel=metadata['id']),indent=2)+'\n')
    print(target)


if __name__=='__main__':
    main()
