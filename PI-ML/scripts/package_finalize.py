"""Package a hosted clean-environment artifact audit and paper build; no new study."""
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
target=root/'delivery/piml-s2-release-audit-20261021-r2'
target.mkdir(parents=True,exist_ok=False)
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for folder in ['src','tests','configs','research','paper','scripts']:
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in {'.py','.json','.md','.tex','.sty'}:
                archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
    for name in ['README.md','LICENSE','pyproject.toml','MANIFEST.in','requirements-hosted.txt']:
        archive.add(root/name,arcname=name)
payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
inputs={}
for phase,receipt in [('development','development_download_verification.json'),('confirmatory','confirmatory_download_verification.json')]:
    inputs[phase]=json.loads((root/'evidence/audit'/receipt).read_text())['archive_sha256']
script=r'''import os,sys,json,base64,hashlib,tarfile,io,subprocess,time,traceback,urllib.request,shutil
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==SOURCE_HASH
root=Path('/kaggle/working/release-source');root.mkdir(exist_ok=False)
def extract(archive,destination):
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        for m in t.getmembers():
            if m.issym() or m.islnk() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('unsafe archive')
        t.extractall(destination)
extract(payload,root)
os.chdir(root)
out=Path('/kaggle/working/release-audit');out.mkdir(exist_ok=False)
paths={}
expected=EXPECTED_INPUTS
for p in Path('/kaggle/input').rglob('piml-evidence.tar.gz'):
    data=p.read_bytes();sha=hashlib.sha256(data).hexdigest()
    for phase,wanted in expected.items():
        if sha==wanted:
            dest=Path('/kaggle/working')/('verified-'+phase);dest.mkdir(exist_ok=False)
            extract(data,dest);paths[phase]=dest/'results'
if set(paths)!=set(expected):raise RuntimeError('missing or nonmatching retained input archives')
def command(args,timeout=600,env=None):
    print('RUN',args,flush=True)
    completed=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,env=env)
    (out/('command-'+str(len(list(out.glob('command-*.log'))))+'.log')).write_text(completed.stdout)
    print(completed.stdout[-3000:],flush=True)
    if completed.returncode:raise RuntimeError('command failed: '+str(args))
started=time.time();status='FAILED'
try:
    venv=Path('/kaggle/working/clean-venv')
    command([sys.executable,'-m','venv','--without-pip',str(venv)])
    py=str(venv/'bin/python')
    bootstrap=urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py',timeout=90).read()
    bootstrap_path=Path('/kaggle/working/get-pip.py');bootstrap_path.write_bytes(bootstrap)
    (out/'pip_bootstrap_sha256.txt').write_text(hashlib.sha256(bootstrap).hexdigest()+'\n')
    command([py,str(bootstrap_path),'pip==25.0.1'])
    command([py,'-m','pip','install','numpy==2.1.3','scipy==1.15.3','matplotlib==3.10.1','setuptools==77.0.3','wheel==0.45.1','build==1.2.2.post1'])
    clean=os.environ.copy();clean.pop('PYTHONPATH',None)
    # First build/install the library, then tests must import its wheel.
    command([py,'-m','build','--no-isolation','--wheel','--outdir',str(out/'packages'),'.'],env=clean)
    wheel=next((out/'packages').glob('*.whl'))
    command([py,'-m','pip','install','--no-deps',str(wheel)],env=clean)
    command([py,'-c','import piml; print(piml.__file__)'],env=clean)
    command([py,'-m','unittest','discover','-s','tests','-v'],env=clean)
    command([py,'-m','piml.s2','verify',str(paths['confirmatory']/'s2-confirmatory')],env=clean)
    command([py,'scripts/independent_audit.py',str(paths['confirmatory']/'s2-confirmatory'),str(out/'independent_audit.json')],env=clean)
    command([py,'scripts/build_paper.py',str(paths['development']),str(paths['confirmatory']),str(root/'paper')],env=clean)
    # Official Tectonic release asset, pinned by the vendor release SHA-256.
    url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
    data=urllib.request.urlopen(url,timeout=90).read()
    if hashlib.sha256(data).hexdigest()!='8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7':raise RuntimeError('Tectonic release digest mismatch')
    binary=Path('/kaggle/working/tectonic-bin');binary.mkdir()
    extract(data,binary)
    tectonic=next(binary.rglob('tectonic'));tectonic.chmod(0o755)
    command([str(tectonic),'--keep-logs',str(root/'paper/main.tex')],timeout=300,env=clean)
    # Build both distributions after generated figures/tables are present.
    command([py,'-m','build','--no-isolation','--outdir',str(out/'packages'),'.'],env=clean)
    command([py,'-m','pip','freeze'],env=clean)
    shutil.copytree(root/'paper',out/'paper')
    status='COMPLETE'
except BaseException:
    (out/'failure.txt').write_text(traceback.format_exc())
    print(traceback.format_exc(),flush=True)
    if (root/'paper').exists() and not (out/'paper').exists():shutil.copytree(root/'paper',out/'paper')
finally:
    (out/'receipt.json').write_text(json.dumps(dict(status=status,scope='Clean wheel environment, full numerical replay, separate SciPy/flux/exact-bootstrap audit and manuscript build; no new scientific run',source_sha256=SOURCE_HASH,input_hashes=EXPECTED_INPUTS,wall_seconds=time.time()-started),indent=2))
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2))
    with tarfile.open('/kaggle/working/piml-release-audit.tar.gz','w:gz') as t:t.add(out,arcname='release-audit')
    print('RELEASE_AUDIT_STATUS',status,flush=True)
if status!='COMPLETE':raise RuntimeError('release audit failed, evidence retained')
'''
script=script.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('SOURCE_HASH',repr(digest)).replace('EXPECTED_INPUTS',repr(inputs))
(target/'release_job.py').write_text(script)
(target/'kernel-metadata.json').write_text(json.dumps(dict(id='ryangomez010/piml-s2-release-audit-20261021-r2',title='piml-s2-release-audit-20261021-r2',code_file='release_job.py',language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=['ryangomez010/piml-s2-development-20261021-r2','ryangomez010/piml-s2-confirmatory-20261021-r1']),indent=2)+'\n')
(target/'package_receipt.json').write_text(json.dumps(dict(source_sha256=digest,input_hashes=inputs),indent=2)+'\n')
print(target)
