"""Build a deterministic self-contained Kaggle notebook for frozen v4 evidence."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path(__file__).resolve().parent
ARCHIVE = DELIVERY / "source-v4.zip"
NOTEBOOK = DELIVERY / "neurogenesis-memory-v4-frozen-evidence.ipynb"

FILES = [
    "audit.py",
    "engine.py",
    "experiment.py",
    "real_transfer.py",
    "test_engine.py",
    "test_pipeline.py",
    "test_ridge.py",
    "transfer.py",
    "SPECIFICATION.md",
    "SUCCESSOR_CONTRACT.md",
    "FLAW_CHECKLIST.md",
    "RELATED_WORK.md",
    "protocol.json",
    "data/manifest.json",
    "data/AirQualityUCI.csv",
    "data/PRSA_data_2010.1.1-2014.12.31.csv",
    "data/beijing-complete.zip",
    "runs/freeze.json",
    "runs/preconfirmation_anchor.json",
]


def deterministic_zip() -> bytes:
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return ARCHIVE.read_bytes()


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


payload = deterministic_zip()
digest = hashlib.sha256(payload).hexdigest()
encoded = base64.b64encode(payload).decode("ascii")

setup = f"""import base64, hashlib, json, os, pathlib, shutil, subprocess, sys, traceback, zipfile
payload=base64.b64decode({encoded!r})
expected={digest!r}
assert hashlib.sha256(payload).hexdigest()==expected
root=pathlib.Path('/kaggle/working/neurogenesis_memory_v4')
if root.exists(): shutil.rmtree(root)
root.mkdir(parents=True)
archive=root/'source-v4.zip'
archive.write_bytes(payload)
with zipfile.ZipFile(archive) as z: z.extractall(root)
os.chdir(root)
print(json.dumps({{'status':'SOURCE_READY','source_archive_sha256':expected,'files':len(list(root.rglob('*')))}}))
"""

execute = """def run(*args):
    print('RUN', *args, flush=True)
    subprocess.run([sys.executable, *map(str,args)], check=True)

terminal={'status':'STARTED','source_archive_sha256':expected,'host':'kaggle' if pathlib.Path('/kaggle/working').exists() else 'other'}
try:
    run('-m','unittest','discover','-s','.','-p','test_*.py')
    run('experiment.py','confirm','--lock','runs/freeze.json','--output','runs/confirmation')
    run('experiment.py','verify','--input','runs/confirmation','--output','runs/confirmation/verification-exact.json','--replay')
    run('audit.py','runs/confirmation','runs/confirmation/audit.json')
    run('experiment.py','analyze','--input','runs/confirmation','--lock','runs/freeze.json','--output','runs/analysis.json')
    run('transfer.py','run','--dataset','air','--lock','runs/freeze.json','--output','runs/air-regression')
    run('transfer.py','verify','--input','runs/air-regression','--lock','runs/freeze.json','--output','runs/air-regression/verification-exact.json','--replay')
    run('audit.py','runs/air-regression','runs/air-regression/audit.json')
    run('transfer.py','run','--dataset','beijing','--lock','runs/freeze.json','--output','runs/beijing-transfer')
    run('transfer.py','verify','--input','runs/beijing-transfer','--lock','runs/freeze.json','--output','runs/beijing-transfer/verification-exact.json','--replay')
    run('audit.py','runs/beijing-transfer','runs/beijing-transfer/audit.json')
    terminal.update({
        'status':'EVIDENCE_RETAINED',
        'synthetic':json.loads((root/'runs/analysis.json').read_text()),
        'air':json.loads((root/'runs/air-regression/summary.json').read_text()),
        'beijing':json.loads((root/'runs/beijing-transfer/summary.json').read_text()),
        'independent_external_replication':False,
    })
except Exception:
    terminal.update({'status':'FAILED','failure':traceback.format_exc()})
    raise
finally:
    (root/'terminal.json').write_text(json.dumps(terminal,sort_keys=True,indent=2,allow_nan=False)+'\\n')
    output=pathlib.Path('/kaggle/working/neurogenesis-memory-v4-results.zip')
    if output.exists(): output.unlink()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(root))
    print(json.dumps({'status':terminal['status'],'result_archive':str(output),'bytes':output.stat().st_size},sort_keys=True),flush=True)
"""

notebook = {
    "cells": [
        cell(
            "markdown",
            "# NEUROGENESIS memory v4: frozen hosted execution\n"
            "Persistent-ridge delayed-credit memory and all controls are locked. This CPU notebook performs no tuning. "
            "It executes the full synthetic confirmation, exposed Air Quality regression, protected Beijing PM2.5 transfer, "
            "exact replay, and separate arithmetic audits. Download `neurogenesis-memory-v4-results.zip` after completion. "
            "A hosted replay is not independent external replication.\n\n"
            f"Embedded source archive SHA-256: `{digest}`.\n",
        ),
        cell("code", setup),
        cell("code", execute),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
(DELIVERY / "manifest.json").write_text(
    json.dumps(
        {
            "source_archive": ARCHIVE.name,
            "source_archive_sha256": digest,
            "source_archive_bytes": len(payload),
            "notebook": NOTEBOOK.name,
            "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            "files": FILES,
            "hosted_execution": False,
        },
        sort_keys=True,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "BUILT", "archive_sha256": digest, "notebook": str(NOTEBOOK)}))
