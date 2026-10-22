"""Read-only release verification spanning preserved and successor studies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def verify(output):
    baseline=json.loads((ROOT/'successors/memory_v2/predecessor_hashes.json').read_text())
    changed=[path for path,expected in baseline.items() if not (ROOT/path).is_file() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected]
    if changed:raise ValueError('historical evidence changed: '+repr(changed))
    root3=ROOT/'successors/memory_v3'
    commands=[
        [sys.executable,'-m','unittest','discover','-s','tests','-v'],
        [sys.executable,'-m','unittest','discover','-s','successors/memory_v3','-p','test_*.py','-v'],
        [sys.executable,'scripts/verify_bundle.py','--root','.', '--bundle','results/development/shared-baselines-v1-20261019','--output',str(output/'v1.json')],
        [sys.executable,'successors/memory_v2/experiment.py','verify','--input','successors/memory_v2/runs/development-02','--output',str(output/'v2.json')],
        [sys.executable,'successors/memory_v3/experiment.py','verify','--input','successors/memory_v3/runs/development-01','--output',str(output/'v3-development.json')],
        [sys.executable,'successors/memory_v3/experiment.py','verify','--input','successors/memory_v3/runs/confirmation-02','--output',str(output/'v3-confirmation.json')],
        [sys.executable,'successors/memory_v3/publication/audit_arithmetic.py','successors/memory_v3/runs/confirmation-02',str(output/'v3-separate-audit.json')],
        [sys.executable,'successors/memory_v3/publication/verify_real_outcome.py','--input','successors/memory_v3/runs/real-transfer-03','--output',str(output/'v3-real-outcome.json')],
        [sys.executable,'successors/memory_v3/publication/audit_arithmetic.py','successors/memory_v3/runs/real-transfer-03',str(output/'v3-real-audit.json')],
    ]
    output.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for i,command in enumerate(commands):
        process=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/f'command-{i:02d}.log').write_text(process.stdout)
        receipts.append({'command':command,'exit_code':process.returncode})
        if process.returncode:break
    result={'status':'VALID' if len(receipts)==len(commands) and all(r['exit_code']==0 for r in receipts) else 'INVALID','scope':'artifact verification; not a successful real-transfer comparison','real_transfer_scientific_status':'INVALID_NUMERICAL_FAILURE','historical_files_unchanged':len(baseline),'commands':receipts,'external_replication':False}
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'historical_files_unchanged':len(baseline),'checks':len(receipts)}))
    return 0 if result['status']=='VALID' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(verify(a.output.resolve()))
