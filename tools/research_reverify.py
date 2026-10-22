#!/usr/bin/env python3
"""Reverify the frozen archive on its numerical platform; preserve all failures.

The Linux attempt remains retained. This does not relax a hash, modify a model,
select a seed, or convert post-outcome computational reproduction into a new test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / 'NEUROGENESIS/successors/memory_v4'


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(p: Path, value: object) -> None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def run(out: Path, name: str, args: list[str], timeout: int=600) -> dict:
    start=time.monotonic()
    log=out/(name+'.log')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    with log.open('x',encoding='utf-8') as f:
        try:
            rc=subprocess.run(args,cwd=V4,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
            status='PASS' if rc==0 else 'FAIL'
        except subprocess.TimeoutExpired:
            rc,status=None,'TIMEOUT'
    receipt={'command':args,'status':status,'returncode':rc,'seconds':time.monotonic()-start,'log_sha256':sha(log)}
    write(out/(name+'.json'),receipt)
    print(json.dumps({'step':name,**receipt}),flush=True)
    if status!='PASS':
        print(log.read_text()[-10000:],flush=True)
        raise RuntimeError('Retained failing step: '+name)
    return receipt


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    status={'portfolio_complete':False,'new_blinded_confirmation':False,'kaggle_execution':False,'external_submission':False,'independent_investigator_replication':False}
    try:
        if platform.system()!='Darwin' or platform.machine()!='arm64' or sys.version_info[:3]!=(3,15,0):
            raise RuntimeError('Use the declared macOS arm64 CPython 3.15.0 verification environment')
        freeze=V4/'runs/freeze.json'
        lock=json.loads(freeze.read_text())
        for name,digest in lock['source'].items():
            if sha(V4/name)!=digest:
                raise RuntimeError('Frozen source mismatch: '+name)
        write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'reason':'Linux input regeneration differs at floating-point roundoff; no verifier tolerance changed.'})
        run(out,'tests',[sys.executable,'-m','unittest','discover','-s','.','-p','test_*.py','-v'])
        confirmation=V4/'runs/confirmation-02'
        run(out,'synthetic-exact-reverification',[sys.executable,'experiment.py','verify','--input',str(confirmation),'--output',str(out/'synthetic-verification-exact.json'),'--replay'],900)
        run(out,'synthetic-arithmetic-audit',[sys.executable,'audit.py',str(confirmation),str(out/'synthetic-audit.json')])
        run(out,'synthetic-reanalysis',[sys.executable,'experiment.py','analyze','--input',str(confirmation),'--lock',str(freeze),'--output',str(out/'analysis.json')])
        # The historical manuscript already contains both stream outcomes. These
        # new directories are post-outcome reproductions, never new blind tests.
        write(out/'REPLICATION_CONTEXT.json',{'classification':'POST_OUTCOME_COMPUTATIONAL_REPRODUCTION','new_blinded_confirmation':False,'independent_investigator_replication':False,'kaggle_execution':False,'model_or_protocol_retuned':False,'prior_linux_evidence':'research-releases/2026-10-22-attempt-01','historical_phase_labels':'Inherited protected_transfer labels refer to the original protocol, not the status of this repetition.'})
        for dataset in ('air','beijing'):
            target=out/dataset
            run(out,dataset+'-run',[sys.executable,'transfer.py','run','--dataset',dataset,'--lock',str(freeze),'--output',str(target)],600)
            run(out,dataset+'-exact-replay',[sys.executable,'transfer.py','verify','--input',str(target),'--lock',str(freeze),'--output',str(target/'verification-exact.json'),'--replay'],600)
            run(out,dataset+'-arithmetic-audit',[sys.executable,'audit.py',str(target),str(target/'audit.json')])
        paper=out/'paper'
        run(out,'paper-generation',[sys.executable,'publication/generate.py','--lock',str(freeze),'--analysis',str(out/'analysis.json'),'--confirmation',str(confirmation),'--air',str(out/'air'),'--beijing',str(out/'beijing'),'--output',str(paper)])
        tex=(V4/'paper/main.tex').read_text()
        tex=tex.replace(r'\usepackage[margin=1in]{geometry}',r'\usepackage{arxiv}')
        tex=tex.replace(r'\date{October 21, 2026}',r'\date{October 22, 2026}')
        tex=tex.replace('K. Goel,','A. Goel,').replace('K. Lai and','B. Lai and')
        tex=tex.replace(r'\begin{document}',r'\renewcommand{\shorttitle}{Persistent-ridge predictor memory}'+'\n'+r'\begin{document}')
        note=(r'\paragraph{Reproduction and portability notice.} This revision reanalyzes retained evidence and reproduces already disclosed stream outcomes; it is not a new blinded confirmation or independent-investigator replication. A Linux audit passed all six project suites and exactly replayed both real streams, but failed the historical synthetic task-identity check. A diagnostic using the recorded Python version found target differences below $10^{-15}$ between Linux and the original macOS environment. We did not relax equality, change a seed, or edit the frozen model: the archive was reverified and replayed on macOS arm64 with CPython 3.15.0. Cross-platform bitwise regeneration remains a limitation. No Kaggle run or external submission is claimed for this revision.'+'\n')
        tex=tex.replace(r'\end{abstract}',r'\end{abstract}'+'\n'+note,1)
        (paper/'main.tex').write_text(tex)
        shutil.copy2(ROOT/'PI-ML/paper/arxiv.sty',paper/'arxiv.sty')
        (paper/'REPRODUCTION.md').write_text('This manuscript was rebuilt from verified frozen evidence. Read ../REPLICATION_CONTEXT.json and ../completion.json. Run pdflatex -interaction=nonstopmode -halt-on-error main.tex twice from this directory. Prior Linux verification failure is retained in research-releases/2026-10-22-attempt-01. The original models, protocols, and negative results were not changed. This package is not a venue submission.\n')
        (out/'python-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        for name,digest in lock['source'].items():
            if sha(V4/name)!=digest:
                raise RuntimeError('Frozen source changed during verification: '+name)
        air=json.loads((out/'air/summary.json').read_text())
        beijing=json.loads((out/'beijing/summary.json').read_text())
        status.update(status='VERIFIED_SOURCE_READY_FOR_PDF_REVIEW',scientific_disposition='MIXED_SCOPE_LIMITED',synthetic_reverified=True,frozen_sources_unchanged=True,air_relative_mse_change=air['relative_mse_change'],air_update_reduction=air['update_reduction'],beijing_relative_mse_change=beijing['relative_mse_change'],beijing_update_reduction=beijing['update_reduction'],candidate_is_best_beijing_method=False,pdf_visual_review_complete=False)
    except Exception as exc:
        status.update(status='BLOCKED_RETAINED',error=str(exc),traceback=traceback.format_exc())
    write(out/'completion.json',status)
    # Include nested scientific manifests; exclude only this outer manifest.
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write(out/'artifact-manifest.json',files)
    print(json.dumps(status,indent=2),flush=True)
    return 0 if status['status']=='VERIFIED_SOURCE_READY_FOR_PDF_REVIEW' else 1


if __name__=='__main__':
    raise SystemExit(main())
