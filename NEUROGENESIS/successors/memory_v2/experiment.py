"""Immutable execution, selection, confirmation, statistics and verification CLI."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time
import traceback
from dataclasses import asdict, replace

from engine import Config, Runner, canonical, digest, metrics, synthetic

ROOT = Path(__file__).resolve().parent
CONDITIONS = ["base", "novel", "delay", "noise", "distractor", "dimension", "horizon"]
PRIMARY = ["base", "delay", "noise", "distractor", "dimension"]
DEV_SEEDS = [1201, 1213, 1223, 1237, 1249]
TEST_SEEDS = list(range(80001, 80021))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def read(path):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=constant)


def source_map():
    return {p.name:sha(p) for p in sorted(ROOT.glob("*.py"))} | {"SPECIFICATION.md":sha(ROOT/"SPECIFICATION.md"),"protocol.json":sha(ROOT/"protocol.json"),"data/manifest.json":sha(ROOT/"data/manifest.json")}


def protocol():
    p=read(ROOT/"protocol.json")
    if p["development_seeds"] != DEV_SEEDS or p["confirmation_seeds"] != TEST_SEEDS or p["conditions"] != CONDITIONS or p["primary_conditions"] != PRIMARY:
        raise ValueError("registered protocol differs from executable matrix")
    if set(DEV_SEEDS)&set(TEST_SEEDS):
        raise ValueError("split overlap")
    return p


def grid():
    configs={}
    for eta in (0.5,1.0):
        for alpha in (0.05,0.2):
            for k in (4,8):
                configs[f"bank_e{eta}_a{alpha}_k{k}"]=Config(eta=eta,alpha=alpha,capacity=k)
    for eta in (0.1,0.5,1.0):
        configs[f"nlms_{eta}"]=Config(kind="nlms",eta=eta)
    for eta in (0.001,0.01,0.05):
        configs[f"adam_{eta}"]=Config(kind="adam",eta=eta)
    for f in (0.95,0.99,1.0):
        configs[f"rls_{f}"]=Config(kind="rls",forgetting=f)
    for kind in ("frozen","last","knn","replay"):
        configs[kind]=Config(kind=kind)
    return configs


def rss_bytes():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform=="darwin" else rss*1024)


def environment():
    return {"python":sys.version,"platform":platform.platform(),"host":"kaggle" if Path('/kaggle/working').exists() else "local","process_peak_rss_bytes":rss_bytes(),"utc":now()}


def host_probe():
    import subprocess
    if sys.platform != "darwin":
        return {"platform":sys.platform,"rss":rss_bytes(),"pageouts":None}
    raw=subprocess.check_output(["/usr/bin/vm_stat"],text=True)
    counters={}
    for line in raw.splitlines()[1:]:
        if ':' in line:
            key,value=line.split(':',1)
            counters[key]=int(value.strip().rstrip('.'))
    page=int(raw.split('page size of ')[1].split(' ')[0])
    return {"platform":sys.platform,"rss":rss_bytes(),"pageouts":counters['Pageouts'],"reclaimable_bytes":page*sum(counters.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative']),"raw":raw}


def admission(output):
    a=host_probe()
    time.sleep(1)
    b=host_probe()
    allowed=b['rss']<512*1024**2
    if sys.platform=="darwin":
        allowed=allowed and b['reclaimable_bytes']>=512*1024**2 and a['pageouts']==b['pageouts']
    write(output/'admission.json',{'before':a,'after':b,'allowed':allowed,'utc':now()})
    if not allowed:
        raise RuntimeError("RESOURCE_ADMISSION_REFUSED: retain receipt; use a fresh output after resources recover")


def execute(output,task_id,rows,name,config):
    target=output/'attempts'/f'{task_id}__{name}'
    target.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    runner=Runner(rows,config)
    write(target/'attempt.json',{'utc':now(),'task_id':task_id,'method':name,'config':asdict(config),'task_sha256':digest(rows)})
    status='COMPLETED'
    failure=None
    try:
        runner.advance()
        if rss_bytes()>512*1024**2:
            raise MemoryError('process RSS cap exceeded')
        result=metrics(runner.trace)
    except Exception:
        status='FAILED'
        failure=traceback.format_exc()
        result={}
    # Gzip with fixed metadata preserves deterministic trace bytes across environments.
    with (target/'trace.jsonl.gz').open('xb') as file:
        with gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0) as zipped:
            for row in runner.trace:
                zipped.write((canonical(row)+'\n').encode())
    with (target/'checkpoint.json.gz').open('xb') as file:
        with gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0) as zipped:
            zipped.write(canonical(runner.snapshot()).encode())
    result.update({'status':status,'failure':failure,'task_id':task_id,'method':name,'config':asdict(config),'task_sha256':digest(rows),'trace_sha256':sha(target/'trace.jsonl.gz'),'checkpoint_sha256':sha(target/'checkpoint.json.gz'),'wall_seconds':time.perf_counter()-start,'process_peak_rss_bytes':rss_bytes()})
    write(target/'result.json',result)
    return result


def run_matrix(output,phase,configs,seeds,conditions,lock=None):
    output.mkdir(parents=True,exist_ok=False)
    admission(output)
    p=protocol()
    manifest={'schema':'neurogenesis.memory.matrix.v2','phase':phase,'utc':now(),'environment':environment(),'source':source_map(),'protocol':p,'configs':{n:asdict(c) for n,c in configs.items()},'seeds':seeds,'conditions':conditions,'expected_attempts':len(configs)*len(seeds)*len(conditions),'lock_sha256':digest(lock) if lock else None}
    write(output/'manifest.json',manifest)
    start=time.perf_counter()
    results=[]
    for condition in conditions:
        for seed in seeds:
            rows=synthetic(seed,condition)
            for name,config in configs.items():
                if time.perf_counter()-start>p['phase_cpu_wall_seconds']:
                    write(output/'interruption.json',{'reason':'phase wall budget exceeded','completed':len(results),'expected':manifest['expected_attempts']})
                    raise TimeoutError('phase budget exceeded; incomplete matrix retained')
                results.append(execute(output,f'{condition}-{seed}',rows,name,config))
            print(f'{phase}: {condition} seed {seed}, {len(results)}/{manifest["expected_attempts"]}',flush=True)
    write(output/'receipt.json',{'completed':sum(r['status']=='COMPLETED' for r in results),'failed':sum(r['status']=='FAILED' for r in results),'expected':manifest['expected_attempts'],'wall_seconds':time.perf_counter()-start,'post_resources':host_probe(),'utc':now()})
    summarize(output)
    return results


def load_results(output):
    return [read(p) for p in sorted((Path(output)/'attempts').glob('*/result.json'))]


def summarize(output):
    rows=load_results(output)
    names=sorted(set(r['method'] for r in rows))
    summary={}
    for name in names:
        selected=[r for r in rows if r['method']==name]
        completed=[r for r in selected if r['status']=='COMPLETED']
        summary[name]={'attempted':len(selected),'completed':len(completed),'failed':len(selected)-len(completed)}
        if completed:
            summary[name].update({'mse':statistics.mean(r['mse'] for r in completed),'updates':statistics.mean(r['usage']['updates'] for r in completed),'coordinate_ops':statistics.mean(r['usage']['coordinate_ops'] for r in completed),'state_bytes':max(r['peak_state_bytes'] for r in completed),'wall_seconds':statistics.mean(r['wall_seconds'] for r in completed)})
    write(Path(output)/'summary.json',summary)
    return summary


def verify(output,replay=False):
    output=Path(output)
    m=read(output/'manifest.json')
    if m['source']!=source_map():
        raise ValueError('source hash mismatch')
    if m['protocol']!=protocol():
        raise ValueError('protocol mismatch')
    results=load_results(output)
    expected={(f'{c}-{s}',n) for c in m['conditions'] for s in m['seeds'] for n in m['configs']}
    found={(r['task_id'],r['method']) for r in results}
    if expected!=found or len(results)!=len(expected):
        raise ValueError('incomplete/duplicate matrix')
    trace_rows=0
    for result in results:
        path=output/'attempts'/f'{result["task_id"]}__{result["method"]}'
        if sha(path/'trace.jsonl.gz')!=result['trace_sha256'] or sha(path/'checkpoint.json.gz')!=result['checkpoint_sha256']:
            raise ValueError('artifact hash mismatch')
        if result['config']!=m['configs'][result['method']]:
            raise ValueError('config mismatch')
        trace=[json.loads(line) for line in gzip.open(path/'trace.jsonl.gz','rt')]
        if result['status']=='COMPLETED':
            measured=metrics(trace)
            if any(measured[k]!=result[k] for k in measured):
                raise ValueError('metric/counter mismatch')
            condition,seed=result['task_id'].rsplit('-',1)
            data=synthetic(int(seed),condition)
            if digest(data)!=result['task_sha256'] or len(trace)!=len(data):
                raise ValueError('task identity mismatch')
            delivered=[]
            for t,row in enumerate(trace):
                if row['step']!=t or row['target']!=data[t]['y'] or row['squared_error']!=(row['prediction']-data[t]['y'])**2:
                    raise ValueError('trace scoring mismatch')
                for s in row['labels_before']:
                    if s>=t or data[s]['available']>t:
                        raise ValueError('future label exposure')
                for s in row['labels_after']:
                    if s>t or data[s]['available']>t:
                        raise ValueError('post-prediction timing violation')
                delivered+=row['labels_before']+row['labels_after']
            intended=[row['step'] for row in data if row['available']<len(data)]
            if sorted(delivered)!=intended or len(set(delivered))!=len(delivered):
                raise ValueError('label denominator mismatch')
            if replay:
                runner=Runner(data,Config(**result['config']))
                runner.advance()
                if runner.trace!=trace:
                    raise ValueError('exact replay mismatch')
                checkpoint=json.loads(gzip.open(path/'checkpoint.json.gz','rt').read())
                if runner.snapshot()!=checkpoint:
                    raise ValueError('checkpoint replay mismatch')
        trace_rows+=len(trace)
    return {'status':'VALID','attempts':len(results),'trace_rows':trace_rows,'exact_replay':replay,'failed':sum(r['status']=='FAILED' for r in results),'artifact_map_sha256':digest({str(p.relative_to(output)):sha(p) for p in sorted((output/'attempts').rglob('*')) if p.is_file()}),'utc':now()}


def freeze(development,output):
    verification=verify(development)
    if verification['failed']:
        raise ValueError('failed development cells prevent selection')
    m=read(Path(development)/'manifest.json')
    if m['phase']!='development' or m['seeds']!=DEV_SEEDS or m['conditions']!=['base','novel'] or m['configs']!={n:asdict(c) for n,c in grid().items()}:
        raise ValueError('development selection denominator mismatch')
    # Selection is recomputed from verified attempt artifacts, never a mutable summary.
    raw_results=load_results(development)
    s={name:{'mse':statistics.mean(r['mse'] for r in raw_results if r['method']==name)} for name in m['configs']}
    candidate=min((n for n in s if n.startswith('bank_')),key=lambda n:(s[n]['mse'],n))
    baseline=min((n for n in s if not n.startswith('bank_')),key=lambda n:(s[n]['mse'],n))
    c=Config(**m['configs'][candidate])
    configs={'candidate':c,'baseline':Config(**m['configs'][baseline]),'all_update':replace(c,suppress=False),'capacity_one':replace(c,capacity=1),'uniform_bank':replace(c,uniform=True),'no_bank':replace(c,kind='nlms'),'replay':Config(kind='replay',eta=c.eta),'knn':Config(kind='knn'),'frozen':Config(kind='frozen')}
    lock={'schema':'neurogenesis.memory.freeze.v2','utc':now(),'source':source_map(),'candidate_name':candidate,'baseline_name':baseline,'development_artifact_map_sha256':verification['artifact_map_sha256'],'development_manifest_sha256':sha(Path(development)/'manifest.json'),'configs':{n:asdict(v) for n,v in configs.items()},'protocol':protocol(),'protected_outcomes_seen':False}
    write(output,lock)
    return lock


def bootstrap(values,draws=10000):
    r=random.Random(20261021)
    means=sorted(statistics.mean(r.choices(values,k=len(values))) for _ in range(draws))
    return [means[int(.025*draws)],means[int(.975*draws)-1]]


def analyze(output,lock):
    output=Path(output)
    verification=verify(output)
    if verification['failed']:
        return {'disposition':'INCONCLUSIVE','reason':'failed cells','verification':verification}
    manifest=read(output/'manifest.json')
    if manifest['phase']!='confirmation' or manifest['lock_sha256']!=digest(lock):
        raise ValueError('confirmation lock mismatch')
    rows=load_results(output)
    lookup={(r['task_id'],r['method']):r for r in rows}
    comparisons={}
    for comparator in ('baseline','all_update','no_bank','replay','knn','frozen'):
        effects=[]
        reductions=[]
        for seed in TEST_SEEDS:
            candidate=statistics.mean(lookup[(f'{c}-{seed}','candidate')]['mse'] for c in PRIMARY)
            control=statistics.mean(lookup[(f'{c}-{seed}',comparator)]['mse'] for c in PRIMARY)
            effects.append(candidate/max(control,1e-30)-1)
            u=sum(lookup[(f'{c}-{seed}','candidate')]['usage']['updates'] for c in PRIMARY)
            v=sum(lookup[(f'{c}-{seed}',comparator)]['usage']['updates'] for c in PRIMARY)
            reductions.append(1-u/v if v else None)
        comparisons[comparator]={'relative_mse_change':statistics.mean(effects),'ci95':bootstrap(effects),'per_seed_effects':effects,'mean_update_reduction':statistics.mean(x for x in reductions if x is not None) if any(x is not None for x in reductions) else None}
    primary=comparisons['baseline']
    reduction=primary['mean_update_reduction']
    support=primary['ci95'][1]<=.02 and reduction is not None and reduction>=.2
    if support:
        disposition='SUPPORTED'
    elif primary['ci95'][0]>.02 or (reduction is not None and reduction<.2):
        disposition='NEGATIVE'
    else:
        disposition='INCONCLUSIVE'
    per_condition={c:{n:statistics.mean(r['mse'] for r in rows if r['task_id'].startswith(c+'-') and r['method']==n) for n in manifest['configs']} for c in CONDITIONS}
    return {'schema':'neurogenesis.memory.analysis.v2','disposition':disposition,'scope':'synthetic recurring conditions, paired over 20 fresh seeds','primary':primary,'comparisons':comparisons,'per_condition_mse':per_condition,'verification':verification,'accuracy_superiority':primary['ci95'][1]<-.05,'all_update_noninferiority':comparisons['all_update']['ci95'][1]<=.02,'independent_external_replication':False,'real_data_included':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['develop','freeze','confirm','verify','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--lock',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.command=='develop':
        run_matrix(args.output,'development',grid(),DEV_SEEDS,['base','novel'])
    elif args.command=='freeze':
        print(canonical(freeze(args.input,args.output)))
    elif args.command=='confirm':
        lock=read(args.lock)
        if lock['source']!=source_map() or lock['protocol']!=protocol() or lock['protected_outcomes_seen']:
            raise ValueError('freeze invalidated')
        run_matrix(args.output,'confirmation',{n:Config(**v) for n,v in lock['configs'].items()},TEST_SEEDS,CONDITIONS,lock)
    elif args.command=='verify':
        receipt=verify(args.input,args.replay)
        write(args.output,receipt)
        print(canonical(receipt))
    else:
        result=analyze(args.input,read(args.lock))
        write(args.output,result)
        print(canonical(result))


if __name__=='__main__':
    main()
