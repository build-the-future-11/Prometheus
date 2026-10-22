"""Separate result auditor, deliberately not importing model or metric code.

Checks raw squared-error arithmetic, feedback legality, state budgets, complete
matrix identities, and all immutable member hashes. It is an independent code
path, not independent external replication or an independent human review.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    rows=0;failures=[];seen=set();mapping={}
    if manifest['phase']=='real_transfer':
        expected={('uci-air-quality',name) for name in manifest['configs']}
    else:
        expected={(f'{condition}-{seed}',name) for condition in manifest['conditions'] for seed in manifest['seeds'] for name in manifest['configs']}
    for path in sorted((root/'attempts').glob('*/result.json')):
        result=json.loads(path.read_text());identity=(result['task_id'],result['method'])
        if identity in seen:raise ValueError('duplicate identity')
        seen.add(identity)
        for kind in ['trace','checkpoint']:
            artifact=path.parent/(kind+('.jsonl.gz' if kind=='trace' else '.json.gz'))
            actual=hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual!=result[kind+'_sha256']:raise ValueError('changed artifact')
            mapping[str(artifact.relative_to(root))]=actual
        if result['status']!='COMPLETED':
            failures.append(identity);continue
        total=0.;count=0;delivered=set();last_counts={}
        trace=list(map(json.loads,gzip.open(path.parent/'trace.jsonl.gz','rt')))
        for t,event in enumerate(trace):
            if event['step']!=t:raise ValueError('nonsequential trace')
            pred=event['prediction'];target=event['target']
            if not math.isfinite(pred):raise ValueError('nonfinite output')
            if target is not None:
                loss=(pred-target)**2
                if loss!=event['squared_error']:raise ValueError('incorrect squared error')
                total+=loss;count+=1
            elif event['squared_error'] is not None:raise ValueError('missing target scored')
            delay=16 if result['task_id'].startswith('delay-') else 4
            for s in event['labels_before']:
                if s>=t or s+delay>t or s in delivered:raise ValueError('illegal/duplicate feedback')
                delivered.add(s)
            for s in event['labels_after']:
                if s>t or s+delay>t or s in delivered:raise ValueError('illegal post-feedback')
                delivered.add(s)
            if event['state_bytes']>131072:raise ValueError('state cap violation')
            for k,v in event['usage'].items():
                if type(v) is not int or v<last_counts.get(k,0):raise ValueError('counter regression')
            last_counts=event['usage']
        if not count or total/count!=result['mse'] or result['scored']!=count or len(trace)!=result['predictions']:
            raise ValueError('aggregate mismatch')
        intended={e['step'] for e in trace if e['target'] is not None and e['step']+delay<len(trace)}
        if delivered!=intended:raise ValueError('feedback denominator mismatch')
        if len(delivered)!=result['usage']['labels']:raise ValueError('label count mismatch')
        if result['usage']!=trace[-1]['usage']:raise ValueError('result counter mismatch')
        if result['config']['kind']=='bank' and result['usage']['updates']+result['usage']['skips']!=len(delivered):
            raise ValueError('bank adaptation accounting mismatch')
        rows+=len(trace)
    if seen!=expected:raise ValueError('missing or extra matrix cells')
    return {'status':'VALID','attempts':len(seen),'failed':failures,'scored_trace_rows':rows,'auditor':'separate arithmetic and timing implementation; not external replication','artifact_map_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.input)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
