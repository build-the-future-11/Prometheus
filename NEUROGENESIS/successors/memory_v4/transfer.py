"""Frozen chronological transfer and exposed regression, with failure-aware replay."""
import argparse
import csv
import datetime as dt
import gzip
import json
import math
import time
from pathlib import Path
from engine import Config, Runner, digest, metrics
from experiment import ROOT, admission, environment, execute, now, read, sha, source_map, write
from real_transfer import load_data as air_data

BEIJING='PRSA_data_2010.1.1-2014.12.31.csv'
FEATURES=[('DEWP',50.,0.),('TEMP',50.,0.),('PRES',100.,1000.),('Iws',1000.,0.),('Is',24.,0.),('Ir',24.,0.)]


def beijing_data(path):
    rows=[];previous=None;count=0
    for raw in csv.DictReader(Path(path).read_text().splitlines()):
        stamp=dt.datetime(*(int(raw[k]) for k in ('year','month','day','hour')))
        if previous is not None and stamp-previous!=dt.timedelta(hours=1):raise ValueError('nonhourly Beijing data')
        previous=stamp;count+=1
        if stamp.year not in (2013,2014):continue
        values=[None if raw[k]=='NA' else float(raw[k]) for k,_,_ in FEATURES]
        x=[0. if v is None else (v-offset)/scale for v,(_,scale,offset) in zip(values,FEATURES)]
        if raw['cbwd'] not in ('NE','NW','SE','cv'):raise ValueError('unknown wind category')
        x += [float(raw['cbwd']==k) for k in ('NE','NW','SE','cv')]
        x += [math.sin(2*math.pi*stamp.hour/24),math.cos(2*math.pi*stamp.hour/24)]
        x += [float(v is None) for v in values]
        t=len(rows);rows.append({'step':t,'x':x,'y':None if raw['pm2.5']=='NA' else float(raw['pm2.5'])/100.,'available':t+4,'regime':0,'return':False})
    if count!=43824 or len(rows)!=17520:raise ValueError('unexpected Beijing calendar coverage')
    return rows


def data(dataset):
    meta=read(ROOT/'data/manifest.json')[dataset];path=ROOT/'data'/meta['file']
    if sha(path)!=meta['sha256']:raise ValueError('data digest mismatch')
    rows=beijing_data(path) if dataset=='beijing' else air_data(path)
    return rows,8760 if dataset=='beijing' else int(.4*len(rows)),10000. if dataset=='beijing' else 100.,meta


def measured(trace,boundary,scale):
    test=trace[boundary:];m=metrics(test)
    usage={k:trace[-1]['usage'][k]-trace[boundary-1]['usage'][k] for k in trace[-1]['usage']}
    return {'test_mse_scaled':m['mse'],'test_mse_original_units':scale*m['mse'],'test_scored':m['scored'],'test_usage':usage,'test_updates':usage['updates'],'test_block_mse_original_units':[metrics(test[len(test)*i//7:len(test)*(i+1)//7])['mse']*scale for i in range(7)]}


def run(output,lock_path,dataset):
    lock=read(lock_path)
    if source_map()!=lock['source']:raise ValueError('frozen source drift')
    output.mkdir(parents=True,exist_ok=False);admission(output)
    rows,boundary,scale,meta=data(dataset);configs=dict(lock['configs']);configs['last_observation']={'kind':'last'}
    write(output/'manifest.json',{'schema':'neurogenesis.transfer.v4','utc':now(),'phase':'protected_transfer' if dataset=='beijing' else 'exposed_regression','dataset':dataset,'source':source_map(),'lock_sha256':digest(lock),'data':meta,'task_sha256':digest(rows),'rows':len(rows),'test_start':boundary,'scale_squared':scale,'configs':configs,'environment':environment()})
    summaries={};start=time.perf_counter()
    for name,raw in configs.items():
        if time.perf_counter()-start>1800:
            write(output/'interruption.json',{'reason':'phase wall budget','completed':len(summaries)});raise TimeoutError('phase wall budget')
        result=execute(output,dataset,rows,name,Config(**raw))
        item={'status':result['status'],'wall_seconds':result['wall_seconds']}
        if result['status']=='COMPLETED':
            trace=[json.loads(line) for line in gzip.open(output/'attempts'/f'{dataset}__{name}'/'trace.jsonl.gz','rt')]
            item.update(measured(trace,boundary,scale));item['peak_state_bytes']=result['peak_state_bytes']
            del trace
        else:item['failure']=result['failure']
        summaries[name]=item;print(dataset,name,item,flush=True)
    complete=all(x['status']=='COMPLETED' for x in summaries.values())
    support=False;relative=None;reduction=None
    if complete:
        a=summaries['candidate'];b=summaries['baseline'];relative=a['test_mse_scaled']/b['test_mse_scaled']-1
        reduction=1-a['test_updates']/b['test_updates'] if b['test_updates'] else None
        support=relative<=.02 and reduction is not None and reduction>=.2
    write(output/'summary.json',{'results':summaries,'all_completed':complete,'descriptive_support':support,'relative_mse_change':relative,'update_reduction':reduction,'phase':'protected_transfer' if dataset=='beijing' else 'exposed_regression','scope':'one chronological stream, no IID or generalization claim','utc':now()})


def verify(output,lock_path,replay=False):
    m=read(output/'manifest.json');lock=read(lock_path)
    if m['source']!=source_map() or m['lock_sha256']!=digest(lock):raise ValueError('provenance mismatch')
    rows,boundary,scale,meta=data(m['dataset']);summaries=read(output/'summary.json')['results']
    if digest(rows)!=m['task_sha256'] or meta!=m['data'] or boundary!=m['test_start']:raise ValueError('task mismatch')
    expected=set(m['configs'])
    paths=list((output/'attempts').glob('*/result.json'))
    if set(summaries)!=expected or len(paths)!=len(expected):raise ValueError('coverage mismatch')
    verified=0;failed=0
    for name,config in m['configs'].items():
        path=output/'attempts'/f'{m["dataset"]}__{name}';result=read(path/'result.json')
        if result['config']!=Config(**config).__dict__ or result['task_sha256']!=digest(rows):raise ValueError('attempt identity')
        if sha(path/'trace.jsonl.gz')!=result['trace_sha256'] or sha(path/'checkpoint.json.gz')!=result['checkpoint_sha256']:raise ValueError('artifact hash')
        trace=[json.loads(line) for line in gzip.open(path/'trace.jsonl.gz','rt')];checkpoint=json.loads(gzip.open(path/'checkpoint.json.gz','rt').read())
        delivered=[]
        for t,row in enumerate(trace):
            source=rows[t]
            error=None if source['y'] is None else (row['prediction']-source['y'])**2
            if row['step']!=t or row['target']!=source['y'] or row['squared_error']!=error:raise ValueError('score mismatch')
            for s in row['labels_before']+row['labels_after']:
                if s>t or rows[s]['available']>t or rows[s]['y'] is None:raise ValueError('label leak')
                delivered.append(s)
        if result['status']=='COMPLETED':
            if len(trace)!=len(rows):raise ValueError('incomplete completed trace')
            intended=[r['step'] for r in rows if r['y'] is not None and r['available']<len(rows)]
            if sorted(delivered)!=intended or len(set(delivered))!=len(delivered):raise ValueError('label coverage')
            for k,v in measured(trace,boundary,scale).items():
                if summaries[name][k]!=v:raise ValueError('summary mismatch')
        else:
            failed+=1
            if not result['failure'] or summaries[name]['status']!='FAILED':raise ValueError('failure receipt missing')
        if replay:
            runner=Runner(rows,Config(**config));failure=None
            try:runner.advance()
            except Exception as exc:failure=type(exc).__name__
            if (failure is None)!=(result['status']=='COMPLETED') or runner.trace!=trace or runner.snapshot()!=checkpoint:raise ValueError('exact replay mismatch')
        verified+=len(trace)
    return {'status':'VALID','scientific_comparison':'INVALID' if failed else 'COMPLETE','methods':len(expected),'failed':failed,'verified_trace_rows':verified,'exact_replay':replay,'utc':now()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--dataset',choices=['air','beijing']);p.add_argument('--replay',action='store_true');a=p.parse_args()
    if a.command=='run':run(a.output,a.lock,a.dataset)
    else:write(a.output,verify(a.input,a.lock,a.replay))
