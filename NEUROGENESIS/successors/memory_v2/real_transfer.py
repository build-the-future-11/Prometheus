"""Chronological UCI transfer; synthetic-selected settings only, no real-data tuning."""
import argparse
import csv
import datetime as dt
import gzip
import json
from pathlib import Path
import statistics

from engine import Config, Runner, digest, metrics
from experiment import ROOT, admission, environment, execute, now, read, sha, source_map, write


FEATURES=[('PT08.S1(CO)',1000.),('PT08.S2(NMHC)',1000.),('PT08.S3(NOx)',1000.),('PT08.S4(NO2)',1000.),('PT08.S5(O3)',1000.),('T',50.),('RH',100.),('AH',2.)]


def load_data(path):
    rows=[]
    previous=None
    for raw in csv.DictReader(Path(path).read_text().splitlines(),delimiter=';'):
        if not raw['Date']:
            continue
        stamp=dt.datetime.strptime(raw['Date']+' '+raw['Time'],'%d/%m/%Y %H.%M.%S')
        if previous is not None and stamp-previous != dt.timedelta(hours=1):
            raise ValueError('dataset time grid is not hourly; protocol needs amendment before evaluation')
        previous=stamp
        values=[float(raw[name].replace(',','.')) for name,_ in FEATURES]
        x=[0. if v == -200 else v/scale for v,(_,scale) in zip(values,FEATURES)]
        x += [float(v == -200) for v in values]
        target=float(raw['CO(GT)'].replace(',','.'))
        t=len(rows)
        rows.append({'step':t,'x':x,'y':None if target == -200 else target/10.,'available':t+4,'regime':0,'return':False})
    return rows


def run(output,lock_path):
    lock=read(lock_path)
    if source_map()!=lock['source']:
        raise ValueError('frozen source drift')
    data_meta=read(ROOT/'data/manifest.json')
    if sha(ROOT/'data/AirQualityUCI.csv')!=data_meta['csv_sha256']:
        raise ValueError('dataset digest mismatch')
    output.mkdir(parents=True,exist_ok=False)
    admission(output)
    rows=load_data(ROOT/'data/AirQualityUCI.csv')
    boundary=int(.4*len(rows))
    configs=dict(lock['configs'])
    configs['last_observation']={'kind':'last'}
    write(output/'manifest.json',{'utc':now(),'phase':'real_transfer','source':source_map(),'lock_sha256':digest(lock),'data':data_meta,'data_sha256':digest(rows),'rows':len(rows),'test_start':boundary,'test_end':len(rows),'scoring':'first 40 percent warmup, last 60 percent protected transfer, state continues legally','configs':configs,'environment':environment()})
    summaries={}
    for name,raw in configs.items():
        result=execute(output,'uci-air-quality',rows,name,Config(**raw))
        if result['status']!='COMPLETED':
            summaries[name]={'status':'FAILED','failure':result['failure']}
            continue
        path=output/'attempts'/f'uci-air-quality__{name}'/'trace.jsonl.gz'
        trace=[json.loads(line) for line in gzip.open(path,'rt')]
        test=trace[boundary:]
        mse=metrics(test)['mse']
        updates=trace[-1]['usage']['updates']-trace[boundary-1]['usage']['updates']
        blocks=[]
        for i in range(7):
            segment=test[len(test)*i//7:len(test)*(i+1)//7]
            blocks.append(metrics(segment)['mse']*100)
        summaries[name]={'status':'COMPLETED','test_mse_scaled':mse,'test_mse_original_units':mse*100,'test_scored':metrics(test)['scored'],'test_updates':updates,'test_block_mse_original_units':blocks,'total_wall_seconds':result['wall_seconds'],'peak_state_bytes':result['peak_state_bytes'],'total_usage':result['usage']}
        print('real transfer:',name,summaries[name],flush=True)
    write(output/'summary.json',{'results':summaries,'independent_datasets':1,'block_interpretation':'descriptive contiguous time blocks; no IID claim','artificial_delay_hours':4,'utc':now()})


def verify(output,lock_path,replay=False):
    output=Path(output);m=read(output/'manifest.json');lock=read(lock_path)
    if m['source']!=source_map() or m['lock_sha256']!=digest(lock):
        raise ValueError('real transfer provenance mismatch')
    rows=load_data(ROOT/'data/AirQualityUCI.csv')
    if digest(rows)!=m['data_sha256'] or sha(ROOT/'data/AirQualityUCI.csv')!=m['data']['csv_sha256']:
        raise ValueError('real data mismatch')
    summaries=read(output/'summary.json')['results']
    if set(summaries)!=set(m['configs']):
        raise ValueError('real transfer coverage mismatch')
    verified=0
    for name,config in m['configs'].items():
        path=output/'attempts'/f'uci-air-quality__{name}'
        result=read(path/'result.json')
        if result['status']!='COMPLETED':
            raise ValueError('real transfer has failed cell')
        if sha(path/'trace.jsonl.gz')!=result['trace_sha256'] or sha(path/'checkpoint.json.gz')!=result['checkpoint_sha256']:
            raise ValueError('real transfer artifact mismatch')
        trace=[json.loads(line) for line in gzip.open(path/'trace.jsonl.gz','rt')]
        if len(trace)!=len(rows):raise ValueError('real trace length')
        for row,source in zip(trace,rows):
            if row['target']!=source['y'] or row['step']!=source['step']:
                raise ValueError('real trace source mismatch')
            if source['y'] is not None and row['squared_error']!=(row['prediction']-source['y'])**2:
                raise ValueError('real trace score mismatch')
            for s in row['labels_before']:
                if s>=row['step'] or rows[s]['available']>row['step']:raise ValueError('real label leak')
        test=trace[m['test_start']:]
        summary=summaries[name]
        if metrics(test)['mse']*100!=summary['test_mse_original_units'] or metrics(test)['scored']!=summary['test_scored']:
            raise ValueError('real summary mismatch')
        if replay:
            runner=Runner(rows,Config(**config));runner.advance()
            if runner.trace!=trace:raise ValueError('real replay mismatch')
        verified+=len(trace)
    return {'status':'VALID','methods':len(summaries),'verified_trace_rows':verified,'exact_replay':replay,'utc':now()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--replay',action='store_true');a=p.parse_args()
    if a.command=='run':run(a.output,a.lock)
    else:write(a.output,verify(a.input,a.lock,a.replay))
