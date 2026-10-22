"""Exposed Air Quality data loader; execution lives in transfer.py."""
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

