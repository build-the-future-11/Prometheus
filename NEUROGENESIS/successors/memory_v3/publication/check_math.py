"""Independent batch linear-algebra check of frozen RLS implementation."""
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine import Policy,Config


def solve(matrix,vector):
    a=[list(row)+[value] for row,value in zip(matrix,vector)]
    n=len(a)
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(a[j][i]))
        a[i],a[pivot]=a[pivot],a[i]
        scale=a[i][i]
        if abs(scale)<1e-15:raise ValueError('singular hand-check system')
        a[i]=[v/scale for v in a[i]]
        for j in range(n):
            if j!=i:
                scale=a[j][i]
                a[j]=[v-scale*w for v,w in zip(a[j],a[i])]
    return [row[-1] for row in a]


def check():
    maximum=0.;cases=0
    for seed in range(10):
        r=random.Random(seed)
        data=[([r.gauss(0,1) for _ in range(3)],r.gauss(0,1)) for _ in range(20)]
        for forgetting in (.95,.99,1.):
            for gated in (False,True):
                p=Policy(3,Config(kind='rls',forgetting=forgetting))
                used=[pair for i,pair in enumerate(data) if not gated or i%3!=0]
                for x,y in used:
                    _,ticket=p.predict(x);p.observe(x,y,ticket)
                n=len(used);a=[[forgetting**n/10*(i==j) for j in range(4)] for i in range(4)];b=[0.]*4
                for t,(x,y) in enumerate(used):
                    x=x+[1.];weight=forgetting**(n-t-1)
                    for i in range(4):
                        b[i]+=weight*x[i]*y
                        for j in range(4):a[i][j]+=weight*x[i]*x[j]
                exact=solve(a,b);error=max(abs(u-v) for u,v in zip(exact,p.w));maximum=max(maximum,error);cases+=1
                if error>1e-10:raise AssertionError((seed,forgetting,gated,error))
    return {'status':'VALID','cases':cases,'max_abs_coefficient_error':maximum,'tolerance':1e-10,'method':'pivoted Gaussian elimination of batch weighted normal equations; full and skipped update sequences','scope':'mathematical implementation check, not efficacy or independent external replication'}


if __name__=='__main__':
    result=check();path=Path(sys.argv[1]);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
