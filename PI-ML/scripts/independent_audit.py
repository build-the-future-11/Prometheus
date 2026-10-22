"""A separate artifact audit: no imports from piml and no solver implementation reuse.

Checks raw data with SciPy's banded direct solver and independently computes block
effects and the primary decision. This is an independent implementation check,
not independent authorship or an externally reproduced research claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


def audit(path):
    failures=[]
    config=json.loads((path/'config.json').read_text())
    analysis=json.loads((path/'analysis.json').read_text())
    rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    cases={c['id']:c for c in map(json.loads,(path/'cases.jsonl').read_text().splitlines())}
    max_reference_difference=0.
    for c in cases.values():
        a=np.array(c['diffusivity']);h=1/(c['grid']-1)
        faces=(a[:-1]+a[1:])/(2*h*h)
        diagonal=faces[:-1]+faces[1:];off=-faces[1:-1]
        ab=np.zeros((3,len(diagonal)))
        ab[1]=diagonal;ab[0,1:]=off;ab[2,:-1]=off
        expected=solve_banded((1,1),ab,np.array(c['forcing'])[1:-1])
        relative=float(np.linalg.norm(expected-c['truth'])/max(np.linalg.norm(expected),1e-30))
        max_reference_difference=max(max_reference_difference,relative)
        if relative>1e-10:failures.append('reference:'+c['id'])
    for row in rows:
        c=cases[row['case_id']]
        x=np.r_[0.,row['solution'],0.]
        a=np.array(c['diffusivity']);h=1/(c['grid']-1)
        flux=.5*(a[:-1]+a[1:])*np.diff(x)/h
        applied=-np.diff(flux)/h
        b=np.array(c['forcing'])[1:-1]
        actual=float(np.linalg.norm(b-applied))
        threshold=max(config['atol'],row['rtol']*float(np.linalg.norm(b)))
        # Algebraically independent flux form may differ by roundoff from the
        # matrix form; require agreement within a separately explicit margin.
        margin=1e-11*max(float(np.linalg.norm(b)),1.)
        if row['converged'] and actual>threshold+margin:failures.append('flux-residual:'+row['case_id'])
        if len(row['timing_samples_ns'])!=config['timing_repeats']:failures.append('timing-count')
    gate=config['primary']
    selected=[r for r in rows if r['grid']==gate['grid'] and r['rtol']==gate['rtol'] and r['regime']=='iid']
    lookup={(r['rep'],r['case_id'],r['method']):r for r in selected}
    primary=[]
    for baseline in ['zero_laplace','physics_laplace']:
        block_means=[]
        for rep in range(config['repetitions']):
            paired=[]
            for r in selected:
                if r['rep']!=rep or r['method']!='structured_laplace':continue
                b=lookup[rep,r['case_id'],baseline]
                paired.append((b['iterations']-r['iterations'])/max(b['iterations'],1))
            block_means.append(math.fsum(paired)/len(paired))
        saved=next(c for c in analysis['comparisons'] if c['grid']==gate['grid'] and c['rtol']==gate['rtol'] and c['regime']=='iid' and c['baseline']==baseline and c['proposed']=='structured_laplace')
        if not np.allclose(block_means,saved['training_block_effects'],rtol=1e-13,atol=1e-13):failures.append('block-effects:'+baseline)
        # Enumerate the full finite bootstrap distribution over five blocks.
        # Its quantiles need not equal Monte Carlo quantiles byte-for-byte;
        # compare the final decision and report the independent interval.
        from itertools import product
        bootstrap=np.array([math.fsum(block_means[i] for i in indices)/len(block_means)
                   for indices in product(range(len(block_means)),repeat=len(block_means))])
        interval=np.quantile(bootstrap,[.025,.975]).tolist()
        mean=math.fsum(block_means)/len(block_means)
        converged=all(r['converged'] for r in selected if r['method'] in ['structured_laplace',baseline])
        primary.append(dict(baseline=baseline,mean=mean,exact_finite_bootstrap_ci95=interval,
                            passed=converged and mean>=gate['minimum_mean'] and interval[0]>=gate['minimum_lower']))
    independent_passed=all(p['passed'] for p in primary)
    if independent_passed!=analysis['primary_passed']:failures.append('decision-exact-bootstrap')
    return dict(status='PASS' if not failures else 'FAIL',failures=failures,cases=len(cases),rows=len(rows),
        max_relative_reference_difference=max_reference_difference,independent_primary=primary,
        scipy_reference='scipy.linalg.solve_banded, separately assembled flux operator',
        flux_acceptance_roundoff_margin='1e-11 * max(norm(rhs),1); does not replace the original strict gate',
        scope='Independent implementation audit by the same research agent; not independent investigator reproduction')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
