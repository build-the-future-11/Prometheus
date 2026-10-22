import copy
import math
import random
import unittest
from engine import Config, Policy


def solve(a,b):
    """Independent pivoted Gaussian elimination reference."""
    m=[r[:]+[y] for r,y in zip(a,b)];n=len(b)
    for j in range(n):
        k=max(range(j,n),key=lambda i:abs(m[i][j]));m[j],m[k]=m[k],m[j]
        z=m[j][j];m[j]=[v/z for v in m[j]]
        for i in range(n):
            if i!=j:
                z=m[i][j];m[i]=[u-z*v for u,v in zip(m[i],m[j])]
    return [r[-1] for r in m]


class PersistentRidgeContracts(unittest.TestCase):
    def test_objective_matches_independent_batch(self):
        for seed in range(10):
            for lam in (.95,.99,1.):
                rng=random.Random(seed);p=Policy(4,Config(kind='rls',forgetting=lam));samples=[]
                for t in range(40):
                    x=[rng.gauss(0,1) for _ in range(4)];y=rng.gauss(0,1)
                    _,ticket=p.predict(x);p.observe(x,y,ticket);samples.append((x+[1.],y))
                    a=[[float(i==j)+math.fsum(lam**(t-k)*z[i]*z[j] for k,(z,_) in enumerate(samples)) for j in range(5)] for i in range(5)]
                    b=[math.fsum(lam**(t-k)*z[i]*v for k,(z,v) in enumerate(samples)) for i in range(5)]
                    expected=solve(a,b)
                    self.assertLess(max(abs(u-v) for u,v in zip(p.w,expected)),1e-11)

    def test_unexcited_and_duplicate_features_remain_bounded(self):
        p=Policy(4,Config(kind='rls',forgetting=.95))
        for i in range(5000):
            x=[0.,0.,float(i%2),float(i%2)]
            _,ticket=p.predict(x);p.observe(x,2.,ticket)
        self.assertEqual(p.information[0][0],1.)
        self.assertEqual(p.w[0],0.)
        self.assertLess(max(map(abs,p.w)),3.)

    def test_periodic_updates_information_on_every_label(self):
        p=Policy(1,Config(kind='periodic',forgetting=.95));q=Policy(1,Config(kind='rls',forgetting=.95))
        for i in range(12):
            for model in (p,q):
                _,ticket=model.predict([float(i)]);model.observe([float(i)],2.,ticket)
            self.assertEqual(p.information,q.information)
            if (i+1)%3==0:self.assertEqual(p.w,q.w)
        self.assertEqual(p.usage['updates'],4)
        self.assertEqual(p.usage['information_updates'],12)

    def test_skip_keeps_information_response_and_weights(self):
        p=Policy(1,Config(minimum=1));p.loss=1.;p.count=1
        p.bank=[{'id':0,'w':[0.,1.],'count':1,'loss':0.}];p.next_id=1
        old=copy.deepcopy((p.information,p.response,p.w))
        _,ticket=p.predict([0.]);p.observe([0.],1.,ticket)
        self.assertEqual(old,(p.information,p.response,p.w))
        self.assertEqual(p.usage['skips'],1)


if __name__=='__main__':unittest.main()
