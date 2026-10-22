import copy
import json
import math
import unittest
from dataclasses import replace

from engine import Config, Policy, Runner, canonical, metrics, synthetic


class ScientificContracts(unittest.TestCase):
    def test_absolute_gate_rejects_less_bad_but_inaccurate_expert(self):
        p=Policy(1,Config(loss_ceiling=.02));p.loss=10.;p.count=10
        p.bank=[{'id':0,'w':[0.,0.],'count':10,'loss':1.}];p.next_id=1
        self.assertIsNone(p.winner())

    def test_rls_bank_all_update_active_matches_rls(self):
        rows=synthetic(33,steps=96)
        a=Runner(rows,Config(kind='bank',suppress=False,forgetting=.95))
        b=Runner(rows,Config(kind='rls',forgetting=.95))
        a.advance();b.advance()
        self.assertEqual(a.policy.w,b.policy.w)
        self.assertEqual(a.policy.cov,b.policy.cov)

    def test_nlms_exact_contraction(self):
        p=Policy(2,Config(kind='nlms',eta=.5))
        x=[2.,-1.]
        before,ticket=p.predict(x)
        p.observe(x,3.,ticket)
        after,_=p.predict(x)
        self.assertAlmostEqual(3.-after,(3.-before)*(1-.5*6/(6+1e-8)))

    def test_scalar_rls_matches_batch_ridge(self):
        p=Policy(1,Config(kind='rls',forgetting=1.))
        for _ in range(10):
            _,ticket=p.predict([0.])
            p.observe([0.],2.,ticket)
        pred,_=p.predict([0.])
        self.assertAlmostEqual(pred,20/10.1,places=12)

    def test_delayed_labels_and_zero_delay(self):
        rows=synthetic(7,steps=64)
        r=Runner(rows,Config(kind='nlms'))
        r.advance()
        self.assertEqual([t['updates_before'] for t in r.trace[:5]],[0,0,0,0,1])
        self.assertEqual(r.policy.usage['labels'],60)
        for row in rows: row['available']=row['step']
        r=Runner(rows,Config(kind='nlms'));r.advance()
        self.assertEqual(r.trace[0]['updates_before'],0)
        self.assertEqual(r.trace[0]['labels_after'],[0])
        self.assertEqual(r.policy.usage['labels'],64)

    def test_future_targets_do_not_change_prefix(self):
        a=synthetic(9,steps=64);b=copy.deepcopy(a)
        for row in b[20:]: row['y']+=10000
        ra=Runner(a,Config());rb=Runner(b,Config())
        ra.advance(24);rb.advance(24)
        self.assertEqual([r['prediction'] for r in ra.trace],[r['prediction'] for r in rb.trace])

    def test_new_expert_cannot_score_prebirth_predictions(self):
        p=Policy(1,Config(interval=1,minimum=1))
        _,old=p.predict([1.]);p.observe([1.],1.,old)
        self.assertEqual(p.bank[0]['count'],0)
        p.observe([1.],1.,old)
        self.assertEqual(p.bank[0]['count'],0)

    def test_capacity_and_eviction_identity(self):
        p=Policy(1,Config(interval=1,capacity=2,suppress=False))
        for i in range(10):
            _,ticket=p.predict([1.]);p.observe([1.],float(i),ticket)
        self.assertEqual([e['id'] for e in p.bank],[8,9])
        self.assertEqual(p.usage['evictions'],8)

    def test_gate_skips_only_after_minimum_observations(self):
        p=Policy(1,Config(minimum=2))
        p.bank=[{'id':0,'w':[0.,1.],'count':1,'loss':0.}];p.next_id=1;p.loss=1.;p.count=1
        self.assertIsNone(p.winner())
        _,ticket=p.predict([0.]);p.observe([0.],1.,ticket)
        self.assertEqual(p.usage['skips'],1)
        self.assertEqual(p.usage['updates'],0)

    def test_resume_identical_for_every_method(self):
        rows=synthetic(12,steps=96)
        for kind in ('bank','nlms','adam','rls','knn','replay','frozen','last'):
            c=Config(kind=kind,interval=4)
            full=Runner(rows,c);full.advance()
            for stop in (0,3,4,31,95):
                r=Runner(rows,c);r.advance(stop)
                restored=Runner.restore(rows,c,json.loads(canonical(r.snapshot())))
                restored.advance()
                self.assertEqual(full.snapshot(),restored.snapshot(),(kind,stop))

    def test_restore_rejects_changed_config_data_and_capacity(self):
        rows=synthetic(1,steps=64);r=Runner(rows,Config());r.advance(40)
        with self.assertRaises(ValueError):Runner.restore(rows,Config(eta=1.),r.snapshot())
        other=copy.deepcopy(rows);other[1]['y']+=1
        with self.assertRaises(ValueError):Runner.restore(other,Config(),r.snapshot())
        s=r.snapshot();s['policy']['w'].pop()
        with self.assertRaises(ValueError):Runner.restore(rows,Config(),s)

    def test_missing_labels_preserve_clock(self):
        rows=synthetic(1,steps=64);rows[0]['y']=None
        r=Runner(rows,Config());r.advance()
        self.assertEqual(metrics(r.trace)['scored'],63)
        self.assertEqual(r.trace[4]['labels_before'],[])
        self.assertEqual(r.trace[5]['labels_before'],[1])

    def test_reset_and_finite_contracts(self):
        rows=synthetic(5,steps=64)
        a=Runner(rows,Config());a.advance()
        b=Runner(rows,Config());b.advance()
        self.assertEqual(a.snapshot(),b.snapshot())
        with self.assertRaises(ValueError):Policy(2,Config()).predict([float('nan'),0])
        with self.assertRaises(ValueError):Policy(2,Config()).predict([0])
        with self.assertRaises(ValueError):Config(eta=-1)

    def test_no_regime_metadata_influences_policy(self):
        a=synthetic(8,steps=64);b=copy.deepcopy(a)
        for row in b:row['regime']=777;row['return']=False
        ra=Runner(a,Config());rb=Runner(b,Config());ra.advance();rb.advance()
        self.assertEqual([r['prediction'] for r in ra.trace],[r['prediction'] for r in rb.trace])

    def test_all_update_control_does_not_skip(self):
        c=Config(suppress=False,interval=2)
        r=Runner(synthetic(11,steps=64),c);r.advance()
        self.assertEqual(r.policy.usage['skips'],0)
        self.assertEqual(r.policy.usage['updates'],60)


if __name__=='__main__':unittest.main()
