import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from piml.s2 import Predictor, features, system, matvec, factor, factored_solve, pcg, make_case, run, verify, source_manifest
from piml.model import RidgeOperator
from piml.pde import generate_case


class S2Tests(unittest.TestCase):
    def test_source_linearity_and_zero(self):
        c, f, g = np.array([.1,-.2,.3,-.1]), np.arange(4.), np.arange(4.)[::-1]
        for kind in ['linear','structured']:
            np.testing.assert_allclose(features(c,2*f-3*g,kind),2*features(c,f,kind)-3*features(c,g,kind),atol=1e-14)
            np.testing.assert_array_equal(features(c,np.zeros(4),kind),0)
        self.assertEqual(features(c,f,'structured').size,60)
        self.assertEqual(features(c,f,'quadratic').size,45)

    def test_fitted_predictor_does_not_read_test_truth(self):
        train = [make_case(300+i,17,'iid',str(i)) for i in range(24)]
        model = Predictor.fit(train,'structured',1e-4)
        held = make_case(999,17,'iid','held')
        before = model.predict(held)
        held['truth'] = [12345.] * 15
        np.testing.assert_array_equal(model.predict(held), before)

    def test_quadratic_optimization_equivalent_to_s1(self):
        old = [generate_case(str(i),'train',300+i,9) for i in range(12)]
        new = [make_case(300+i,9,'iid',str(i)) for i in range(12)]
        s1 = RidgeOperator.fit(old,1e-4)
        s2 = Predictor.fit(new,'quadratic',1e-4)
        held = generate_case('held','test',999,9)
        np.testing.assert_allclose(s1.predict(held)[1:-1],s2.predict(make_case(999,9,'iid','held')),atol=1e-10,rtol=1e-8)

    def test_independent_dense_reference_and_residual(self):
        for regime in ['iid','coefficient_ood','forcing_ood']:
            case = make_case(118,33,regime,'case')
            l,d,u,b = system(case)
            dense = np.diag(d)+np.diag(l,-1)+np.diag(u,1)
            truth = np.linalg.solve(dense,b)
            np.testing.assert_allclose(factored_solve(factor(l,d,u),b),truth,rtol=1e-12,atol=1e-13)
            for pre in ['jacobi','laplace']:
                x,k,apps,ok,hist,restarts = pcg(l,d,u,b,np.zeros_like(b),pre,1e-10,1e-12,128)
                self.assertTrue(ok)
                self.assertLessEqual(np.linalg.norm(b-dense@x),1.01*max(1e-12,1e-10*np.linalg.norm(b)))
                np.testing.assert_allclose(x,truth,rtol=1e-8,atol=1e-10)

    def test_constant_coefficient_reference_converges_in_one_step(self):
        n=31
        l,u,d = -np.ones(n-1),-np.ones(n-1),2*np.ones(n)
        b=np.cos(np.arange(n))
        x,k,apps,ok,hist,restarts=pcg(l,d,u,b,np.zeros(n),'laplace',1e-10,1e-12,124)
        self.assertTrue(ok);self.assertEqual(k,1)

    def test_manufactured_solution_refinement(self):
        errors=[]
        for n in [15,31,63]:
            h=1/(n+1);xs=np.arange(1,n+1)*h
            b=np.pi**2*np.sin(np.pi*xs)
            x=factored_solve(factor(-np.ones(n-1)/h**2,2*np.ones(n)/h**2,-np.ones(n-1)/h**2),b)
            errors.append(np.linalg.norm(x-np.sin(np.pi*xs))/np.sqrt(n))
        self.assertGreater(errors[0]/errors[1],3.8)
        self.assertGreater(errors[1]/errors[2],3.8)

    def test_nonfinite_inputs_and_zero_rhs(self):
        with self.assertRaises(ValueError):
            features(np.array([np.nan,0,0,0]),np.zeros(4),'structured')
        n=7;l=-np.ones(n-1);d=2*np.ones(n)
        result=pcg(l,d,l,np.zeros(n),np.zeros(n),'laplace',1e-10,1e-12,28)
        self.assertTrue(result[3]);self.assertEqual(result[1],0)
        with self.assertRaises(ValueError):
            pcg(l,d,l,np.ones(n),np.zeros(n),'laplace',float('nan'),1e-12,28)

    def test_frozen_source_mismatch_rejected_before_evaluation(self):
        config=json.loads(Path('configs/s2_smoke.json').read_text())
        config.update(phase='confirmatory',frozen=True)
        with tempfile.TemporaryDirectory() as tmp:
            freeze=Path(tmp)/'freeze.json'
            freeze.write_text(json.dumps({'source':{},'config_sha256':'bad'}))
            with self.assertRaisesRegex(ValueError,'changed'):
                run(config,Path(tmp)/'run',Path('.'),freeze)
            self.assertFalse((Path(tmp)/'run').exists())

    def test_full_matrix_and_semantic_replay_detect_rehashed_tampering(self):
        config=json.loads(Path('configs/s2_smoke.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            run(config,path,Path('.'))
            self.assertEqual(verify(path)['status'],'PASS')
            rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
            rows[0]['converged']=not rows[0]['converged']
            (path/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            from piml.experiment import sha256_file
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['metrics.jsonl']=sha256_file(path/'metrics.jsonl')
            (path/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify(path)['status'],'FAIL')


if __name__=='__main__':
    unittest.main()
