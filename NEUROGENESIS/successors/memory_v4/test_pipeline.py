import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from engine import Config, Runner, synthetic, digest
from experiment import execute, read, bootstrap, grid, DEV_SEEDS, TEST_SEEDS
from real_transfer import FEATURES, load_data


class PipelineContracts(unittest.TestCase):
    def test_real_dataset_identity_and_calendar(self):
        from transfer import data
        rows,boundary,scale,meta=data('beijing')
        self.assertEqual((len(rows),boundary,scale),(17520,8760,10000.))
        self.assertEqual(len(rows[0]['x']),18)
        self.assertTrue(all(r['available']==r['step']+4 for r in rows))
        self.assertEqual(sum(r['y'] is not None for r in rows[boundary:]),8661)

    def test_grid_and_split(self):
        self.assertEqual(sum(k.startswith('bank_') for k in grid()),8)
        self.assertEqual(len(grid()),22)
        self.assertFalse(set(DEV_SEEDS)&set(TEST_SEEDS))
        self.assertFalse({101,211,307,401,503}&set(DEV_SEEDS+TEST_SEEDS))

    def test_bootstrap_known_constant(self):
        self.assertEqual(bootstrap([-.1]*20,100),[-.1,-.1])

    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):read(p)

    def test_execution_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=synthetic(22,steps=64)
            r=execute(root,'test',data,'bank',Config())
            self.assertEqual(r['status'],'COMPLETED')
            self.assertEqual(r['task_sha256'],digest(data))
            path=root/'attempts/test__bank/trace.jsonl.gz'
            self.assertEqual(len(gzip.open(path,'rt').readlines()),64)
            with self.assertRaises(FileExistsError):execute(root,'test',data,'bank',Config())

    def test_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=synthetic(22,steps=64)
            data[3]['x']=[0.]
            r=execute(root,'bad',data,'bank',Config())
            self.assertEqual(r['status'],'FAILED')
            self.assertIn('dimension mismatch',r['failure'])
            self.assertTrue((root/'attempts/bad__bank/trace.jsonl.gz').exists())

    def test_real_missing_rows_and_physical_scaling(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv'
            names=['Date','Time','CO(GT)']+[n for n,_ in FEATURES]
            with p.open('w') as f:
                w=csv.DictWriter(f,fieldnames=names,delimiter=';');w.writeheader()
                for i in range(3):
                    row={n:'1000' for n,_ in FEATURES};row.update({'Date':'10/03/2004','Time':f'{18+i:02d}.00.00','CO(GT)':'-200' if i==1 else '2,0'})
                    if i==1:row['T']='-200'
                    w.writerow(row)
            rows=load_data(p)
            self.assertEqual(len(rows),3);self.assertIsNone(rows[1]['y'])
            self.assertEqual(rows[2]['available'],6)
            self.assertEqual(rows[0]['x'][0],1.)
            self.assertEqual(rows[1]['x'][5],0.)
            self.assertEqual(rows[1]['x'][13],1.)
            self.assertEqual(rows[0]['y'],.2)


if __name__=='__main__':unittest.main()
