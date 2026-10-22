"""Verify retained successful AND failed transfer artifacts, without admitting efficacy.

The frozen efficacy verifier correctly rejects failed cells. This supplementary
auditor validates failure evidence; it never upgrades an incomplete comparison.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys


def verify(root, source, replay=False):
    sys.path.insert(0, str(source))
    from engine import Config, Runner, digest
    from real_transfer import load_data
    from experiment import source_map
    read = lambda p: json.loads(p.read_text())
    manifest = read(root / 'manifest.json')
    summary = read(root / 'summary.json')['results']
    if manifest['source'] != source_map():
        raise ValueError('source drift')
    rows = load_data(source / 'data/AirQualityUCI.csv')
    if digest(rows) != manifest['data_sha256']:
        raise ValueError('dataset drift')
    if set(summary) != set(manifest['configs']):
        raise ValueError('coverage mismatch')
    completed, failed = [], []
    count = 0
    for method, config in manifest['configs'].items():
        path = root / 'attempts' / ('uci-air-quality__' + method)
        result = read(path / 'result.json')
        if result['config'] != asdict(Config(**config)) or result['status'] != summary[method]['status']:
            raise ValueError('identity/status mismatch')
        for kind, filename in [('trace', 'trace.jsonl.gz'), ('checkpoint', 'checkpoint.json.gz')]:
            if hashlib.sha256((path / filename).read_bytes()).hexdigest() != result[kind + '_sha256']:
                raise ValueError('artifact digest mismatch')
        trace = [json.loads(line) for line in gzip.open(path / 'trace.jsonl.gz', 'rt')]
        checkpoint = json.loads(gzip.open(path / 'checkpoint.json.gz', 'rt').read())
        for t, row in enumerate(trace):
            if row['step'] != t or row['target'] != rows[t]['y']:
                raise ValueError('trace identity mismatch')
            expected = None if row['target'] is None else (row['prediction'] - row['target'])**2
            if expected != row['squared_error']:
                raise ValueError('trace arithmetic mismatch')
            for s in row['labels_before'] + row['labels_after']:
                if s >= t or rows[s]['available'] > t or rows[s]['y'] is None:
                    raise ValueError('illegal delayed feedback')
        if result['status'] == 'COMPLETED':
            if len(trace) != len(rows):
                raise ValueError('successful trace incomplete')
            test = trace[manifest['test_start']:]
            scored = [r['squared_error'] for r in test if r['squared_error'] is not None]
            if sum(scored) / len(scored) * 100 != summary[method]['test_mse_original_units']:
                raise ValueError('transfer aggregate mismatch')
            if trace[-1]['usage']['updates'] - trace[manifest['test_start']-1]['usage']['updates'] != summary[method]['test_updates']:
                raise ValueError('transfer update count mismatch')
            completed.append(method)
        else:
            if 'RLS lost positive covariance' not in result['failure']:
                raise ValueError('unexpected failure class')
            failed.append({'method': method, 'predictions_before_failure': len(trace), 'reason': 'RLS lost positive covariance'})
        if replay:
            runner = Runner(rows, Config(**config))
            failure = None
            try:
                runner.advance()
            except ArithmeticError as exc:
                failure = str(exc)
            if (failure is None) != (result['status'] == 'COMPLETED'):
                raise ValueError('replay terminal state differs')
            if failure is not None and failure != 'RLS lost positive covariance':
                raise ValueError('replay failure differs')
            if runner.trace != trace or runner.snapshot() != checkpoint:
                raise ValueError('exact replay differs')
        count += len(trace)
    return {'artifact_status': 'VALID', 'scientific_comparison': 'INVALID_NUMERICAL_FAILURE' if failed else 'COMPLETE', 'methods': len(summary), 'completed': completed, 'failed': failed, 'trace_rows': count, 'exact_replay': replay, 'external_replication': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    result = verify(args.input, args.source, args.replay)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
