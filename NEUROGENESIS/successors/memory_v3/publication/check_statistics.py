"""Recompute the registered statistics without importing experiment/model code."""
import argparse
import json
import random
from pathlib import Path


def check(runs):
    lock = json.loads((runs / 'freeze.json').read_text())
    analysis = json.loads((runs / 'analysis.json').read_text())
    protocol = lock['protocol']
    records = [json.loads(p.read_text()) for p in (runs / 'confirmation-02/attempts').glob('*/result.json')]
    index = {(r['task_id'], r['method']): r for r in records}
    effects = []
    reductions = []
    for seed in protocol['confirmation_seeds']:
        pairs = [(index[(f'{c}-{seed}', 'candidate')], index[(f'{c}-{seed}', 'baseline')]) for c in protocol['primary_conditions']]
        effects.append(sum(a['mse'] for a, b in pairs) / sum(b['mse'] for a, b in pairs) - 1)
        reductions.append(1 - sum(a['usage']['updates'] for a, b in pairs) / sum(b['usage']['updates'] for a, b in pairs))
    rng = random.Random(protocol['bootstrap_seed'])
    samples = sorted(sum(effects[int(rng.random() * len(effects))] for _ in effects) / len(effects) for _ in range(protocol['bootstrap_draws']))
    ci = [samples[int(.025 * len(samples))], samples[int(.975 * len(samples)) - 1]]
    effect = sum(effects) / len(effects)
    reduction = sum(reductions) / len(reductions)
    expected = analysis['primary']
    errors = [abs(effect - expected['relative_mse_change']), abs(reduction - expected['mean_update_reduction'])] + [abs(a - b) for a, b in zip(ci, expected['ci95'])]
    if max(errors) > 1e-12:
        raise ValueError('Independent statistical calculation disagrees')
    support = ci[1] <= protocol['noninferiority_margin'] and reduction >= protocol['minimum_update_reduction']
    if support != (analysis['disposition'] == 'SUPPORTED'):
        raise ValueError('Disposition disagrees')
    return {'status': 'VALID', 'seeds': len(effects), 'primary_attempts_per_method': len(effects) * len(protocol['primary_conditions']), 'relative_mse_change': effect, 'ci95': ci, 'mean_update_reduction': reduction, 'maximum_difference': max(errors), 'external_replication': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.runs)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
