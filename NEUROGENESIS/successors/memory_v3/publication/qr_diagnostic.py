"""Post-outcome numerical diagnostic, NOT part of the frozen candidate.

Square-root information RLS via Givens rotations. Uses exactly the same
accepted-update weighted ridge objective; no new regularization or tuning.
This diagnostic does not establish efficacy or amend the registered result.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import Config, Policy, Runner, finite, metrics
from real_transfer import load_data
from check_math import solve


class QRPolicy(Policy):
    def __init__(self, dimension, config):
        super().__init__(dimension, config)
        self.qr = [[math.sqrt(.1) * (i == j) for j in range(self.d)] for i in range(self.d)] if self.cov else []
        self.response = [0.] * self.d if self.cov else []
        self.cov = []

    def rls(self, features, target):
        scale = math.sqrt(self.config.forgetting)
        self.qr = [[scale * v for v in row] for row in self.qr]
        self.response = [scale * v for v in self.response]
        row = list(features)
        response = target
        for i in range(self.d):
            h = math.hypot(self.qr[i][i], row[i])
            if h == 0:
                raise ArithmeticError('QR zero diagonal')
            c, s = self.qr[i][i] / h, row[i] / h
            for j in range(i, self.d):
                a, b = self.qr[i][j], row[j]
                self.qr[i][j], row[j] = c * a + s * b, -s * a + c * b
            a, b = self.response[i], response
            self.response[i], response = c * a + s * b, -s * a + c * b
        for i in reversed(range(self.d)):
            self.w[i] = (self.response[i] - sum(self.qr[i][j] * self.w[j] for j in range(i + 1, self.d))) / self.qr[i][i]
        finite(self.w + self.response + [v for row in self.qr for v in row])
        self.usage['updates'] += 1
        self.usage['coordinate_ops'] += 6 * self.d * self.d + 12 * self.d

    def numeric_bytes(self):
        return super().numeric_bytes() + 8 * (self.d * len(self.qr) + len(self.response))


def run(output):
    import random
    checks = []
    for seed in range(10):
        rng = random.Random(seed)
        data = [([rng.gauss(0, 1) for _ in range(3)], rng.gauss(0, 1)) for _ in range(20)]
        for forgetting in (.95, .99, 1.):
            policy = QRPolicy(3, Config(kind='rls', forgetting=forgetting))
            for x, y in data:
                _, ticket = policy.predict(x)
                policy.observe(x, y, ticket)
            a = [[forgetting**len(data) / 10 * (i == j) for j in range(4)] for i in range(4)]
            b = [0.] * 4
            for t, (x, y) in enumerate(data):
                x = x + [1.]
                weight = forgetting**(len(data) - t - 1)
                for i in range(4):
                    b[i] += weight * x[i] * y
                    for j in range(4):
                        a[i][j] += weight * x[i] * x[j]
            error = max(abs(u-v) for u, v in zip(solve(a, b), policy.w))
            if error > 1e-10:
                raise ValueError('QR objective mismatch')
            checks.append(error)
    rows = load_data(ROOT / 'data/AirQualityUCI.csv')
    lock = json.loads((ROOT / 'runs/freeze.json').read_text())
    results = {}
    for method in ('baseline', 'candidate'):
        config = Config(**lock['configs'][method])
        runner = Runner(rows, config)
        runner.policy = QRPolicy(len(rows[0]['x']), config)
        try:
            runner.advance()
            measured = metrics(runner.trace[int(.4*len(rows)):])
            results[method] = {'status': 'COMPLETED', 'mse_original_units': measured['mse'] * 100, 'max_abs_weight': max(map(abs, runner.policy.w)), 'predictions': len(runner.trace)}
        except Exception as exc:
            results[method] = {'status': 'FAILED', 'error': repr(exc), 'predictions': len(runner.trace)}
    result = {'status': 'DIAGNOSTIC_ONLY', 'protected_data_already_exposed': True, 'math_cases': len(checks), 'math_max_abs_error': max(checks), 'results': results, 'warning': 'No change to frozen v3. No efficacy claim; conditioning and full replay qualification remain required.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    run(Path(sys.argv[1]))
