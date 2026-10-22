# Retained results

## Evidence status

The admitted development matrix `shared-baselines-v1-20261019` completed 20 of 20 serial CPU attempts across five fixed seeds. No attempt failed, no seed was discarded, no candidate mechanism ran, and protected evaluation remained at zero runs. The independent verifier checked 20 result files and 9,600 trace rows and returned `VALID` with artifact-map SHA-256 `f353206d36fd03ec596a2b4006ef181715c524c43ec13e53d07e2ec28dc089d6`.

These are development-only baseline findings. They do not validate either candidate hypothesis.

## Development aggregate

| Policy | Overall MSE, mean ± sample SD | Switch MSE | Return MSE | Mean wall seconds | Peak state bytes | Total updates |
|---|---:|---:|---:|---:|---:|---:|
| no adaptation | 1.0110 ± 0.0671 | 0.9611 | 1.2241 | 0.00237 | 72 | 0 |
| SGD, lr 0.01 | **0.9484 ± 0.0627** | 1.9652 | 2.2782 | 0.00434 | 72 | 2,380 |
| SGD, lr 0.05 | 3.3480 ± 0.2161 | 6.7383 | 12.2006 | 0.00384 | 72 | 2,380 |
| mechanism-off Adam, lr 0.01 | 0.9978 ± 0.0943 | 1.9225 | 1.9003 | 0.00593 | 224 | 2,380 |

Relative to no adaptation, paired mean MSE differences across the same five seeds were:

- SGD 0.01: −0.0626, sample SD 0.0356;
- Adam 0.01: −0.0132, sample SD 0.0578;
- SGD 0.05: +2.3369, sample SD 0.1835.

All policies produced zero non-finite predictions. Maximum absolute predictions were 0.0, 3.1154, 11.2766, and 2.6592 for frozen, SGD 0.01, SGD 0.05, and Adam respectively. Maximum update norms were 0.0, 0.1, 0.5, and 0.03.

## Interpretation

SGD 0.01 is the development-selected conventional baseline because it has the lowest overall prequential MSE. That selection should not be read as uniformly better adaptation: its switch MSE was about twice the frozen control and its returning-regime MSE was 2.2782 versus 1.2241. It fit stable stretches while paying a larger transient and forgetting cost.

Adam 0.01 had only a small overall MSE improvement over no adaptation, with between-seed variation larger than the paired mean difference, and it also worsened switch and return MSE. SGD 0.05 failed scientifically despite completing numerically: it degraded every seed, amplified predictions, and had severe switch/return error. This negative configuration remains retained.

The evidence therefore establishes a working harness and a demanding baseline tradeoff. It provides no evidence that a learned rule works and no evidence that a memory-weight policy works, because neither exists in protocol v1.

## Resource observation

The retained bundle occupies about 2.3 MiB. Reported algorithmic policy state peaked at 224 bytes and mean per-run wall time stayed below 0.006 seconds. Host `vm_stat` pageouts changed from 518,722 before the matrix to 518,766 after aggregation. That system-wide counter cannot be attributed to this process; no additional numerical experiment was initiated after the required matrix. Pages throttled remained zero.

## Evidence locations

- Run manifest: `results/development/shared-baselines-v1-20261019/run_manifest.json`
- Attempt receipt: `results/development/shared-baselines-v1-20261019/receipt.json`
- Independent verification: `results/development/shared-baselines-v1-20261019/verification.json`
- Aggregate JSON/CSV: `results/development/shared-baselines-v1-20261019/aggregate/`
- Per-seed results, traces, and checkpoints: `results/development/shared-baselines-v1-20261019/attempts/`

