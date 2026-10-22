# Experimental results

## Registered execution

The private Kaggle development-v2 kernel completed on CPU with GPU and internet
disabled. Before evaluation, it verified 23 source files against the uploaded
manifest and passed 25 tests. Ten registered PJM series contributed 366 evaluation
days each, yielding 3,660 raw prediction rows. Dates from 2017 onward were not parsed.

## Mean MASE

| Rank | Method | Equal-series mean MASE |
|---:|---|---:|
| 1 | AdaHedge | 0.6144090818 |
| 2 | Hedge | 0.6144647650 |
| 3 | Fixed share | 0.6214673034 |
| 4 | Directed fixed share | 0.6219158835 |
| 5 | Adaptive share | 0.6236388805 |
| 6 | DDFS | 0.6246624969 |
| 7 | Uniform | 0.6628933273 |
| 8 | Seasonal naive | 1.0265396398 |

## Registered contrasts

Positive differences mean the left-hand method is worse. All intervals are the
retained synchronized seven-day moving-block bootstrap intervals from 5,000 draws.

| Contrast | Estimate | 95% interval | Left wins |
|---|---:|---:|---:|
| DDFS - fixed share | +0.0031951934 | [+0.0014690581, +0.0049077382] | 0/10 |
| DDFS - adaptive share | +0.0010236164 | [+0.0004661807, +0.0015807069] | 0/10 |
| DDFS - directed fixed share | +0.0027466134 | [+0.0012472227, +0.0042401278] | 0/10 |
| Directed fixed share - fixed share | +0.0004485800 | [+0.0002079598, +0.0006878615] | 0/10 |
| Adaptive share - fixed share | +0.0021715770 | [+0.0009805327, +0.0033602219] | 0/10 |
| DDFS - AdaHedge | +0.0102534151 | [+0.0055371592, +0.0151760836] | 0/10 |

The primary relative improvement was -0.5141%. Both registered advancement gates
failed. No protected confirmation was run.

## Claim boundary

The results contradict revision 2 on this development domain. They do not estimate a
confirmatory population effect, cross-domain generalization, compute efficiency, or
SOTA performance.
