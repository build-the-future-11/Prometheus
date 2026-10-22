# Paper package

`main.tex` is an evidence-bound execution record. It deliberately contains no performance metric because no retained run completed all official folds under the frozen no-pageout guard.

Compile with the Codex LaTeX editor or:

```bash
tectonic main.tex
```

Before replacing the status language with numerical claims, run the release gate from the repository root:

```bash
.venv/bin/python scripts/reproduce_submission.py --run-id <completed-run-id>
```
