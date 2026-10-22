"""Build inspectable release manifests and source/research archives without execution."""
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {'RELEASE_MANIFEST.json', 'release/SHA256SUMS',
             'release/PI-ML-research-artifact.tar.gz', 'release/PI-ML-arxiv-source.zip'}


def files():
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in {'__pycache__', '.git', '.pytest_cache', '.latex-cache',
                        '.venv', 'build', 'dist', 'hosted-downloads'} or part.endswith('.egg-info')
               for part in rel.parts):
            continue
        if rel.parts[0] in {'delivery', 'runs'} or p.name == '.DS_Store' or p.suffix in {'.pyc', '.pyo'}:
            continue
        if str(rel) in GENERATED or str(rel) in {'paper/main.pdf', 'paper/main.aux', 'paper/main.log', 'paper/main.out', 'paper/main.bbl', 'paper/main.blg'}:
            continue
        yield p


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    release = ROOT / 'release'; release.mkdir(exist_ok=True)
    chosen = list(files())
    manifest = {'scope': 'Complete curated research snapshot; immutable evidence retained. Generated archive checksums are separate to avoid self-reference.',
                'excluded_generated_outputs': sorted(GENERATED),
                'files': {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in chosen}}
    (ROOT / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(release / 'PI-ML-arxiv-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in chosen:
            rel = p.relative_to(ROOT)
            if rel.parts[0] == 'paper' and (p.suffix in {'.tex', '.sty'} or (p.parent.name == 'figures' and p.suffix == '.pdf')):
                z.write(p, str(p.relative_to(ROOT / 'paper')))
    with tarfile.open(release / 'PI-ML-research-artifact.tar.gz', 'w:gz') as t:
        for p in chosen + [ROOT / 'RELEASE_MANIFEST.json']:
            t.add(p, arcname=str(Path('PI-ML') / p.relative_to(ROOT)), recursive=False)
    outputs = [release / 'PI-ML-arxiv-source.zip', release / 'PI-ML-research-artifact.tar.gz', release / 'PI-ML-study.pdf']
    outputs += sorted((release / 'packages').glob('*'))
    (release / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in outputs))
    print(json.dumps({'files': len(chosen), 'bytes': sum(p.stat().st_size for p in chosen),
                      'outputs': [str(p.relative_to(ROOT)) for p in outputs]}, indent=2))
