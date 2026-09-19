"""Export a source-only review ZIP without Git history or local artifacts."""
import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, default=ROOT / 'review_exports' / 'anonymous-code.zip')
p.add_argument('--redact', action='append', default=[], help='Additional identifying text to replace; repeat as needed.')
a = p.parse_args()
# Use the Git index to avoid accidentally including untracked data or checkpoints.
paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
excluded = {'SOURCE_MANIFEST.json', 'docs/environment-observed.txt', 'docs/REVIEW_ACCESS.md', 'scripts/export_review.py'}
allowed_roots = {'src', 'scripts', 'configs', 'docs', 'vendor'}
allowed_files = {'README.md', 'LICENSE', 'requirements.txt', '.gitignore'}
terms = sorted(set([ROOT.name, str(Path.home()), *a.redact]) - {''}, key=len, reverse=True)
files = []
for relative in sorted(filter(None, paths)):
    path = ROOT / relative
    parts = Path(relative).parts
    if relative in excluded or relative.startswith('docs/assets/') or (parts[0] not in allowed_roots and relative not in allowed_files):
        continue
    if '__pycache__' in parts or path.suffix in {'.pyc', '.pt', '.pth', '.safetensors', '.ckpt'}:
        continue
    if path.is_symlink():
        raise SystemExit(f'Refusing symlink: {relative}')
    data = path.read_bytes()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        # The CLIP vocabulary is the only required binary source asset.
        if relative != 'vendor/openai_clip/clip/bpe_simple_vocab_16e6.txt.gz':
            raise SystemExit(f'Unexpected binary file: {relative}')
    else:
        for term in terms:
            text = text.replace(term, 'AnonymousVirtualStaining' if term == ROOT.name else 'ANONYMIZED')
        if relative == 'README.md':
            # Exclude the public demo and its identifying repository links.
            start = text.find('## System demo\n')
            end = text.find('## Training workflow\n')
            if start != -1 and end > start:
                text = text[:start] + text[end:]
            text = text.replace('[Demo](#system-demo) · ', '')
            text = '\n'.join(line for line in text.splitlines() if 'SOURCE_MANIFEST.json' not in line and 'docs/REVIEW_ACCESS.md' not in line) + '\n'
        if relative == 'docs/IMPLEMENTATION_NOTES.md':
            text = text.split('## Source provenance')[0]
        data = text.encode('utf-8')
    files.append((relative, data))
if not any(name == 'README.md' for name, _ in files):
    raise SystemExit('No staged README; stage the reviewed source files before exporting.')
a.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(a.output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for relative, data in files:
        info = zipfile.ZipInfo('anonymous-code/' + relative, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        z.writestr(info, data)
digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
a.output.with_suffix('.zip.sha256').write_text(f'{digest}  {a.output.name}\n')
print(f'Created {a.output.name}: {len(files)} files, {a.output.stat().st_size} bytes')
print('Inspect the archive before sharing. It has not been uploaded anywhere.')
