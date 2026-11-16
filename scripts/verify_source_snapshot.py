"""Verify the published source layout without loading data, models or project modules."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.flags.optimize:
        raise RuntimeError('Unset PYTHONOPTIMIZE and omit -O/-OO; source checks must not be optimized away.')
    if sys.version_info < (3, 11):
        raise RuntimeError('Activate a Python 3.11 or newer environment before source verification.')
    manifest = json.loads((ROOT / 'SOURCE_SNAPSHOT.json').read_text())
    archive = json.loads((ROOT / 'SOURCE_ARCHIVE.json').read_text())
    rows = manifest['source_files'] + manifest['generated_files']
    assert rows and len(archive['files']) == 414, 'empty or incomplete source manifest'
    links = 0
    python_files = 0
    for row in rows:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts, row['path']
        path = ROOT / relative
        if row['kind'] == 'symlink':
            assert path.is_symlink(), f"Checkout must preserve the Git symlink: {row['path']}"
            data = os.readlink(path).encode()
            assert data.decode() == row['link_target'], row['path']
            target = path.resolve(strict=True)
            assert target.is_relative_to(ROOT) and not (ROOT / row['target_path']).is_symlink(), row['path']
            assert target == ROOT / row['target_path'], row['path']
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row['resolved_sha256'], row['path']
            links += 1
        else:
            assert path.is_file() and not path.is_symlink(), row['path']
            data = path.read_bytes()
        assert len(data) == row['bytes'], row['path']
        assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=row['path'])
            python_files += 1
    for row in archive['files']:
        path = ROOT / row['source']
        target = ROOT / row['target']
        assert path.is_symlink() and os.readlink(path) == row['relative_symlink'], row['source']
        assert path.resolve() == target and target.is_file() and not target.is_symlink(), row['source']
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256'], row['source']
    assert links == len(archive['files']), 'archive/link coverage differs'
    print(json.dumps({
        'status': 'pass', 'manifest_entries': len(rows), 'archive_links': links,
        'Python_AST_checks': python_files, 'model_or_data_loaded': False,
        'failure_signal': 'non-zero exit or status != pass or archive_links != 414',
    }))


if __name__ == '__main__':
    main()
