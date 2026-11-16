"""Move reviewed source files with reversible legacy links. No source is executed."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/chenzhe/work/lunar-pathloss-rm-challenge/CC')
HERE = ROOT / 'audit/cc_cleanup_20261117'
ARCHIVE = ROOT / 'archive/20261117_cc_cleanup'


def read_json(name):
    return json.loads((HERE / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def entries():
    plan = read_json('archive_plan.json')
    assert plan['moves'], 'empty archive plan'
    seen = set()
    for row in plan['moves']:
        relative = Path(row['source'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in {'src', 'scripts', 'runs', 'releases'}
        assert relative.suffix in {'.py', '.sh'}
        assert str(relative) not in seen
        seen.add(str(relative))
        target = ARCHIVE / relative
        assert row['target'] == str(target.relative_to(ROOT))
        yield row, ROOT / relative, target


def assert_original(row, path):
    assert identity(path) == row['identity'], f'changed identity: {path}'
    assert digest(path) == row['sha256'], f'changed bytes: {path}'


def undo():
    restored = 0
    for row, original, target in reversed(list(entries())):
        if not target.exists():
            assert original.exists() and not original.is_symlink(), str(original)
            assert_original(row, original)
            continue
        assert_original(row, target)
        if original.is_symlink():
            assert original.resolve() == target.resolve(), str(original)
            original.unlink()
        else:
            assert not original.exists(), f'refusing overwrite: {original}'
        target.rename(original)
        assert_original(row, original)
        restored += 1
    print(json.dumps({'restored': restored, 'records_modified': 0}))


def apply():
    rows = list(entries())
    for row, original, target in rows:
        assert original.is_file() and not original.is_symlink(), str(original)
        assert not target.exists() and not target.is_symlink(), str(target)
        assert_original(row, original)
    try:
        for row, original, target in rows:
            target.parent.mkdir(parents=True, exist_ok=True)
            assert target.parent.stat().st_dev == original.stat().st_dev
            original.rename(target)
            original.symlink_to(os.path.relpath(target, original.parent))
            assert_original(row, original)
            assert_original(row, target)
        receipt = verify()
    except BaseException:
        undo()
        raise
    (HERE / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt['counts'], ensure_ascii=False))


def verify():
    before = read_json('inventory_before.json')
    plan = read_json('archive_plan.json')
    moves = {row['source']: (row, original, target) for row, original, target in entries()}
    failures = []
    for relative, record in before['files'].items():
        path = ROOT / relative
        if relative == 'HANDOFF.md':
            continue
        if not path.exists():
            failures.append(['missing', relative])
            continue
        if identity(path) != record['identity']:
            failures.append(['identity_changed', relative])
        link = os.readlink(path) if path.is_symlink() else None
        if relative in moves:
            row, original, target = moves[relative]
            if not path.is_symlink() or path.resolve() != target.resolve():
                failures.append(['legacy_link_invalid', relative])
            if not target.is_file() or target.is_symlink():
                failures.append(['archive_target_invalid', relative])
            elif identity(target) != record['identity'] or digest(target) != row['sha256']:
                failures.append(['archive_target_changed', relative])
        elif link != record['symlink']:
            failures.append(['file_type_or_link_changed', relative])
    for relative, record in before['sources'].items():
        if relative != 'HANDOFF.md' and digest(ROOT / relative) != record['sha256']:
            failures.append(['source_bytes_changed', relative])
    for relative, value in before['previous_protected_artifacts'].items():
        if identity(ROOT / relative) != value:
            failures.append(['protected_artifact_changed', relative])
    for relative in plan['keep_regular']:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            failures.append(['keep_not_regular', relative])
    for relative in plan['keep_directories']:
        path = ROOT / relative
        if not path.is_dir() or path.is_symlink():
            failures.append(['keep_directory_invalid', relative])
    protocol = json.loads((ROOT / 'audit/c157_pstack_review/protocol.json').read_text())
    manifest = json.loads((ROOT / protocol['publication_manifest']).read_text())
    aliases = {row['path']: row.get('source_original_path', row['path'])
               for row in manifest['source_files']}
    checked_bindings = 0
    for relative, record in protocol['source_bindings'].items():
        if relative == 'HANDOFF.md':
            continue
        path = ROOT / aliases.get(relative, relative)
        if digest(path) != record['sha256'] or identity(path) != record['identity']:
            failures.append(['frozen_binding_changed', relative])
        checked_bindings += 1
    for row, original, target in moves.values():
        if digest(original) != digest(target):
            failures.append(['old_and_archive_bytes_differ', row['source']])
    receipt = {
        'status': 'pass' if not failures else 'fail',
        'counts': {
            'baseline_files_checked': len(before['files']) - 1,
            'original_sources_sha256_checked': len(before['sources']) - 1,
            'frozen_source_bindings_checked': checked_bindings,
            'previous_protected_artifacts_checked': len(before['previous_protected_artifacts']),
            'archived_sources': len(moves),
            'legacy_links_checked': len(moves),
            'retained_regular_paths_checked': len(plan['keep_regular']),
            'protected_directories_checked': len(plan['keep_directories']),
            'failures': len(failures),
        },
        'failures': failures,
        'changed_original_document': 'HANDOFF.md',
        'weights_predictions_records_preserved': not failures,
        'source_execution': False,
        'training_or_neural_inference': False,
        'GPU_used': False,
        'checks': [
            'Original followed file identity includes device, inode, size, mtime_ns.',
            'Every original source SHA256 is unchanged, excluding authorized HANDOFF.md.',
            'Every legacy link resolves to its reviewed archive target.',
            'Active and path-sensitive keep paths remain regular files.',
            'C157 frozen source bindings still resolve with their original SHA256 and identity.',
        ],
        'failure_signal': 'non-zero exit or status != pass or failures > 0',
    }
    assert not failures, json.dumps(receipt, ensure_ascii=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'verify', 'restore'])
    args = parser.parse_args()
    if args.action == 'apply':
        apply()
    elif args.action == 'restore':
        undo()
    else:
        receipt = verify()
        (HERE / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(receipt['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
