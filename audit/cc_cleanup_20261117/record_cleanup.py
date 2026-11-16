"""Record housekeeping or a linked correction in server-local SwanLab."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def read(name):
    return json.loads((OUT / name).read_text())


def identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correction', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    receipt = OUT / ('tracking_correction.json' if args.correction else 'tracking.json')
    assert not receipt.exists(), 'this record already exists'
    result = read('verification.json')
    assert result['status'] == 'pass' and result['counts']['archived_sources'] == 414
    before = read('tracking_before.json')
    assert before['original_rows']['experiment'] and before['original_rows']['tag']
    previous = read('tracking.json') if args.correction else None
    extra_rows = {}
    extra_files = {}
    if previous:
        assert previous['status'] == 'pass'
        with sqlite3.connect('file:' + before['database'] + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            previous_id = previous['native_experiment_id']
            extra_rows = {
                'experiment': [dict(row) for row in db.execute('select * from experiment where id=?', (previous_id,))],
                'tag': [dict(row) for row in db.execute('select * from tag where experiment_id=?', (previous_id,))],
            }
        assert extra_rows['experiment'] and extra_rows['tag']
        extra_files = {str(path): identity(path) for path in Path(previous['dir']).rglob('*') if path.is_file()}
        assert extra_files
    from src import cc_tracking
    import swanlab
    handoff_sha = hashlib.sha256((ROOT / 'HANDOFF.md').read_bytes()).hexdigest()
    run = cc_tracking.init(
        name='CC archive documentation correction 20261117' if args.correction else 'CC source archive 20261117',
        campaign='CC_MAINTENANCE',
        config={
            'run_kind': 'analysis.housekeeping', '_cc_dashboard': {'kind': 'analysis'},
            'source_plan': 'audit/cc_cleanup_20261117/archive_plan.json',
            'plan_sha256': hashlib.sha256((OUT / 'archive_plan.json').read_bytes()).hexdigest(),
            'counts': result['counts'], 'training': False, 'neural_inference': False,
            'GPU_used': False, 'weights_or_predictions_changed': False,
            'documentation_review': 'see audit/cc_cleanup_20261117/review.json',
            'HANDOFF_sha256': handoff_sha,
            'correction_of_run_id': previous['run_id'] if previous else None,
            'correction_of_native_experiment_id': previous['native_experiment_id'] if previous else None,
            'correction_evidence': 'audit/cc_cleanup_20261117/repair_delta.json' if previous else None,
        },
        description='保留原说明，通过关联记录更正成绩称谓与历史动作时间。' if args.correction else '依赖核对与历史源码归档，保留旧路径和全部既有科研记录。',
        tags=['housekeeping', 'archive', 'CPU', *(['correction'] if args.correction else [])],
    )
    tracking = {'run_id': run.id, 'dir': str(run.dir), 'mode': 'local', 'project': 'Lunar_CC'}
    receipt.write_text(json.dumps(tracking, ensure_ascii=False, indent=2) + '\n')
    report_tag = 'cleanup/correction' if args.correction else 'cleanup/result'
    prefix = ''
    if previous:
        prefix = (
            f"更正原说明 run {previous['run_id']} / native {previous['native_experiment_id']}。\n"
            "原文保留，以本说明为准。C145是主要冻结参照，不能称最低分。\n"
            "既有C148 selected16的monitor为2.138235/2.407435 dB，点估计更低；高频收益CI跨0。\n"
            "GitHub同步、C149统计补充和viewer重启均为历史动作，本轮未执行。\n\n"
        )
    try:
        section = (ROOT / 'HANDOFF.md').read_text().split('## 环境与有效约定')[0]
        cc_tracking.log({report_tag: swanlab.Text(prefix + section, caption='CC 归档说明更正' if previous else 'CC 源码归档与核验')}, step=1)
    finally:
        cc_tracking.finish()
    record = None
    for attempt in range(20):
        with sqlite3.connect('file:' + before['database'] + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('select * from experiment where run_id=?', (Path(tracking['dir']).name,)).fetchone()
            if row:
                record = dict(row)
                tags = [dict(value) for value in db.execute('select * from tag where experiment_id=?', (record['id'],))]
                if any(value['name'] == report_tag for value in tags):
                    break
        time.sleep(.5)
    assert record and any(value['name'] == report_tag for value in tags), 'report not ingested'
    with sqlite3.connect('file:' + before['database'] + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for table, rows in before['original_rows'].items():
            current = {value['id']: dict(value) for value in db.execute('select * from ' + table)}
            for old in [*rows, *extra_rows.get(table, [])]:
                assert current.get(old['id']) == old, (table, old['id'], 'original row changed')
    for relative, value in before['log_file_identities'].items():
        assert identity(ROOT / 'tracking/swanlab' / relative) == value, relative
    for path, value in extra_files.items():
        assert identity(Path(path)) == value, path
    tracking.update(
        native_experiment_id=record['id'],
        view_url=f"http://127.0.0.1:16007/experiment/{record['id']}/chart",
        report_tag=report_tag, report_tag_verified=True,
        existing_experiment_rows_unchanged=len(before['original_rows']['experiment']),
        existing_tag_rows_unchanged=len(before['original_rows']['tag']),
        existing_log_files_unchanged=len(before['log_file_identities']),
        corrected_HANDOFF_sha256=handoff_sha,
        correction_of_run_id=previous['run_id'] if previous else None,
        correction_of_native_experiment_id=previous['native_experiment_id'] if previous else None,
        preserved_prior_run_files=len(extra_files),
        preserved_prior_run_rows=sum(len(rows) for rows in extra_rows.values()),
        status='pass',
    )
    receipt.write_text(json.dumps(tracking, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(tracking, ensure_ascii=False))


if __name__ == '__main__':
    main()
