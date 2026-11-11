#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
while kill -0 "$1" 2>/dev/null; do sleep 2; done
CC_COMPILE_FLAG=$(.venv/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('audit/compile_benchmark.json')
b=json.loads(Path('audit/throughput_benchmark.json').read_text())
eager=next(x['maps_per_second'] for x in b['results'] if x['batch']==32)
use=p.exists() and json.loads(p.read_text())['maps_per_second']>eager*1.10
print('--compile' if use else '')
PY
)
printf 'selected compile flag: %s\n' "$CC_COMPILE_FLAG"
exec ./run_gpu3.sh scripts/train.py --run b0_raw_s20261031 --mode raw --batch 32 --epochs 60 --patience 10 --seed 20261031 $CC_COMPILE_FLAG
