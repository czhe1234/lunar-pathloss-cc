#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
card=${1:?Use physical card 1, 3 or 4}
shift
case "$card" in
  1) device=GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 ;;
  3) device=GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0 ;;
  4) device=GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 ;;
  *) exit 2 ;;
esac
shared=0
for argument in "$@"; do
  if [[ "$argument" == "--allow-shared-gpu" ]]; then shared=1; fi
done
if (( shared )) && [[ "$card" != 3 ]]; then
  echo "Shared C145 execution is authorized only on physical card 3." >&2
  exit 2
fi
if (( ! shared )); then
for check in 1 2 3; do
  jobs=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader,nounits)
  stats=$(nvidia-smi -i "$device" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r used utilization <<< "${stats/,/ }"
  if [[ -n "$jobs" || ! "$used" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]]; then
    echo "Card $card is occupied or its idle state cannot be verified." >&2
    exit 75
  fi
  if (( used > 256 || utilization != 0 )); then
    echo "Card $card is occupied." >&2
    exit 75
  fi
  if (( check < 3 )); then sleep 1; fi
done
fi
exec ./run_gpu.sh "$card" src/c145_train.py "$@"
