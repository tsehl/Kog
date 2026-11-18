#!/usr/bin/env bash
# Full regression against the oracle, then obs_check, vmap_check and sb_check (test bank, then procedural bank 2).
# Usage: engine/regress.sh [log_dir]; JOBS=n parallel JAX processes (default: 3).
# Summary at the end: one line "k/n identical games" per mode.
set -uo pipefail
status=0
cd "$(dirname "$0")"
out=${1:-/tmp/ygo-regress}
mkdir -p "$out"
jobs=${JOBS:-3}  # ~2.6 GB per compare.py at E = 4; lower JOBS on a constrained host
run=${YGO_PYTHON:-"uv run --quiet --with jax python"}

# sb_check first, alone, on the test bank then on procedural bank 2 (Fusions, ATK filters): its second part peaks
# at ~9-10 GB at E = 4, nothing else fits next to it in 15 GB.
$run sb_check.py > "$out"/sb.log 2>&1 || status=1
SB_PROC=2 $run sb_check.py > "$out"/sbp.log 2>&1 || status=1
# The other heavy checks also run alone: vmap (~6.6 GB) next to two compare.py processes exceeds 8 GB.
$run vmap_check.py > "$out"/vmap.log 2>&1 || status=1
$run obs_check.py > "$out"/obs.log 2>&1 || status=1
# Only the oracle modes then share the compilation queue.
printf '%s\n' "trig 100 1000" "quick 80 1600" "cont 100 3000" "dmg 100 4000" "ds 100 5000" "flag 100 6000" "neg 100 7000" "fus 100 8000" "rit 100 9000" "syn 100 10000" "xyz 100 11000" "lnk 100 12000" "xlk 100 13000" "kn 100 14000" "pen 100 15000" "chain 60 1200" "mixed 60 1300" "normal 40 1" \
  | xargs -P "$jobs" -L 1 sh -c '
      '"$run"' compare.py "$1" "$2" "$0" > '"$out"'/"$0".log 2>&1' || status=1

for m in trig quick cont dmg ds flag neg fus rit syn xyz lnk xlk kn pen chain mixed normal; do
  printf '%-7s %s\n' "$m" "$(grep -h 'identical games' "$out/$m.log" || echo 'FAILED (see the log)')"
done
printf '%-7s %s\n' obs "$(tail -n 1 "$out/obs.log")"
printf '%-7s %s\n' vmap "$(tail -n 1 "$out/vmap.log")"
printf '%-7s %s\n' sb "$(tail -n 1 "$out/sb.log")"
printf '%-7s %s\n' sbp "$(tail -n 1 "$out/sbp.log")"

exit "$status"
