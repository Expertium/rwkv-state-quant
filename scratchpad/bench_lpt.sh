#!/usr/bin/env bash
# Compare dispatch makespan: OLD (barrier every NPROC, id-order) vs NEW (LPT order + work-queue).
set -e
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=reference/champ_h2k16.safetensors \
       RWKV_TRACE_DIR=reference_big RWKV_PRED_DIR=preds
BIN=./engine/target/release/rwkv-infer.exe
NPROC=10
REF=reference_big

# 80 dev users in id order (natural size variance incl. the big early ones)
IDORDER=$(head -80 dev_users.txt)
# same 80, LPT order (largest trace first)
LPT=$(for u in $IDORDER; do printf '%s %s\n' "$(stat -c%s "$REF/trace_user_${u}.safetensors")" "$u"; done \
        | sort -rn | awk '{print $2}')

run1() { ( "$BIN" "$1" >/dev/null 2>&1 ) & }

old_dispatch() { # barrier every NPROC, id order
  local i=0
  for u in $IDORDER; do
    run1 "$u"; i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}
new_dispatch() { # LPT order + work-queue (refill on any finish)
  for u in $LPT; do
    run1 "$u"
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
timed() { local t0 t1; t0=$(date +%s.%N); "$1"; t1=$(date +%s.%N); awk "BEGIN{printf \"%.1f\", $t1-$t0}"; }

echo "users=$(echo $IDORDER | wc -w)  NPROC=$NPROC"
echo "size spread (MB): $(for u in $IDORDER; do stat -c%s "$REF/trace_user_${u}.safetensors"; done | sort -n | awk 'NR==1{min=$1} {max=$1; s+=$1; n++} END{printf "min %.1f  mean %.1f  max %.1f", min/1e6, s/n/1e6, max/1e6}')"
echo "warming..."; new_dispatch >/dev/null
for r in 1 2; do
  o=$(timed old_dispatch); n=$(timed new_dispatch)
  awk "BEGIN{printf \"round %d:  OLD(barrier,id) %ss   NEW(LPT,queue) %ss   speedup %.2fx\n\", $r, $o, $n, $o/$n}"
done
echo "BENCH_LPT_DONE"
