#!/usr/bin/env bash
# Benchmark eval throughput at different NPROC (real conditions: FSRS may be running). One warmup pass
# (so OS file-cache is warm and doesn't bias), then time the uncompressed pass over a fixed user set at
# NPROC 10, 16, 24. Reports wall seconds (lower = better).
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
W=reference/qat_tail.safetensors
REF=reference_big; PRED=preds
COUNT=${1:-60}
tr -d '\r' < val_users.txt | head -n "$COUNT" > scratchpad/bench_users.txt
USERS=$(cat scratchpad/bench_users.txt)
run() { # $1 = nproc
  local N=$1 start end dur
  start=$(date +%s%3N)
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="" RWKV_QUANT_SHIFTS=0 RWKV_LOWRANK_PERCOL=0 "$BIN" "$u" >/dev/null 2>&1 ) &
    while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n 2>/dev/null || true; done
  done
  wait
  end=$(date +%s%3N); dur=$((end-start))
  awk -v n="$N" -v c="$COUNT" -v d="$dur" 'BEGIN{printf "NPROC=%-3s users=%s  wall=%.1fs  (%.2f users/s)\n", n, c, d/1000, c/(d/1000)}'
}
echo "warmup (N=16)..."; run 16 >/dev/null
echo "=== timed ==="
run 10
run 16
run 24
echo "BENCH_DONE"
