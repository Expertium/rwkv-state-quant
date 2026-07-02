#!/usr/bin/env bash
# Benchmark NPROC=10 vs 16 on the real eval pattern (N single-threaded user-processes in parallel).
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
W=reference/champ_h2k16.safetensors
REF=reference_big
# 40 small-moderate dev users (3-8 MB band -> fast, balanced waves, CPU-bound)
USERS=$(awk '$1>=6000 && $1<=6435 && $2>=3e6 && $2<=8e6' scratchpad/trace_sizes.txt | sort -k2 -n | head -40 | awk '{print $1}')
NU=$(echo $USERS | wc -w)

run_pass() {  # $1 = NPROC
  local NPROC=$1 i=0
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF $BIN $u >/dev/null 2>&1 ) &
    i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}
timed() {  # $1 = NPROC ; echoes seconds
  local t0 t1
  t0=$(date +%s.%N); run_pass "$1"; t1=$(date +%s.%N)
  awk "BEGIN{printf \"%.1f\", $t1-$t0}"
}

echo "users=$NU  (median-size dev band)  cores=16phys/32thr"
echo "warming page cache..."; run_pass 16 >/dev/null
for r in 1 2; do
  t10=$(timed 10); t16=$(timed 16)
  awk "BEGIN{printf \"round %d:  NPROC=10 -> %ss   NPROC=16 -> %ss   speedup(10/16)= %.2fx\n\", $r, $t10, $t16, $t10/$t16}"
done
echo "BENCH_DONE"
