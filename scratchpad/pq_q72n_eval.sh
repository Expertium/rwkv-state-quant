#!/usr/bin/env bash
# 76-b and 72-b probes, PURE PTQ: the q88N winner deployed with int1 norms (2 levels, 76 b, tag q76n1p)
# and FIXED midpoint norms (0 stored bits, 72 b, tag q72n0p; engine RWKV_PQ_NORM_BITS=0 = new mode).
# int5==int4==int3==int2 all measured FREE - riding the lever to its end. Gate +0.0025 both.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q88N.safetensors
PQ3=scratchpad/pq_cb_m2b3.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q76n1p + q72n0p PTQ eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 norm-bits  $2 tag
  echo "  pass $2 (NORM_BITS=$1)"
  for u in $USERS; do
    [ -f "$PRED/rust_pred_${2}_${u}.json" ] && continue
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
        RWKV_LOWRANK_PQ=$PQ3 RWKV_SHIFT_PQ=reference/pq_cb_shift_q88N.txt \
        RWKV_PQ_NORM_BITS=$1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass 1 q76n1p
pass 0 q72n0p
echo "=== VAL SCORE norm-lever endgame: q76n1p (int1, 76 b) + q72n0p (FIXED norms, 72 b). Gate +0.0025. int2 was +0.0023/+0.0006 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q76n1p q72n0p
echo "Q72NEVAL_DONE"
