#!/usr/bin/env bash
# DE-RISK (F21): PTQ rank-1 PQ on the F15 rank-1-int4 QAT weights (NO retrain), full VAL, vs CHAMPION fp32.
# All prior PQ rows in the table are RANK-2 (which blows up via the re-SVD runaway -> +0.1778). Rank-1 is
# STABLE (power-iter, no comp2) so rank-1 PQ has never been measured. This establishes the STARTING penalty
# that a future PQ+QAT must improve from (cf int4 rank-1: PTQ +0.0036 -> QAT +0.0024).
#   qi4r1 = F15 rank-1 int4, shifts int4         -> 512 b/card (F15 sanity, should be +0.0024/+0.0021)
#   pqr1  = rank-1 PQ WKV (m2b8) + shifts int4    -> ~96 (WKV) + 256 (shift) ~= 352 b/card
#   pqr1_6= rank-1 PQ WKV (m2b6) + shifts int4    -> ~80 (WKV) + 256 (shift) ~= 336 b/card
# Args: $1=NPROC(14) $2=weights(F15) $3=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_lr1i4.safetensors}; UF=${3:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds
PQ8=scratchpad/pq_cb_m2b8.txt; PQ6=scratchpad/pq_cb_m2b6.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "pq-r1-PTQ val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
pass() { # $1 extra-env  $2 tag  $3 desc
  echo "  pass $2 ($3)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 \
        RWKV_LOWRANK_PERCOL=1 $1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass ""                          qi4r1  "F15 rank-1 int4, shifts int4 (512 b sanity)"
pass "RWKV_LOWRANK_PQ=$PQ8"      pqr1   "rank-1 PQ m2b8 WKV + shifts int4 (~352 b)"
pass "RWKV_LOWRANK_PQ=$PQ6"      pqr1_6 "rank-1 PQ m2b6 WKV + shifts int4 (~336 b)"
echo "=== VAL SCORE (vs champion fp32). qi4r1=512b(F15 sanity), pqr1=~352b, pqr1_6=~336b ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 qi4r1 pqr1 pqr1_6
echo "PQR1PTQ_DONE"
