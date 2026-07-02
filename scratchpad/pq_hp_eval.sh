#!/usr/bin/env bash
# F24 deploy-eval: the 4 HP-sweep PQ+QAT candidates on full VAL vs CHAMPION fp32. For each weight:
# uncompressed (base_drift) + rank-1 PQ m2b8 WKV + shift int4 (~352 b). Gate +0.0025 both.
#   wd0 = WD=0 raw ; wd0e = WD=0 EMA ; cl01 = clip0.1 raw ; cl01e = clip0.1 EMA
# Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "pq-HP val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag
  echo "  pass $5 (W=$1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
LR="card:1:int4,note:1:int4"; PQE="RWKV_LOWRANK_PQ=$PQ8"
for tag in wd0 wd0_ema cl01 cl01_ema; do
  W=reference/qat_pq_${tag}.safetensors
  pass "$W" ""    ""     0 ${tag}_base
  pass "$W" "$LR" "$PQE" 1 ${tag}_pq
done
echo "=== VAL SCORE (vs champion fp32). *_base=base_drift; *_pq=total(~352b); gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 \
  wd0_base wd0_pq wd0_ema_base wd0_ema_pq cl01_base cl01_pq cl01_ema_base cl01_ema_pq
echo "PQHPEVAL_DONE"
