#!/usr/bin/env bash
# F24-fix deploy-eval: TRUE WD=0 PQ+QAT (raw + EMA) on full VAL vs CHAMPION fp32. 4 passes.
#   wdf_base/wdf_pq = raw ; wdfe_base/wdfe_pq = EMA.  PQ = rank-1 m2b8 + shift int4 (~352 b).
# Compare vs F22 (base_drift +0.0037/+0.0044, total +0.0043/+0.0037). Gate +0.0025 both. Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "wd0fix val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
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
pass reference/qat_pq_wd0fix.safetensors     ""    ""     0 wdf_base
pass reference/qat_pq_wd0fix.safetensors     "$LR" "$PQE" 1 wdf_pq
pass reference/qat_pq_wd0fix_ema.safetensors ""    ""     0 wdfe_base
pass reference/qat_pq_wd0fix_ema.safetensors "$LR" "$PQE" 1 wdfe_pq
echo "=== VAL SCORE (vs champion fp32). TRUE WD=0. F22 ref: base +0.0037/+0.0044, total +0.0043/+0.0037 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 wdf_base wdf_pq wdfe_base wdfe_pq
echo "WD0FIXEVAL_DONE"
