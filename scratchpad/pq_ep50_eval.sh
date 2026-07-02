#!/usr/bin/env bash
# F25b deploy-eval: 0.5-ep PQ+QAT (raw + EMA) on full VAL vs CHAMPION fp32. 4 passes.
#   e50_base/e50_pq = raw ; e50e_base/e50e_pq = EMA.  PQ = rank-1 m2b8 + shift int4 (~352 b).
# Trend ref: 0.3 ep total +0.0031/+0.0020 (ahead passes). Gate +0.0025 both. Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "ep50 val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
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
pass reference/qat_pq_ep50.safetensors     ""    ""     0 e50_base
pass reference/qat_pq_ep50.safetensors     "$LR" "$PQE" 1 e50_pq
pass reference/qat_pq_ep50_ema.safetensors ""    ""     0 e50e_base
pass reference/qat_pq_ep50_ema.safetensors "$LR" "$PQE" 1 e50e_pq
echo "=== VAL SCORE (vs champion fp32). 0.5 ep. Trend: 0.3ep total +0.0031/+0.0020. Gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 e50_base e50_pq e50e_base e50e_pq
echo "EP50EVAL_DONE"
