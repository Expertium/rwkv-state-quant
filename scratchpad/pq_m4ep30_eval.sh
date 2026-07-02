#!/usr/bin/env bash
# F27 deploy-eval: the m4b8 x 0.3ep COMBO (raw + EMA) on full VAL vs CHAMPION fp32. 4 passes.
#   c30_base/c30_pq = raw ; c30e_base/c30e_pq = EMA.  PQ = rank-1 m4b8 + shift int4 (~416 b).
# Singles: m4b8@0.1ep +0.0032/+0.0029; m2b8@0.3ep +0.0031/+0.0020. Combo should stack. Gate +0.0025 both.
# Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ4=scratchpad/pq_cb_m4b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "m4ep30 combo val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag
  echo "  pass $5 (W=$1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
LR="card:1:int4,note:1:int4"; PQE="RWKV_LOWRANK_PQ=$PQ4"
pass reference/qat_pq_m4ep30.safetensors     ""    ""     0 c30_base
pass reference/qat_pq_m4ep30.safetensors     "$LR" "$PQE" 1 c30_pq
pass reference/qat_pq_m4ep30_ema.safetensors ""    ""     0 c30e_base
pass reference/qat_pq_m4ep30_ema.safetensors "$LR" "$PQE" 1 c30e_pq
echo "=== VAL SCORE (vs champion fp32). COMBO m4b8 x 0.3ep, ~416 b. Gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 c30_base c30_pq c30e_base c30e_pq
echo "M4EP30EVAL_DONE"
