#!/usr/bin/env bash
# F27b deploy-eval: m4b8 x 0.5ep (raw only — EMA has been ±0.0001 in 4 paired evals). 2 passes.
# Question: does m4b8 scale better with epochs than m2b8? (m4b8: 0.1ep +0.0032, 0.3ep +0.0024, 0.5ep -> ?)
# If <= ~+0.0018 the m4b8+int3shift@352b lever revives; else the m2b8 epoch line stands alone.
# Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ4=scratchpad/pq_cb_m4b8.txt
W=reference/qat_pq_m4ep50.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "m4ep50 val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
pass() { # $1 lowrank-scope  $2 extra-env  $3 shifts  $4 tag
  echo "  pass $4"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$3" RWKV_LOWRANK_PERCOL=1 $2 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${4}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass ""                        ""                     0 m450_base
pass "card:1:int4,note:1:int4" "RWKV_LOWRANK_PQ=$PQ4" 1 m450_pq
echo "=== VAL SCORE (vs champion fp32). m4b8 x 0.5ep, ~416 b. m4b8@0.3ep was +0.0024/+0.0013 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 m450_base m450_pq
echo "M4EP50EVAL_DONE"
