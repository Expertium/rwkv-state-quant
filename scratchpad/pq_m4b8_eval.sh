#!/usr/bin/env bash
# F26 deploy-eval: m4b8 finer-codebook PQ+QAT on full VAL vs CHAMPION fp32.
#   m4_base = uncompressed (base_drift)   m4_pq = rank-1 PQ m4b8 WKV + shift int4 (~416 b/card)
# Compare vs F22 m2b8 (+0.0043/+0.0037; base_drift +0.0037/+0.0044, comp ~free). Gate +0.0025 both.
# Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ4=scratchpad/pq_cb_m4b8.txt
W=reference/qat_pq_m4b8.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "pq-m4b8 val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
pass() { # $1 lowrank-scope  $2 extra-env  $3 shifts  $4 tag
  echo "  pass $4"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$3" RWKV_LOWRANK_PERCOL=1 $2 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${4}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass ""                        ""                            0 m4_base
pass "card:1:int4,note:1:int4" "RWKV_LOWRANK_PQ=$PQ4"        1 m4_pq
echo "=== VAL SCORE (vs champion fp32). m4_base=base_drift; m4_pq=total(~416b); gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 m4_base m4_pq
echo "PQM4EVAL_DONE"
