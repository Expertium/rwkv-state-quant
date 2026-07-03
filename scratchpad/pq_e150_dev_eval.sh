#!/usr/bin/env bash
# Dev-confirmation of the FINAL champion: e150_pq over the 400 DEV users vs champion fp32. ep=1.5 was
# selected on VAL (methodology caveat) — this checks the pick isn't a val fluke. 1 pass + score.
# VAL was +0.0010 imm / -0.0003 ahead. Args: $1=NPROC.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
W=reference/qat_pq_ep150.safetensors
tr -d '\r' < dev_users.txt > scratchpad/dev_users_clean.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(cat scratchpad/dev_users_clean.txt)
echo "e150 DEV-confirm: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQ8 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_e150dev_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== DEV SCORE (vs champion fp32). VAL was +0.0010/-0.0003. Gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py scratchpad/dev_users_clean.txt fp32 e150dev
echo "E150DEV_DONE"
