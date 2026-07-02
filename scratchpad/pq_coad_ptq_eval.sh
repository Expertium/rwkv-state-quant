#!/usr/bin/env bash
# Co-adapt DIAGNOSTIC (cheap PTQ): ep75 weights deployed with the CO-ADAPTED codebook (no retraining).
# 1 pass. vs e75_pq (+0.0021/+0.0012, champion codebook). If consistency (train==deploy) dominates it
# hurts; if codebook-fit dominates it helps. Either way it bounds what the co-adapt re-QAT can gain.
# Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQC=scratchpad/pq_cb_m2b8_coad.txt
W=reference/qat_pq_ep75.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "coad-PTQ val: $(echo $USERS|wc -w) users, W=$W, cb=$PQC, NPROC=$NPROC"
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQC $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_e75coad_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE (vs champion fp32). e75+coad-codebook PTQ vs e75_pq (+0.0021/+0.0012) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 e75_pq e75coad
echo "COADPTQ_DONE"
