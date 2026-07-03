#!/usr/bin/env bash
# 288-b PTQ diagnostic (new objective: <=288 b at the +0.0025 gate): champion e150 weights deployed with
# int3 token-shifts (no shift-QAT). Card = m2b8 PQ 96 b + int3 shifts 192 b = 288 b. 1 pass + score.
# Ref: e150_pq (int4 shifts, 352 b) = +0.0010/-0.0003; int3 PTQ tax at 0.1 ep was +0.0007/+0.0004 (F18).
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
W=reference/qat_pq_ep150.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "e150s3 val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQ8 RWKV_STATE_SHIFT_LEVEL=int3 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_e150s3_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE e150+int3shifts @288 b (vs champion fp32). e150_pq@352 was +0.0010/-0.0003. Gate +0.0025 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 e150s3
echo "E150S3_DONE"
