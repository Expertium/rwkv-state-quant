#!/usr/bin/env bash
# Deploy eval of the WKV-int4 + shift-int3 QAT weights (448 b/card) on FULL VAL, vs CHAMPION fp32.
# q448base = uncompressed (base_drift); q448 = WKV int4 rank-1 + shift int3 (base+compression = gate penalty).
# Args: $1=NPROC(14) $2=weights $3=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_lr1i4_s3.safetensors}; UF=${3:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "448b val eval: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
# base pass: no compression (base_drift vs champion fp32)
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="" RWKV_QUANT_SHIFTS=0 RWKV_LOWRANK_PERCOL=0 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q448base_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done
wait
# 448 b deploy: WKV int4 rank-1 (256 b) + token-shifts int3 (192 b)
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_STATE_SHIFT_LEVEL=int3 \
      RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q448_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done
wait
echo "=== VAL SCORE (vs champion fp32). q448base=base_drift, q448=total (WKV int4 + shift int3, 448 b) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q448base q448
echo "VAL448_DONE"
