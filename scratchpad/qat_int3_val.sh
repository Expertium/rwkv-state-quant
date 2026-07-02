#!/usr/bin/env bash
# Deploy eval of the rank-1 int3 QAT weights (384 b/card) on FULL VAL, vs the CHAMPION fp32 baseline.
# Emits: qi3base = qfp32 (uncompressed QAT weights -> base_drift) and qi3r1 = int3 rank-1 deploy
# (base_drift + compression_cost = total gate penalty). Args: $1=NPROC(14) $2=weights $3=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_lr1i3.safetensors}; UF=${3:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "int3 val eval: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
# base pass: QAT weights, NO compression (measures base_drift vs champion fp32)
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="" RWKV_QUANT_SHIFTS=0 RWKV_LOWRANK_PERCOL=0 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_qi3base_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done
wait
# int3 rank-1 deploy pass (card/note 384 b, shifts follow at int3)
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int3,note:1:int3" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_qi3r1_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done
wait
echo "=== VAL SCORE (vs champion fp32). qi3base=base_drift, qi3r1=total (base+compression) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 qi3base qi3r1
echo "INT3VAL_DONE"
