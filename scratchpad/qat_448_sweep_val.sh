#!/usr/bin/env bash
# Eval both lower-LR 448 b shift-QAT variants (WKV int4 rank-1 + shift int3) on FULL VAL vs CHAMPION fp32.
# q448_5e4 = lr5e4 weights, q448_2e4 = lr2e4 weights. Pick the one that clears +0.0025 imm (ahead already ok).
# Args: $1=NPROC(14) $2=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "448b LR-sweep val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 tag
  echo "  pass $2 ($1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS="$1" RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_STATE_SHIFT_LEVEL=int3 \
        RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass reference/qat_lr1i4_s3_lr5e4.safetensors q448_5e4
pass reference/qat_lr1i4_s3_lr2e4.safetensors q448_2e4
echo "=== VAL SCORE (vs champion fp32). 448 b = WKV int4 + shift int3. WIN = <=+0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q448_5e4 q448_2e4
echo "SWEEP448_DONE"
