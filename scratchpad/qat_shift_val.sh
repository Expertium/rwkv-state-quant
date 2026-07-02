#!/usr/bin/env bash
# PTQ test of asymmetric shift compression on the F15 rank-1 int4 QAT weights (NO retrain), full VAL,
# vs CHAMPION fp32. WKV stays int4 (256 b, F15-proven ~free); only the token-shifts get coarser:
#   qi4s4 = shifts int4 -> 512 b/card (== F15 sanity)   qi4s3 = shifts int3 -> 448 b   qi4s2 = shifts int2 -> 384 b
# Args: $1=NPROC(14) $2=weights(F15) $3=users-file.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_lr1i4.safetensors}; UF=${3:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "shift-PTQ val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
pass() { # $1 shift-level  $2 tag
  echo "  pass $2 (WKV int4, shifts $1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_STATE_SHIFT_LEVEL="$1" \
        RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass int4 qi4s4
pass int3 qi4s3
pass int2 qi4s2
echo "=== VAL SCORE (vs champion fp32). qi4s4=512b(F15 sanity), qi4s3=448b, qi4s2=384b ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 qi4s4 qi4s3 qi4s2
echo "SHIFTVAL_DONE"
