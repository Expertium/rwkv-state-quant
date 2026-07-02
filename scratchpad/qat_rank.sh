#!/usr/bin/env bash
# FAST ranking eval for the tail sweep: qfp32 (base) + qi4r1 (rank-1 int4, 256b) over a fixed subset,
# scored vs the CHAMPION fp32 baseline. Args: $1=NPROC(14) $2=weights $3=tag-prefix $4=users-file(subset).
# Use a VAL subset for a quick ranking; confirm the winner on FULL val with qat_eval.sh afterwards.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_tail.safetensors}; TP=${3:-t}; UF=${4:-scratchpad/rank_users.txt}
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "rank eval: $(echo $USERS|wc -w) users, W=$W, tags=${TP}fp32/${TP}i4r1, NPROC=$NPROC"
pass() { # $1 scope $2 tag
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done
  wait
}
# qfp32 uses no compression -> shifts/percol irrelevant; keep percol/shifts default for the compressed pass.
for u in $USERS; do
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="" RWKV_QUANT_SHIFTS=0 RWKV_LOWRANK_PERCOL=0 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${TP}fp32_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done
wait
pass "card:1:int4,note:1:int4" "${TP}i4r1"
echo "=== SCORE (vs champion fp32) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 ${TP}fp32 ${TP}i4r1
echo "RANK_DONE_${TP}"
