#!/usr/bin/env bash
# 96-b via NORM QUANT (PTQ, no retraining): the proven q128L scheme (true 112 b) with its four norm
# scalars at int4 (=96 b, tag q128n4) and int5 (=100 b, tag q128n5). Full 400-user CPU eval — the
# GPU eval can't do norm quant (engine-only feature), so this IS the verdict path.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q128L.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q128L norm-quant PTQ: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 norm-bits  $2 tag
  echo "  pass $2 (NORM_BITS=$1)"
  for u in $USERS; do
    [ -f "$PRED/rust_pred_${2}_${u}.json" ] && continue
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
        RWKV_LOWRANK_PQ=scratchpad/pq_cb_m2b4.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q128L.txt \
        RWKV_PQ_NORM_BITS=$1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass 4 q128n4
pass 5 q128n5
echo "=== VAL SCORE norm-quant PTQ on q128L (96 b @ int4, 100 b @ int5). Gate +0.0025. q128L was +0.0018/-0.0002 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q128n4 q128n5
echo "Q128NEVAL_DONE"
