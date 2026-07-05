#!/usr/bin/env bash
# 80-b-via-norms probe, PURE PTQ (no training): the q88N winner (norms MODELED at int4 in QAT) deployed with
# int2 norms (RWKV_PQ_NORM_BITS=2) = 88 - 8 = 80 b/card, tag q80n2p. Rationale: int3==int4==int5 PTQ showed
# the norm tax is RANGE-limited, not resolution-limited, so int3 may be ~free. Gate +0.0025 both.
# q88m4 (int4 norms) was +0.0023/+0.0006. Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q88N.safetensors
PQ3=scratchpad/pq_cb_m2b3.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q80n2p PTQ eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q80n2p_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQ3 RWKV_SHIFT_PQ=reference/pq_cb_shift_q88N.txt \
      RWKV_PQ_NORM_BITS=2 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q80n2p_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 80-b-via-norms PTQ probe: q80n2p (q88N weights, int2 norms). Gate +0.0025. int4 was +0.0023/+0.0006 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q80n2p
echo "Q84N3EVAL_DONE"
