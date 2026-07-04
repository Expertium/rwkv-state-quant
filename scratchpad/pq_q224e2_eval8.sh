#!/usr/bin/env bash
# q224e2 eval RESTART at NPROC=8 (Andrew 12:20: don't max the CPU while GPU runs). RESUME-AWARE: skips
# users whose tagged pred already exists (the killed 14-proc run finished 360/400 of q224e2_base).
# 2-pass VAL eval @ 224 b (m2b8 + RWKV_STATE_SHIFT_LEVEL=int2) + score. 1.5-ep point was +0.0027/+0.0012.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-8}; UF=${2:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_q224e2.log
grep -q DONE_EXIT_0 "$TLOG" || { echo "Q224E2 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
[ -f reference/qat_pq_q224e2.safetensors ] || { echo "MISSING CONVERTED WEIGHTS - ABORT"; exit 1; }
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q224e2 val (resume, NPROC=$NPROC): $(echo $USERS|wc -w) users"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag
  echo "  pass $5 (W=$1)"
  for u in $USERS; do
    [ -f "$PRED/rust_pred_${5}_${u}.json" ] && continue
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
LR="card:1:int4,note:1:int4"
pass reference/qat_pq_q224e2.safetensors ""    ""                                                    0 q224e2_base
pass reference/qat_pq_q224e2.safetensors "$LR" "RWKV_LOWRANK_PQ=$PQ8 RWKV_STATE_SHIFT_LEVEL=int2"    1 q224e2_pq
echo "=== VAL SCORE q224e2 @224 b, 2.0 ep (vs champion fp32). Gate +0.0025. 1.5 ep was +0.0027/+0.0012 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q224e2_base q224e2_pq
echo "Q224E2EVAL_DONE"
