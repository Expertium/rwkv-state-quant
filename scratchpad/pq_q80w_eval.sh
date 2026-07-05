#!/usr/bin/env bash
# task22 80-b RETRY VERDICT chain (q80w = BOTH codebooks learned): waits for the q80w GPU eval, converts
# the checkpoint + BOTH learned codebooks, then full 400-user CPU deploy-engine eval at true 80 b
# (RWKV_PQ_NORM_BITS=4, tag q80w4 - lowercase-unique). Fixed-WKV-cb q80m4 was +0.0027/+0.0017 FAIL.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q80w.log
echo "q80w chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q80w.log || { echo "Q80W TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q80w_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q80W CHECKPOINT - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q80w_shiftcb_*.txt 2>/dev/null | head -1)
WCB=$(ls -t gpu_train/reference/qat_pq_q80w_wkvcb_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] && [ -n "$WCB" ] || { echo "MISSING EXPORTED LEARNED CODEBOOK(S) - ABORT"; exit 1; }
echo "q80w: converting $PTH  (shift cb: $SCB  wkv cb: $WCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q80w.safetensors
cp "$SCB" reference/pq_cb_shift_q80w.txt
cp "$WCB" reference/pq_cb_wkv_q80w.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q80w.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q80w CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q80w4_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q80w.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q80w.txt \
      RWKV_PQ_NORM_BITS=4 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q80w4_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 80-b RETRY (BOTH cbs learned): q80w4 = TRUE 80 b VERDICT. Gate +0.0025. Fixed-cb q80m4 was +0.0027/+0.0017 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q80w4
echo "Q80WEVAL_DONE"
