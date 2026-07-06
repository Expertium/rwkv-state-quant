#!/usr/bin/env bash
# task22 72-b rung VERDICT chain (q72f = joint m1b5 WKV cb, both cbs learned, 1-bit norms): waits for
# the q72f GPU eval, converts checkpoint + BOTH learned codebooks, then full 400-user CPU deploy eval
# at true 72 b (RWKV_PQ_NORM_BITS=1, tag q72fv - lowercase-unique). 76-b champ q76n1p = +0.0023/+0.0005.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q72f.log
echo "q72f chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q72f.log || { echo "Q72D TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q72f_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q72D CHECKPOINT - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q72f_shiftcb_*.txt 2>/dev/null | head -1)
WCB=$(ls -t gpu_train/reference/qat_pq_q72f_wkvcb_*.txt 2>/dev/null | head -1)
RCB=$(ls -t gpu_train/reference/qat_pq_q72f_shiftrot_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] && [ -n "$WCB" ] && [ -n "$RCB" ] || { echo "MISSING EXPORTED CODEBOOK/ROTATION - ABORT"; exit 1; }
echo "q72f: converting $PTH  (shift cb: $SCB  wkv cb: $WCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q72f.safetensors
cp "$SCB" reference/pq_cb_shift_q72f.txt
cp "$WCB" reference/pq_cb_wkv_q72f.txt
cp "$RCB" reference/pq_rot_shift_q72f.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q72f.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q72f CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q72fv_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q72f.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q72f.txt RWKV_SHIFT_ROT=reference/pq_rot_shift_q72f.txt \
      RWKV_PQ_NORM_BITS=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q72fv_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 72-b rung (q72c + LEARNED SHIFT ROTATION on m2b12): q72fv = TRUE 72 b VERDICT. A/B vs champion q72cv +0.001295/+0.000212. Gate +0.0025 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q72fv
echo "Q72FEVAL_DONE"
