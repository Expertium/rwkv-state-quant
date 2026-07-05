#!/usr/bin/env bash
# task22 64-b rung VERDICT chain (q64a = q64r + soft-to-hard selection ANNEALING tau0 0.05): waits for the q64a
# GPU eval, converts checkpoint + BOTH learned codebooks + the LEARNED ROTATION, then full 400-user
# CPU deploy eval at true 64 b (RWKV_SHIFT_ROT + RWKV_PQ_NORM_BITS=1, tag q64av). m4b5 withOUT
# rotation failed twice at +0.0027; 72-b champ q72jv = +0.0018/+0.0016. Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q64a.log
echo "q64a chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q64a.log || { echo "Q64A TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q64a_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q64A CHECKPOINT - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q64a_shiftcb_*.txt 2>/dev/null | head -1)
WCB=$(ls -t gpu_train/reference/qat_pq_q64a_wkvcb_*.txt 2>/dev/null | head -1)
RCB=$(ls -t gpu_train/reference/qat_pq_q64a_shiftrot_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] && [ -n "$WCB" ] && [ -n "$RCB" ] || { echo "MISSING EXPORTED CODEBOOK/ROTATION - ABORT"; exit 1; }
echo "q64a: converting $PTH  (shift cb: $SCB  wkv cb: $WCB  rot: $RCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q64a.safetensors
cp "$SCB" reference/pq_cb_shift_q64a.txt
cp "$WCB" reference/pq_cb_wkv_q64a.txt
cp "$RCB" reference/pq_rot_shift_q64a.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q64a.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q64a CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q64av_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q64a.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q64a.txt \
      RWKV_SHIFT_ROT=reference/pq_rot_shift_q64a.txt RWKV_PQ_NORM_BITS=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q64av_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 64-b rung (ROTATION + m4b5 + ANNEAL tau0 0.05): q64av = TRUE 64 b VERDICT. A/B vs q64rv +0.002492/+0.0012. Gate +0.0025. m4b5 sans rotation failed +0.0027; 72-b champ +0.0018/+0.0016 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q64av
echo "Q64AEVAL_DONE"
