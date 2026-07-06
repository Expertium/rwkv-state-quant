#!/usr/bin/env bash
# task22 52-b rung VERDICT chain (q52s = m1b4 joint WKV + m4b4 shifts + rotation + anneal + KD 0.2 +
# resurrection): waits for the q52s GPU eval, converts checkpoint + BOTH learned codebooks + the
# LEARNED ROTATION, then full 400-user CPU deploy eval at true 52 b (RWKV_SHIFT_ROT +
# RWKV_PQ_NORM_BITS=1, tag q52sv). 56-b champ q56sv = +0.002443/+0.002243 (the stack cracked the
# 16-entry shift wall); m1b4 alone failed ahead +0.0014 over gate at 68 b. Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q52s.log
echo "q52s chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q52s.log || { echo "Q52S TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q52s_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q52S CHECKPOINT - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q52s_shiftcb_*.txt 2>/dev/null | head -1)
WCB=$(ls -t gpu_train/reference/qat_pq_q52s_wkvcb_*.txt 2>/dev/null | head -1)
RCB=$(ls -t gpu_train/reference/qat_pq_q52s_shiftrot_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] && [ -n "$WCB" ] && [ -n "$RCB" ] || { echo "MISSING EXPORTED CODEBOOK/ROTATION - ABORT"; exit 1; }
echo "q52s: converting $PTH  (shift cb: $SCB  wkv cb: $WCB  rot: $RCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q52s.safetensors
cp "$SCB" reference/pq_cb_shift_q52s.txt
cp "$WCB" reference/pq_cb_wkv_q52s.txt
cp "$RCB" reference/pq_rot_shift_q52s.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q52s.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q52s CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q52sv_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q52s.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q52s.txt \
      RWKV_SHIFT_ROT=reference/pq_rot_shift_q52s.txt RWKV_PQ_NORM_BITS=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q52sv_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 52-b rung (BOTH 16-entry walls + full lever stack): q52sv = TRUE 52 b VERDICT. Gate +0.0025. 56-b champ q56sv +0.002443/+0.002243; m1b4 w/o levers failed ahead +0.0039 at 68 b ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q52sv
echo "Q52SEVAL_DONE"
