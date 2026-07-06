#!/usr/bin/env bash
# task22 56-b SEED-VARIANCE verdict chain (q56b = exact q56s win recipe, only seed 1234->4321):
# waits for the q56b GPU eval, converts checkpoint + BOTH learned codebooks + the LEARNED ROTATION,
# then full 400-user CPU deploy eval at true 56 b (RWKV_SHIFT_ROT + RWKV_PQ_NORM_BITS=1, tag q56bv).
# Decides whether q56sv +0.002443/+0.002243 (imm margin 0.000057) is REAL or seed luck — q64bv
# already proved the 64-b boundary win was luck (seed alone moved ahead +0.0018).
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q56b.log
echo "q56b chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q56b.log || { echo "Q56B TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q56b_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q56B CHECKPOINT - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q56b_shiftcb_*.txt 2>/dev/null | head -1)
WCB=$(ls -t gpu_train/reference/qat_pq_q56b_wkvcb_*.txt 2>/dev/null | head -1)
RCB=$(ls -t gpu_train/reference/qat_pq_q56b_shiftrot_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] && [ -n "$WCB" ] && [ -n "$RCB" ] || { echo "MISSING EXPORTED CODEBOOK/ROTATION - ABORT"; exit 1; }
echo "q56b: converting $PTH  (shift cb: $SCB  wkv cb: $WCB  rot: $RCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q56b.safetensors
cp "$SCB" reference/pq_cb_shift_q56b.txt
cp "$WCB" reference/pq_cb_wkv_q56b.txt
cp "$RCB" reference/pq_rot_shift_q56b.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q56b.safetensors
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q56b CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q56bv_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q56b.txt RWKV_SHIFT_PQ=reference/pq_cb_shift_q56b.txt \
      RWKV_SHIFT_ROT=reference/pq_rot_shift_q56b.txt RWKV_PQ_NORM_BITS=1 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q56bv_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 56-b SEED-VARIANCE (q56s recipe, seed 4321): q56bv. REPRODUCIBILITY test of q56sv +0.002443/+0.002243 (gate +0.0025; q64bv already exposed the 64-b win as seed luck) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q56bv
echo "Q56BEVAL_DONE"
