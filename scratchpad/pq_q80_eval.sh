#!/usr/bin/env bash
# task22 80-b rung VERDICT chain: waits for the q80 GPU eval, converts the checkpoint, then full
# 400-user CPU deploy-engine eval at true 80 b (RWKV_PQ_NORM_BITS=4, tag q80m4 - lowercase-unique,
# NTFS is case-insensitive). Reference: 88-b q88m4 was +0.0023/+0.0006. Gate +0.0025 both.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q80.log
echo "q80 chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q80.log || { echo "Q80 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q80_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q80 CHECKPOINT (trainer crash despite exit 0?) - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q80_shiftcb_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] || { echo "NO EXPORTED LEARNED CODEBOOK - ABORT"; exit 1; }
echo "q80: converting $PTH  (learned shift cb: $SCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q80.safetensors
cp "$SCB" reference/pq_cb_shift_q80.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q80.safetensors
PQ3=scratchpad/pq_cb_m2b3.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q80 CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q80m4_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQ3 RWKV_SHIFT_PQ=reference/pq_cb_shift_q80.txt \
      RWKV_PQ_NORM_BITS=4 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q80m4_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 80-b rung: q80m4 = TRUE 80 b VERDICT. Gate +0.0025. 88-b q88m4 was +0.0023/+0.0006 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q80m4
echo "Q80EVAL_DONE"
