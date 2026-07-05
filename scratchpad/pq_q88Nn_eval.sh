#!/usr/bin/env bash
# task22 88-b RETRY VERDICT chain (q88N = norm quant MODELED in QAT): waits for the q88N GPU eval,
# converts the checkpoint, then full 400-user CPU deploy-engine eval at true 88 b (RWKV_PQ_NORM_BITS=4,
# tag q88m4). Reference: q88n4 (un-modeled) was +0.0026/+0.0010. Gate +0.0025 both.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q88N.log
echo "q88Nn chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q88N.log || { echo "Q88N TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q88N_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q88N CHECKPOINT (trainer crash despite exit 0?) - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q88N_shiftcb_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] || { echo "NO EXPORTED LEARNED CODEBOOK - ABORT"; exit 1; }
echo "q88Nn: converting $PTH  (learned shift cb: $SCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q88N.safetensors
cp "$SCB" reference/pq_cb_shift_q88N.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q88N.safetensors
PQ3=scratchpad/pq_cb_m2b3.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q88N CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
for u in $USERS; do
  [ -f "$PRED/rust_pred_q88m4_${u}.json" ] && continue
  ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
      RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
      RWKV_LOWRANK_PQ=$PQ3 RWKV_SHIFT_PQ=reference/pq_cb_shift_q88N.txt \
      RWKV_PQ_NORM_BITS=4 $BIN $u >/dev/null 2>&1
    cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_q88m4_${u}.json" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
done; wait
echo "=== VAL SCORE 88-b RETRY (norm quant MODELED in QAT): q88m4 = TRUE 88 b VERDICT. Gate +0.0025. Un-modeled q88n4 was +0.0026/+0.0010 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q88m4
echo "Q88NNEVAL_DONE"
