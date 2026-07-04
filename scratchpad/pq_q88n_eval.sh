#!/usr/bin/env bash
# task22 88-b rung VERDICT chain: waits for the q88L GPU eval to finish (serial queue), converts the
# q88L checkpoint, then full 400-user CPU eval: q88L_pq (fp norms, drift readout) + q88n4 (int4 log2
# norms = the TRUE 88 b/card). GPU eval can't do norm quant (engine-only), so q88n4 IS the verdict.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
GLOG=scratchpad/gpu_eval_q88L.log
echo "q88n chain: polling $GLOG for the GPU-eval end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$GLOG" 2>/dev/null && break; sleep 30; done
grep -qE 'DONE_EXIT_[0-9]' "$GLOG" || { echo "GPU EVAL NEVER FINISHED - ABORT"; exit 1; }
grep -q DONE_EXIT_0 scratchpad/qat_qat_pq_q88L.log || { echo "Q88L TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q88L_*.pth 2>/dev/null | grep -v ema | grep -v optim | head -1)
[ -n "$PTH" ] || { echo "NO Q88L CHECKPOINT (trainer crash despite exit 0?) - ABORT"; exit 1; }
SCB=$(ls -t gpu_train/reference/qat_pq_q88L_shiftcb_*.txt 2>/dev/null | head -1)
[ -n "$SCB" ] || { echo "NO EXPORTED LEARNED CODEBOOK - ABORT"; exit 1; }
echo "q88n: converting $PTH  (learned shift cb: $SCB)"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q88L.safetensors
cp "$SCB" reference/pq_cb_shift_q88L.txt
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
W=reference/qat_pq_q88L.safetensors
PQ3=scratchpad/pq_cb_m2b3.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q88L CPU eval: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 extra-env  $2 tag
  echo "  pass $2"
  for u in $USERS; do
    [ -f "$PRED/rust_pred_${2}_${u}.json" ] && continue
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
        RWKV_LOWRANK_PQ=$PQ3 RWKV_SHIFT_PQ=reference/pq_cb_shift_q88L.txt \
        $1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass "RWKV_PQ_NORM_BITS=4" q88n4
pass ""                    q88L_pq
echo "=== VAL SCORE 88-b rung: q88n4 (TRUE 88 b, THE VERDICT) + q88L_pq (fp norms). Gate +0.0025. 96-b was +0.0022/+0.0003 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q88n4 q88L_pq
echo "Q88NEVAL_DONE"
