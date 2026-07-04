#!/usr/bin/env bash
# task22 chained convert+eval: q144 (m2b4 WKV + PQ shifts, FIXED codebook control, 2.0 ep). 2-pass VAL
# eval @ 144 b (RWKV_LOWRANK_PQ=m2b4 + RWKV_SHIFT_PQ=shift cb) + score. Gate +0.0025 both.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-6}; UF=${2:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_q144.log
echo "q144 chain: polling $TLOG for the training end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$TLOG" 2>/dev/null && break; sleep 30; done
grep -q DONE_EXIT_0 "$TLOG" || { echo "Q144 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q144_*.pth | grep -v ema | grep -v optim | grep -v shiftcb | head -1)
echo "q144: converting $PTH"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q144.safetensors
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
PQ4=scratchpad/pq_cb_m2b4.txt; SCB=scratchpad/pq_cb_shift_m4b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q144 val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag
  echo "  pass $5 (W=$1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
LR="card:1:int4,note:1:int4"
pass reference/qat_pq_q144.safetensors ""    ""                                                 0 q144_base
pass reference/qat_pq_q144.safetensors "$LR" "RWKV_LOWRANK_PQ=$PQ4 RWKV_SHIFT_PQ=$SCB"          1 q144_pq
echo "=== VAL SCORE q144 @144 b (m2b4 + PQ shifts, fixed cb). Gate +0.0025 both ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q144_base q144_pq
echo "Q144EVAL_DONE"
