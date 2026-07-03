#!/usr/bin/env bash
# task22 rung 2 chained convert+eval: waits for the q272 training, converts, 2-pass VAL eval @ 272 b
# (m2b6 codebook + RWKV_STATE_SHIFT_LEVEL=int3) + score. Gate +0.0025 both. Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_q272.log
echo "q272 chain: polling $TLOG for the training end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$TLOG" 2>/dev/null && break; sleep 30; done
grep -q DONE_EXIT_0 "$TLOG" || { echo "Q272 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_q272_*.pth | grep -v ema | grep -v optim | head -1)
echo "q272: converting $PTH"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_q272.safetensors
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; PQ6=scratchpad/pq_cb_m2b6.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "q272 val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
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
pass reference/qat_pq_q272.safetensors ""    ""                                                    0 q272_base
pass reference/qat_pq_q272.safetensors "$LR" "RWKV_LOWRANK_PQ=$PQ6 RWKV_STATE_SHIFT_LEVEL=int3"    1 q272_pq
echo "=== VAL SCORE q272 @272 b (vs champion fp32). Gate +0.0025 both ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q272_base q272_pq
echo "Q272EVAL_DONE"
