#!/usr/bin/env bash
# task22 chained convert+eval: waits for the s3e150 training's DONE_EXIT, converts the final RAW ckpt,
# then 2-pass VAL eval @ 288 b (m2b8 codebook + RWKV_STATE_SHIFT_LEVEL=int3) + score.
# Beat: gate +0.0025 both; refs e150_pq@352 +0.0010/-0.0003, e150s3 PTQ diagnostic (see pq_e150s3.log).
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_s3e150.log
echo "s3e150 chain: polling $TLOG for the training end marker"
for i in $(seq 1 3000); do grep -qE 'DONE_EXIT_[0-9]' "$TLOG" 2>/dev/null && break; sleep 30; done
grep -q DONE_EXIT_0 "$TLOG" || { echo "S3E150 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_s3e150_*.pth | grep -v ema | grep -v optim | head -1)
echo "s3e150: converting $PTH"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_s3e150.safetensors
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "s3e150 val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
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
pass reference/qat_pq_s3e150.safetensors ""    ""                                                    0 s3150_base
pass reference/qat_pq_s3e150.safetensors "$LR" "RWKV_LOWRANK_PQ=$PQ8 RWKV_STATE_SHIFT_LEVEL=int3"    1 s3150_pq
echo "=== VAL SCORE s3e150 @288 b (vs champion fp32). Gate +0.0025. e150_pq@352 was +0.0010/-0.0003 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 s3150_base s3150_pq
echo "S3E150EVAL_DONE"
