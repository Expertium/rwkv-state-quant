#!/usr/bin/env bash
# F28 chained convert+eval: waits for the m4s3 training's DONE_EXIT (m4b8 WKV + int3-shift QAT, 0.75 ep),
# converts the final RAW ckpt, then 2-pass VAL eval @ EXACTLY 352 b: m4b8 codebook + RWKV_STATE_SHIFT_LEVEL=int3.
# Est. +0.0019-0.0024. Beat = e75_pq +0.0021/+0.0012. Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_m4s3.log
echo "m4s3 chain: polling $TLOG for DONE_EXIT"
for i in $(seq 1 3000); do grep -q DONE_EXIT "$TLOG" 2>/dev/null && break; sleep 30; done
grep -q DONE_EXIT_0 "$TLOG" || { echo "M4S3 TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_m4s3_*.pth | grep -v ema | grep -v optim | head -1)
echo "m4s3: converting $PTH"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_m4s3.safetensors
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; PQ4=scratchpad/pq_cb_m4b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "m4s3 val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
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
pass reference/qat_pq_m4s3.safetensors ""    ""                                                    0 m4s3_base
pass reference/qat_pq_m4s3.safetensors "$LR" "RWKV_LOWRANK_PQ=$PQ4 RWKV_STATE_SHIFT_LEVEL=int3"    1 m4s3_pq
echo "=== VAL SCORE m4s3 @ EXACTLY 352 b (vs champion fp32). Beat: e75_pq +0.0021/+0.0012 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 m4s3_base m4s3_pq
echo "M4S3EVAL_DONE"
