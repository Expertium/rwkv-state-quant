#!/usr/bin/env bash
# Sweep3 chained convert+eval (task21): waits for the epN training's DONE_EXIT, converts the final RAW
# checkpoint, then 2-pass VAL eval (RAW only — EMA proven ±0.0001) + score. Tags eN_base / eN_pq.
# m2b8 codebook + int4 shifts (~352 b). Args: $1=N (100|150|200)  $2=NPROC  $3=users-file.
# Trend ref: 0.5ep +0.0028/+0.0018 -> 0.75ep +0.0021/+0.0012 (champion e75_pq). Gate +0.0025 both.
set -e
cd "$(dirname "$0")/.."
N=$1; NPROC=${2:-14}; UF=${3:-scratchpad/valfull_users.txt}
TLOG=scratchpad/qat_qat_pq_ep${N}.log
echo "ep${N} chain: polling $TLOG for DONE_EXIT"
for i in $(seq 1 3000); do grep -q DONE_EXIT "$TLOG" 2>/dev/null && break; sleep 30; done
grep -q DONE_EXIT_0 "$TLOG" || { echo "EP${N} TRAINING DID NOT END DONE_EXIT_0 - ABORT"; exit 1; }
PTH=$(ls -t gpu_train/reference/qat_pq_ep${N}_*.pth | grep -v ema | grep -v optim | head -1)
echo "ep${N}: converting $PTH"
/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe scratchpad/pth_to_sft.py "$PTH" reference/qat_pq_ep${N}.safetensors
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "ep${N} val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag
  echo "  pass $5 (W=$1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
LR="card:1:int4,note:1:int4"; PQE="RWKV_LOWRANK_PQ=$PQ8"
pass reference/qat_pq_ep${N}.safetensors ""    ""     0 e${N}_base
pass reference/qat_pq_ep${N}.safetensors "$LR" "$PQE" 1 e${N}_pq
echo "=== VAL SCORE ep${N} (vs champion fp32). Trend: 0.5ep +0.0028/+0.0018, 0.75ep +0.0021/+0.0012 ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 e${N}_base e${N}_pq
echo "EP${N}EVAL_DONE"
