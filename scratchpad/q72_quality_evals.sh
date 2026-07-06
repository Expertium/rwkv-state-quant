#!/usr/bin/env bash
# task22 quality pass at the CONFIRMED 72-b champion: four FREE (CPU-only) candidates, serial.
#   q72je = q72j EMA checkpoint (decay 0.99) + q72j codebooks   [EMA was ~nil at 352 b; re-check at 72 b]
#   q72be = q72b EMA checkpoint + q72b codebooks
#   q72sj = (q72j + q72b)/2 weight SOUP + q72j codebooks        [same-basin 2-seed average; co-adapt
#   q72sb = same soup + q72b codebooks                           lesson predicts failure -- cheap to test]
# Reference: q72jv +0.0018/+0.0016 (seed 1234), q72bv +0.0019/+0.0008 (seed 4321), gate +0.0025.
# Args: $1=NPROC $2=users-file.
set -e
cd "$(dirname "$0")/.."
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
PY=/c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe

# --- build the two EMA safetensors + the soup checkpoints ---
"$PY" - <<'EOF'
import torch
ref = "gpu_train/reference"
j = torch.load(f"{ref}/qat_pq_q72j_6702.pth", weights_only=True)
b = torch.load(f"{ref}/qat_pq_q72b_6702.pth", weights_only=True)
soup = {k: (j[k].float() + b[k].float()) / 2 if j[k].is_floating_point() else j[k] for k in j}
torch.save(soup, f"{ref}/qat_pq_q72soup_6702.pth")
print("soup checkpoint written (elementwise mean of q72j/q72b fp32 masters)")
EOF
"$PY" scratchpad/pth_to_sft.py gpu_train/reference/qat_pq_q72j_ema_6702.pth reference/qat_pq_q72je.safetensors
"$PY" scratchpad/pth_to_sft.py gpu_train/reference/qat_pq_q72b_ema_6702.pth reference/qat_pq_q72be.safetensors
"$PY" scratchpad/pth_to_sft.py gpu_train/reference/qat_pq_q72soup_6702.pth reference/qat_pq_q72soup.safetensors

BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")

run_tag () {  # $1=tag $2=weights $3=wkv-cb $4=shift-cb
  local tag=$1 w=$2 wcb=$3 scb=$4
  echo "=== eval $tag: $(echo $USERS|wc -w) users, NPROC=$NPROC (w=$w) ==="
  for u in $USERS; do
    [ -f "$PRED/rust_pred_${tag}_${u}.json" ] && continue
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$w RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
        RWKV_LOWRANK_PQ=$wcb RWKV_SHIFT_PQ=$scb RWKV_PQ_NORM_BITS=1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${tag}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}

run_tag q72je reference/qat_pq_q72je.safetensors  reference/pq_cb_wkv_q72j.txt reference/pq_cb_shift_q72j.txt
run_tag q72be reference/qat_pq_q72be.safetensors  reference/pq_cb_wkv_q72b.txt reference/pq_cb_shift_q72b.txt
run_tag q72sj reference/qat_pq_q72soup.safetensors reference/pq_cb_wkv_q72j.txt reference/pq_cb_shift_q72j.txt
run_tag q72sb reference/qat_pq_q72soup.safetensors reference/pq_cb_wkv_q72b.txt reference/pq_cb_shift_q72b.txt

echo "=== VAL SCORE 72-b QUALITY PASS: EMA x2 + 2-seed soup x2 vs the raw champions (q72jv +0.0018/+0.0016, q72bv +0.0019/+0.0008, gate +0.0025) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 q72je q72be q72sj q72sb
echo "Q72QEVAL_DONE"
