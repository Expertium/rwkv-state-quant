#!/usr/bin/env bash
# F23 deploy-eval: the two lower-LR PQ+QAT weights on full VAL vs CHAMPION fp32. For each: uncompressed
# (base_drift) + rank-1 PQ m2b8 WKV + shift int4 (~352 b). Gate +0.0025 both. Args: $1=NPROC $2=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; UF=${2:-scratchpad/valfull_users.txt}
REF=reference_big; PRED=preds; PQ8=scratchpad/pq_cb_m2b8.txt
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "pq-QAT lower-LR val: $(echo $USERS|wc -w) users, NPROC=$NPROC"
pass() { # $1 weights  $2 lowrank-scope  $3 extra-env  $4 shifts  $5 tag  $6 desc
  echo "  pass $5 (W=$1 ; $6)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$1 RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$4" RWKV_LOWRANK_PERCOL=1 $3 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${5}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
W5=reference/qat_pq_m2b8_lr5e4.safetensors
W2=reference/qat_pq_m2b8_lr2e4.safetensors
pass "$W5" ""                        ""                     0 pq5e4_base "uncompressed"
pass "$W5" "card:1:int4,note:1:int4" "RWKV_LOWRANK_PQ=$PQ8"  1 pq5e4      "PQ + shift int4 (~352 b)"
pass "$W2" ""                        ""                     0 pq2e4_base "uncompressed"
pass "$W2" "card:1:int4,note:1:int4" "RWKV_LOWRANK_PQ=$PQ8"  1 pq2e4      "PQ + shift int4 (~352 b)"
echo "=== VAL SCORE (vs champion fp32). base_drift=*_base-fp32; total=*-fp32; gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 pq5e4_base pq5e4 pq2e4_base pq2e4
echo "PQLREVAL_DONE"
