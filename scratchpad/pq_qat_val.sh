#!/usr/bin/env bash
# F22 deploy-eval: PQ+QAT weights (qat_pq_m2b8) on full VAL vs CHAMPION fp32.
#   pqqat_base = uncompressed (fp32)                          -> base_drift = pqqat_base - champion_fp32
#   pqqat      = rank-1 PQ m2b8 WKV + shift int4 (~352 b/card) -> total deploy penalty = pqqat - champion_fp32
# compression_cost = pqqat - pqqat_base. Gate: total <= +0.0025 in BOTH imm AND ahead, robust per-user.
# Compare to F21 PTQ (+0.0046/+0.0040) and the F15 512-b win (+0.0024/+0.0021). Args: $1=NPROC $2=weights $3=users.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}; W=${2:-reference/qat_pq_m2b8.safetensors}; UF=${3:-scratchpad/valfull_users.txt}
TAG=${4:-pqqat}; PQ=${5:-scratchpad/pq_cb_m2b8.txt}
REF=reference_big; PRED=preds
PQ8=$PQ
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < "$UF")
echo "pq-QAT val: $(echo $USERS|wc -w) users, W=$W, NPROC=$NPROC"
pass() { # $1 lowrank-scope  $2 extra-env  $3 shifts  $4 tag  $5 desc
  echo "  pass $4 ($5)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$3" RWKV_LOWRANK_PERCOL=1 $2 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${4}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
pass ""                          ""                     0 ${TAG}_base "uncompressed (base_drift)"
pass "card:1:int4,note:1:int4"   "RWKV_LOWRANK_PQ=$PQ8" 1 ${TAG}      "rank-1 PQ + shift int4 (~352 b)"
echo "=== VAL SCORE (vs champion fp32). base_drift=${TAG}_base-fp32; total=${TAG}-fp32; gate +0.0025 BOTH ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py "$UF" fp32 ${TAG}_base ${TAG}
echo "PQQATVAL_DONE"
