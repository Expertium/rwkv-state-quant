#!/usr/bin/env bash
# Rank the (FIXED-LR) rank-1 int4 QAT sweep on FULL DEV by qi4r1 penalty vs champion fp32. lr1e3==F15 (reuse
# its cached dv_base preds); eval only the 3 new LRs. Lowest total that stays robust wins -> VAL-confirm.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
REF=reference_big; PRED=preds; NPROC=${1:-14}
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
USERS=$(tr -d '\r' < dev_users.txt)
echo "dev sweep: $(echo $USERS|wc -w) dev users, NPROC=$NPROC"
qi4r1() { # $1 weights  $2 tag
  echo "  pass $2 ($1)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS="$1" RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${2}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done; wait
}
qi4r1 reference/qat_lr1i4_lr5e4.safetensors nlr5e4
qi4r1 reference/qat_lr1i4_lr2e4.safetensors nlr2e4
qi4r1 reference/qat_lr1i4_lr1e4.safetensors nlr1e4
echo "=== DEV SCORE (qi4r1 vs champ fp32). dv_base = lr1e3 = F15 (cached). Pick lowest total, then VAL-confirm ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py dev_users.txt fp32 dv_base nlr5e4 nlr2e4 nlr1e4
echo "DEVSWEEP_DONE"
