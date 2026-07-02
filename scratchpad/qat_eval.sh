#!/usr/bin/env bash
# Eval the QAT model (reference/qat_fmi2.safetensors) on VAL for the parameter-free deploy schemes, scored
# vs the CHAMPION fp32 (tag fp32, the fair no-QAT-0.27 baseline). q-prefixed tags so champ baselines aren't
# clobbered. Answers: does full-matrix-int2 QAT make a low-rank ≤256-b deploy pass the gate?
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-14}
W=${2:-reference/qat_fmi2.safetensors}   # arg2 = QAT weights to eval (default qat_fmi2)
REF=reference_big; PRED=preds
# champion fp32 baseline must be present (restore from backup to be safe)
cp preds_baseline_backup/rust_pred_fp32_*.json "$PRED"/ 2>/dev/null || true
tr -d '\r' < val_users.txt > scratchpad/qat_users.txt
USERS=$(while read -r u; do printf '%s %s\n' "$(stat -c%s "$REF/trace_user_${u}.safetensors" 2>/dev/null||echo 0)" "$u"; done < scratchpad/qat_users.txt | sort -rn | awk '{print $2}')
echo "QAT eval: $(echo $USERS|wc -w) val users, weights=$W, NPROC=$NPROC"
pass() { # $1 lowrank_scope  $2 shifts  $3 percol  $4 tag
  echo "  pass: $4 (lowrank='$1' shifts=$2 percol=$3)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$2" RWKV_LOWRANK_PERCOL="$3" \
        $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${4}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done
  wait
}
pass ""                          0 0 qfp32     # QAT model, uncompressed (its own fp32)
pass "card:1:int4,note:1:int4"   1 1 qi4r1     # rank-1 int4 (256 b) -- the key ≤256-b test
pass "card:2:int2,note:2:int2"   1 1 qi2       # rank-2 int2 (256 b) -- did QAT stabilize it?
pass "card:2:int4,note:2:int4"   1 1 qi4       # rank-2 int4 (512 b) -- did QAT fix the blow-up?
echo "=== SCORE (penalty vs CHAMPION fp32 = fair no-QAT-0.27 baseline; qfp32 vs fp32 = base change from QAT) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py scratchpad/qat_users.txt fp32 qfp32 qi4r1 qi2 qi4
echo "QAT_EVAL_DONE"
