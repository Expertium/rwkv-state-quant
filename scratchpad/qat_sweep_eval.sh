#!/usr/bin/env bash
# Eval the tail-length sweep (e05/e20/e27) on FULL val, qfp32+qi4r1 only (skip the slow pathological qi2/qi4).
# Each qat_rank run scores vs the champion fp32. 0.1ep (qat_tail) already scored separately.
set -e
cd "$(dirname "$0")/.."
tr -d '\r' < val_users.txt > scratchpad/rank_users.txt
for m in e05 e20 e27; do
  echo "########## TAIL $m ##########"
  bash scratchpad/qat_rank.sh 14 reference/qat_tail_${m}.safetensors "${m}" scratchpad/rank_users.txt
done
echo "SWEEP_EVAL_DONE"
