#!/usr/bin/env bash
# Candle A/B: re-confirm the fast-path i4r1 / i4 penalties on the new H=2/K=16 model are NOT a
# fast-path numerical artifact. Runs a subset of VAL users through BOTH the candle path (RWKV_USE_CANDLE=1)
# and reuses the cached fast-path tagged preds, then compares the PENALTY (scheme - fp32) computed within
# each path on the SAME users. If candle penalty ~= fast penalty, the +0.0036 i4r1 result is real.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
W=reference/champ_h2k16.safetensors
REF=reference_big; PRED=preds
# ~12 VAL users in a medium size band (1-25 MB trace) so candle (~6.7x slower) stays fast but the
# recurrence is long enough to expose any accumulation bias; spread across the band.
SUB=$(while read u; do u=$(echo "$u"|tr -d '\r'); s=$(stat -c%s "$REF/trace_user_${u}.safetensors" 2>/dev/null||echo 0);
        echo "$s $u"; done < val_users.txt | sort -n | awk '$1>1000000 && $1<25000000{print $2}' | awk 'NR%12==1' | head -12)
echo "subset users: $SUB"
printf '%s\n' $SUB > scratchpad/abusers.txt

run_candle() { # $1 tag  $2 lowrank_scope  $3 shifts  $4 percol
  for u in $SUB; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_USE_CANDLE=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$3" RWKV_LOWRANK_PERCOL="$4" \
        $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${1}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge 12 ]; do wait -n 2>/dev/null || true; done
  done
  wait
}
echo "=== candle fp32c ==="; run_candle fp32c "" 0 0
echo "=== candle i4r1c ==="; run_candle i4r1c "card:1:int4,note:1:int4" 1 1
echo; echo "########## CANDLE (subset) -- penalty = i4r1c - fp32c (compute by hand) ##########"
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py scratchpad/abusers.txt fp32c i4r1c
echo; echo "########## FAST (same subset, cached tagged preds) ##########"
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py scratchpad/abusers.txt fp32 i4r1
echo CANDLE_AB_DONE
