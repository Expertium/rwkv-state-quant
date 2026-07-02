#!/usr/bin/env bash
# Frontier / ceiling diagnostics on DEV (400). Parameter-free schemes, but run on DEV to keep VAL clean for
# final confirmation. Reuses cached fp32 preds for the penalty. Fast path (RWKV_USE_CANDLE unset), NPROC=16.
# Candidates:
#   r1fp  card:1,note:1            shifts=0  -> PURE rank-1 ceiling (only the rank-1 truncation, nothing quantized)
#   r2fp  card:2,note:2            shifts=0  -> PURE rank-2 ceiling (does the 2nd component carry signal, unquantized?)
#   r1i5  card:1:int5,note:1:int5  shifts=1  -> rank-1 int5 (320 bits, over budget) -- how much precision closes the gap
#   r1i3  card:1:int3,note:1:int3  shifts=1  -> rank-1 int3 (192 bits, under budget)
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
W=reference/champ_h2k16.safetensors
REF=reference_big; PRED=preds
( cd engine && cargo build --release ) 2>&1 | tail -1
# DEV users, LPT order (biggest first) so the work-queue doesn't strand a monster at the end.
USERS=$(while read u; do u=$(echo "$u"|tr -d '\r'); printf '%s %s\n' "$(stat -c%s "$REF/trace_user_${u}.safetensors" 2>/dev/null||echo 0)" "$u"; done < dev_users.txt | sort -rn | awk '{print $2}')
printf '%s\n' $USERS > scratchpad/frontier_users.txt
echo "dev users: $(echo $USERS|wc -w)"

run() { # $1 tag  $2 lowrank_scope  $3 shifts
  echo "=== $1  (lowrank='$2' shifts=$3) ==="
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$3" RWKV_LOWRANK_PERCOL=1 \
        $BIN $u >/dev/null 2>&1
      cp "$PRED/rust_pred_${u}.json" "$PRED/rust_pred_${1}_${u}.json" ) &
    while [ "$(jobs -rp | wc -l)" -ge 16 ]; do wait -n 2>/dev/null || true; done
  done
  wait
}
run r1fp "card:1,note:1"           0
run r2fp "card:2,note:2"           0
run r1i5 "card:1:int5,note:1:int5" 1
run r1i3 "card:1:int3,note:1:int3" 1

echo; echo "########## DEV frontier (penalty vs fp32) ##########"
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED python score.py dev_users.txt fp32 i4r1 r1fp r2fp r1i5 r1i3
echo FRONTIER_DONE
