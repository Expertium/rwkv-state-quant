#!/usr/bin/env bash
# Shift-PQ step 1: dump TOKEN-SHIFT vector corpus from the fp32 CHAMPION (same choice as the WKV codebook —
# champion-trained beat co-adapted), raw fp32 (no compression env). Same 20 dev users as the WKV corpus.
# -> scratchpad/corpus_shift/. Then pq_train_shift.py trains the 2-role codebook. Args: $1=NPROC.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-6}
W=reference/champ_h2k16.safetensors
OUT=scratchpad/corpus_shift
mkdir -p "$OUT"
USERS="6013 6023 6036 6044 6056 6059 6109 6121 6150 6152 6184 6245 6270 6275 6283 6294 6408 6419 6423 6435"
echo "shift corpus dump: W=$W -> $OUT (20 users, card+note, fp32)"
for u in $USERS; do
  for s in card note; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=reference_big RWKV_PRED_DIR=preds \
        $BIN --dump-shift-corpus $u $s > "$OUT/${s}_${u}.txt" 2>/dev/null ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done
done; wait
echo "dumped: $(ls "$OUT" | wc -l) files, TS=$(cat "$OUT"/*.txt | grep -c '^TS') CS=$(cat "$OUT"/*.txt | grep -c '^CS')"
echo "SHIFT_DUMP_DONE"
