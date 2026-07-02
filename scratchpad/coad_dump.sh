#!/usr/bin/env bash
# Codebook CO-ADAPTATION step 1: dump WKV state corpus from the ep75 QAT net, running UNDER the PQ deploy
# (so the states are from the distribution the codebook will actually serve). Same 20 dev users as the
# original champion corpus -> scratchpad/corpus_coad/. Then step 2 (pq_train.py) retrains the codebook.
# Args: $1=NPROC.
set -e
cd "$(dirname "$0")/.."
BIN=./engine/target/release/rwkv-infer.exe
NPROC=${1:-10}
W=reference/qat_pq_ep75.safetensors
OUT=scratchpad/corpus_coad
mkdir -p "$OUT"
USERS="6013 6023 6036 6044 6056 6059 6109 6121 6150 6152 6184 6245 6270 6275 6283 6294 6408 6419 6423 6435"
echo "co-adapt corpus dump: W=$W -> $OUT (20 users, card+note, PQ-deploy env)"
for u in $USERS; do
  for s in card note; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=reference_big RWKV_PRED_DIR=preds \
        RWKV_STATE_LOWRANK_SCOPE="card:1:int4,note:1:int4" RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 \
        RWKV_LOWRANK_PQ=scratchpad/pq_cb_m2b8.txt \
        $BIN --dump-corpus $u $s > "$OUT/${s}_${u}.txt" 2>/dev/null ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done
done; wait
echo "dumped: $(ls "$OUT" | wc -l) files, $(cat "$OUT"/*.txt | grep -c STATE) states total"
echo "=== step 2: retrain m2b8 codebook on the co-adapted corpus ==="
OMP_NUM_THREADS=8 python scratchpad/pq_train.py scratchpad/pq_cb_m2b8_coad.txt "$OUT"/*.txt --m 2 --bits 8
echo "COAD_DUMP_DONE"
