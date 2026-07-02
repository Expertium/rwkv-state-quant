#!/usr/bin/env bash
# THE LOOP (H=2/K=16 phase): build the engine, run each scheme over the DEV (or VAL) user set, score by
# by-user-mean LOGLOSS. This is the REAL metric (recall-prediction log-loss after the compressed state
# flows through the recurrence + heads) -- NOT a reconstruction proxy. Run:  bash run_eval.sh [NPROC] [dev|val|both]
#
# A "scheme" = how the per-card / per-note WKV state is compressed each review step, set by env vars on the
# engine (or new Rust in engine/src/model.rs::lowrank_roundtrip). fp32/int4/int2 baselines + the current
# champion i4r1 always run. Phase log = research_log_h2k16.md (same methodology as research_log.md).
set -e
cd "$(dirname "$0")"
PY=python                                            # needs numpy + scikit-learn
BIN=./engine/target/release/rwkv-infer.exe
# NEW champion (2026-06-30): champ_h2k16 = H=2/K=16 (d_model=32 = 2 heads x 16). Per-layer WKV state is
# two 16x16 per-head matrices (512 floats) -- HALF the old single 32x32 (1024). Engine auto-derives H/K/C.
W="${RWKV_WEIGHTS:-reference/champ_h2k16.safetensors}"
REF="${REF_DIR:-reference_big}"                      # INPUT traces (trace_user_*) -- users 6000-6999
PRED="${PRED_DIR:-preds}"                            # OUTPUT predictions (rust_pred_*) -- kept SEPARATE
mkdir -p "$PRED"
NPROC=${1:-10}                                       # 10 CPU threads (each proc RAYON/OMP=1 -> NPROC procs)
# 400+400 split (2026-06-30): the export landed 836 of 6000-6999; the 36 LARGEST users (by review count)
# were moved to excluded36/ to speed runs, leaving 800 = 400 DEV (ids 6000-6435, dev_users.txt) + 400 VAL
# (ids 6436-6999, val_users.txt). Develop on DEV; CONFIRM the chosen scheme on HELD-OUT VAL before a win.
# A real WIN clears +0.0025 (BOTH imm AND ahead, <=256 bits) on BOTH dev AND val.
MODE="${2:-dev}"
# NOTE: tr -d '\r' guards against CRLF in the user-list files (Python on Windows writes \r\n) -- a
# trailing \r would corrupt every trace path ("trace_user_6000\r.safetensors").
case "$MODE" in
  val)  tr -d '\r' < val_users.txt > scratchpad/active_users.txt ;;
  both) cat dev_users.txt val_users.txt | tr -d '\r' > scratchpad/active_users.txt ;;
  *)    tr -d '\r' < dev_users.txt > scratchpad/active_users.txt; MODE=dev ;;
esac
# LPT (Longest Processing Time first): order users by trace .safetensors size DESCENDING so the biggest
# users start first and don't become end-of-run stragglers (with the work-queue dispatch in pass(), this
# ~minimizes makespan -- LPT is within 4/3 of optimal). Self-contained: stats the actual trace files.
USERS=$(while read -r u; do
          printf '%s %s\n' "$(stat -c%s "$REF/trace_user_${u}.safetensors" 2>/dev/null || echo 0)" "$u"
        done < scratchpad/active_users.txt | sort -rn | awk '{print $2}')
echo "MODE=$MODE  ($(echo $USERS | wc -w) users, LPT-ordered)  weights=$W  traces=$REF  preds=$PRED  NPROC=$NPROC"

echo "=== build engine (incremental) ==="
( cd engine && cargo build --release )

# CACHE the baselines: fp32/i4/i2/i4r1 are deterministic per (engine binary, weights), so once computed
# they're reused across candidate runs. Invalidated if the binary or weights mtime changed (any engine
# change could affect them). To keep the cache warm while iterating on a new scheme, make its tunable part
# (e.g. a PQ codebook) a RUNTIME input so no rebuild is needed; a real rebuild correctly forces a recompute.
BASELINE_TAGS=" fp32 i4 i2 i4r1 "
TOKEN="$(stat -c%Y "$BIN" 2>/dev/null)-$(stat -c%Y "$W" 2>/dev/null)"
if [ "$(cat "$PRED/.baseline_token" 2>/dev/null)" != "$TOKEN" ]; then
  echo "  (engine/weights changed -> clearing cached baseline preds)"
  rm -f "$PRED"/rust_pred_fp32_*.json "$PRED"/rust_pred_i4_*.json "$PRED"/rust_pred_i2_*.json "$PRED"/rust_pred_i4r1_*.json
  echo "$TOKEN" > "$PRED/.baseline_token"
fi

# pass: $1 LOWRANK_SCOPE  $2 QUANT_SCOPE  $3 SHIFTS  $4 PERCOL  $5 HADAMARD  $6 4LEVEL  $7 EXTRA_ENV  $8 tag
pass() {
  # CACHE: skip a BASELINE tag if all its preds already exist (token above guarantees they match the engine).
  # Candidates (non-baseline tags) always run.
  if [[ "$BASELINE_TAGS" == *" $8 "* ]]; then
    local miss=0
    for u in $USERS; do [ -f "$PRED/rust_pred_${8}_${u}.json" ] || { miss=1; break; }; done
    if [ "$miss" -eq 0 ]; then echo "  pass: $8 (cached -- skipping)"; return; fi
  fi
  echo "  pass: $8  (lowrank='$1' quant='$2' shifts=$3 percol=$4 had=$5 4lvl=$6 $7)"
  for u in $USERS; do
    ( env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_STATE_QUANT_SCOPE="$2" RWKV_QUANT_SHIFTS="$3" \
        RWKV_LOWRANK_PERCOL="$4" RWKV_LOWRANK_HADAMARD="$5" RWKV_LOWRANK_4LEVEL="$6" $7 \
        $BIN $u >/dev/null 2>&1
      cp $PRED/rust_pred_${u}.json $PRED/rust_pred_${8}_${u}.json ) &
    # WORK QUEUE: keep <= NPROC workers running; refill a slot the INSTANT any worker finishes (no
    # per-wave barrier). Combined with LPT order above, this avoids idle threads waiting on a straggler.
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n 2>/dev/null || true; done
  done
  wait
}

# ---- baselines + current champion (always run). NOTE: bit budgets below are PER LAYER across BOTH heads
#      (per-head 16x16: rank-r factors = 2*16*r/head * 2 heads). i4r1 = 256 bits/layer (the int2 budget).
pass ""                          "" 0 0 0 0 "" fp32   # uncompressed reference
pass "card:2:int4,note:2:int4"   "" 1 1 0 0 "" i4     # rank-2 int4 (~512 bits/layer) -- current deploy / THE BAR
pass "card:2:int2,note:2:int2"   "" 1 1 0 0 "" i2     # rank-2 int2 (256 bits/layer) -- the "dies" baseline.
# NOTE: int2's recurrence can transiently blow up (->inf) on some fragile power users; preds are sanitized
# to 0.5 (san() in main.rs) so scoring never crashes, but this INFLATES i2's fast-path mean -- it's a
# throwaway reference (known to die), so that's fine. For its stable value use RWKV_USE_CANDLE=1.
pass "card:1:int4,note:1:int4"   "" 1 1 0 0 "" i4r1   # ★ CHAMPION: rank-1 int4 (256 bits/layer)

# ---- CANDIDATES: add schemes here; put tags in CANDIDATE_TAGS. Engine flags available (H=1/K=32 phase
#      findings, re-validate on 16x16): RWKV_LOWRANK_COMPAND=<p>, RWKV_LOWRANK_VLEVEL=intN (asym U/V),
#      RWKV_LOWRANK_ALS=<n>, RWKV_LOWRANK_MIXED53=1. All judged ONLY by the log-loss below.
# Supplementary VAL sweep — fill the last master-table cells for the distinct deployable approaches.
#   i4ef       rank-2 int4 + FULL e   (EF ceiling on val)
#   pqm2b8     PQ alone (no stabilizer -- expect blow-up, the ≤256-b in-budget-but-fails point)
#   pqm2b8epq  PQ + PQ'd rank-1 e     (240-b in-budget attempt)
PQ8=scratchpad/pq_cb_m2b8.txt
pass "card:2:int4,note:2:int4" "" 1 1 0 0 "RWKV_LOWRANK_EF=1" i4ef
pass "card:2:int4,note:2:int4" "" 1 1 0 0 "RWKV_LOWRANK_PQ=$PQ8" pqm2b8
pass "card:2:int4,note:2:int4" "" 1 1 0 0 "RWKV_LOWRANK_PQ=$PQ8 RWKV_LOWRANK_EF=1 RWKV_EF_ERANK=1 RWKV_EF_PQ=1" pqm2b8epq
CANDIDATE_TAGS="i4ef pqm2b8 pqm2b8epq"

echo "=== SCORE ($MODE; penalty = scheme - fp32; WIN = penalty <= +0.0025 in BOTH imm AND ahead at <=256 bits) ==="
RWKV_TRACE_DIR=$REF RWKV_PRED_DIR=$PRED $PY score.py scratchpad/active_users.txt fp32 i4 i2 i4r1 $CANDIDATE_TAGS
echo "RUN_EVAL_DONE"
