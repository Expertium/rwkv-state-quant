# rwkv-state-quant — card/note state compression autoresearch (LOG-LOSS loop)

You are a **standalone autoresearch agent** with one job: find a way to compress an RWKV-7 net's
per-card / per-note recurrent state to a **256-bit budget** (the "int2" budget) **without hurting
recall-prediction accuracy**, measured by **real log-loss** — not by any reconstruction proxy.

CPU-only. Everything you need is in this folder. You do NOT touch the parent project
(`../rwkv-anki-autoresearch`) — a winning scheme gets ported back there by its owner.

**Compaction timing (soft rule, Andrew 2026-07-05): compact at ~50–70% context fullness** — long-context
rot degrades quality well before hard limits. But NOT mid-task: finish the current unit of work (verdict
recorded, build landed, chain launched), request compaction at that seam, then continue.

## ★ TRANSITION COMPLETE (2026-06-30) — read first. New phase log = `research_log_h2k16.md`.

The parent repo changed the model + dataset; this is now the active setup:
- **Model → `champ_h2k16` (H=2 / K=16).** `reference/champ_h2k16.safetensors` is in place (the prior
  `champ_decay15` H=1/K=32 also remains for the old phase). d_model C=32 = 2 heads × 16; per-layer WKV
  state = **two 16×16 per-head matrices = 512 floats**, HALF the old single 32×32 (1024). Layers/stream:
  card 1, deck 4, note 3, preset 3, user 3. Engine **auto-derives H/K/C from the loaded weights**.
  **The reshape HALVED the raw state but NOT the low-rank deploy size** (2×(16×16) has the same 2·K·r
  factor count as one 32×32) — so the 256-bit budget & per-card/note byte sizes carry over; what must be
  re-measured is whether the i4r1 *penalty* is still ~free on the new geometry.
- **Dataset → 400+400 (this phase).** `reference_big/` landed 836 of users 6000–6999. The **36 LARGEST**
  users (by review count; 6810=3.1 GB … 6944=240 MB) were **DELETED** (17 GB; ids recorded in
  `excluded36.txt`) to speed runs → **DEV = 400 (ids 6000–6435, `dev_users.txt`)** + **VAL = 400 (ids
  6436–6999, `val_users.txt`)**, exactly 800 trace pairs remain in `reference_big/`. The
  engine + `score.py` read traces/preds from `reference_big/` via `RWKV_TRACE_DIR` (set in `run_eval.sh`).
- **Compute → 10 CPU threads** for everything (`run_eval.sh` NPROC default = 10; each proc RAYON/OMP=1).
- **Runs are launched DETACHED (WMI)** so they survive Esc — `scratchpad/detach.ps1` +
  `scratchpad/run_eval_detached.cmd`, poll `scratchpad/eval.log` for `RUN_EVAL_DONE`/`DONE_EXIT`.
- **Scoring runs on the FAST engine by default** (`engine/src/fast.rs`, plain-Rust f32, B=1) — **~6.7×**
  faster than the old candle path (5.5× forward + ~1.2× from a rank-1 power-iteration SVD shortcut). The
  per-step state compression (`compress_wkv_state`) is SHARED between the candle and fast paths, so results
  are identical to ~1e-5 (below the gate). Set `RWKV_USE_CANDLE=1` for the candle path (parity A/B).

## The problem

The net schedules Anki spaced-repetition reviews. To ship inside Anki it stores one small recurrent
"WKV state" **per card** and **per note** (per stream layer: H per-head K×K matrices + two C-vectors;
currently **two 16×16 matrices + two 32-vectors** at H=2/K=16 — was one 32×32, see the transition note).
State size dominates deploy memory (a power user has ~1M cards and ~0.9M notes). The deploy compresses
each WKV matrix as **low-rank factors with quantized entries**:

| scheme | state bits (WKV, per layer) | by-user log-loss penalty vs fp32 | status |
|---|---|---|---|
| **int4** rank-2 factors | ~512 bits (card 96 B, note 288 B) | **+0.0005 / +0.0002** (old H=1/K=32) | current deploy — **FREE** |
| **int2** rank-2 factors (per-column) | ~256 bits (card 48 B, note 144 B) | **+0.014 / +0.012** (old) | half the size, **"dies"** |
| **rank-1 int4 (`i4r1`)** | **256 bits** (card 64 B, note 192 B) | **+0.0017 / −0.0001** (old) — **WON** | champion; re-confirm on h2k16 |

> **⚠ Penalty numbers above are from the OLD H=1/K=32 phase** (`research_log.md`). The per-head **16×16**
> bit/byte accounting is re-derived (it matches the old per-layer numbers — same 2·K·r factor count), but
> the **penalties are being re-measured on `champ_h2k16` + the 400+400 set** → see `research_log_h2k16.md`.

**Your goal:** get a **≤256-bit** scheme down toward int4's ~free penalty. That halves the dominant
note-state memory. Or prove, cleanly, that 256 bits fundamentally cannot do it (a useful negative result).

### ★ ACCEPTANCE GATE (a "win")
A scheme **wins** iff its log-loss penalty vs fp32 is **≤ +0.0025 in BOTH modes** (imm AND ahead) at
**≤256 state bits/layer** (across both heads). (Reference: int4 ≈ +0.0005 — passes but 512 bits; int2 ≈
+0.014 — fails; i4r1 rank-1 int4 ≈ +0.0017 at 256 bits — won on the old model.) Both modes must pass.

### ★ DEV / VAL split (400 + 400 users) + measurement methodology
DEV = `dev_users.txt` (ids 6000–6435), VAL = `val_users.txt` (ids 6436–6999), 400 each (36 largest deleted;
ids in `excluded36.txt`). **Routing by whether the scheme has empirically-tuned parameters:**
1. **No tuned params → skip dev, measure DIRECTLY on VAL** (fp32/i4/i2/i4r1 are all parameter-free).
2. **Has tuned params → tune on DEV, then measure on VAL.** A handful of fitted constants won't generalize.
3. **Reported tables (research_log_h2k16.md) hold VALIDATION numbers ONLY.** Dev is scratch for tuning.
A scheme WINS iff its VAL penalty ≤ +0.0025 in BOTH modes at ≤256 bits (and is robust per-user — no user wrecked).
- `bash run_eval.sh [NPROC] dev|val|both`. (Traces in `reference_big/` via `RWKV_TRACE_DIR`.)

### Mathematically-clean approaches AND heuristics are both fair game
Use whatever works — a principled method (optimal/Lloyd–Max quantizers, ALS, low-rank theory) OR a
heuristic (special-casing zeros, hand-tuned codebooks, clamping rules). **But beware overfitting:**
- Prefer schemes whose parameters are derived **per-matrix from that matrix's own statistics** (data-
  driven), NOT global magic constants you hand-tuned until the dev number dropped. A handful of fitted
  constants will not generalize — and the val set will catch it.
- Watch the **per-user spread**, not just the mean — a scheme that helps most users and wrecks a few is
  fragile. The current winner `i4r1` (rank-1 int4) is robust precisely because it has *zero* fitted
  constants ("rank 1, 4 bits").

## ★ THE CRITICAL LESSON (why this is a log-loss loop, not a numpy one)

We first tried minimizing **Frobenius reconstruction error** `‖Â−A‖`. **It is ANTI-CORRELATED with
log-loss here.** A "4-level 2-bit" scheme had the *best* int2 Frobenius error (0.75 vs 0.97) but the
*worst* log-loss (+0.046, 3× worse than plain int2) — because its grid had no exact zero, biasing the
many near-zero state entries, and that bias compounds through the recurrence. **So: judge schemes ONLY
by `run_eval.sh` log-loss. There is no reliable fast proxy.** You may prototype the *math* of a scheme
in Python (e.g. dump states with `engine/target/release/rwkv-infer.exe --dump-card-corpus <user>`), but
NEVER decide a scheme is good from reconstruction error — always confirm in log-loss.

## How to run

```bash
bash run_eval.sh [NPROC] [dev|val|both]   # default: 10 dev. dev=6000-6435, val=6436-6999 (400 each), both=800.
```
Builds the engine, runs fp32 + int4 + int2 baselines + the i4r1 champion + your candidates over the chosen
user set, scores via `score.py scratchpad/active_users.txt fp32 i4 i2 i4r1 <tags>`. Output = by-user-mean
imm/ahead log-loss + penalty (scheme − fp32). Needs `python` with **numpy + scikit-learn**; engine builds
with `cargo` (deps vendored; offline OK). **Launch DETACHED so it survives Esc** (it's a long run — 400
users × several schemes):
```bash
powershell -NoProfile -File scratchpad/detach.ps1 -Script "C:\Users\Andrew\rwkv-state-quant\scratchpad\run_eval_detached.cmd" -ArgList "10 dev"
# then poll: until grep -q DONE_EXIT scratchpad/eval.log; do sleep 5; done   (scores in the SCORE block)
```

## How to add a scheme

The per-step state compression lives in **`engine/src/model.rs::lowrank_roundtrip`** (and the quantizer
helpers `quant_factor_*`, `quant_codes`). It is called every review step on each compressed stream's
WKV state. To try something new:
1. Implement it in `lowrank_roundtrip` / a new helper, gated by a new env flag (copy the pattern of the
   existing `RWKV_LOWRANK_PERCOL` / `RWKV_LOWRANK_HADAMARD` / `RWKV_LOWRANK_4LEVEL` flags — env read in
   `Model::load`, struct field, threaded to the call site).
2. Add a `pass ... <your_tag>` line in `run_eval.sh` and put the tag in `CANDIDATE_TAGS`.
3. Run (detached) → read the penalty. Append the result (win or dead-end) to **`research_log_h2k16.md`**
   (this phase's log; the old H=1/K=32 results are in `research_log.md`).

Existing engine flags (from the H=1/K=32 phase; re-validate on 16×16): `RWKV_LOWRANK_PERCOL` (on in
baselines, keep), `RWKV_LOWRANK_COMPAND=<p>` (companded quant), `RWKV_LOWRANK_VLEVEL=intN` (asym U/V bits),
`RWKV_LOWRANK_ALS=<n>` (direct factor opt), `RWKV_LOWRANK_HADAMARD`, `RWKV_LOWRANK_4LEVEL`, `RWKV_LOWRANK_MIXED53`.

Honest **≤256-bit/layer** accounting (per-head 16×16): rank-r factors = 2·16·r per head × 2 heads (rank-2 =
128 values = 256 bits at int2, 512 at int4; **rank-1 int4 = 2·16·1·2 = 64 values × 4 = 256 bits = the
champion**). Per-column scales / tiny shared codebooks are O(1) and amortizable; count anything per-matrix you add.

## What's been tried (don't redo — full numbers in research_log.md)

- **Per-column scaling** — each rank component its own scale. ON in the int2 baseline (it's what makes
  int2 +0.014 instead of catastrophic +0.051). Keep it.
- **4-level 2-bit (`RWKV_LOWRANK_4LEVEL`)** — DEAD: +0.046 log-loss (worse), despite better Frobenius.
- **Hadamard / incoherence rotation (`RWKV_LOWRANK_HADAMARD`, QuIP#/QuaRot)** — neutral: no help; cancels
  4-level's harm but adds nothing over plain int2. Don't pursue.

## Promising / untested ideas (H=1/K=32 phase resolved most; re-validate the winner on 16×16)

- **rank-1 int4 (`i4r1`) WON the rank/bit trade** (256 bits, +0.0017/−0.0001 on the old model). Re-confirm
  on `champ_h2k16` — the immediate task. Tried-and-rejected on the old model (full numbers in
  `research_log.md`): companding (noise/non-robust), packed 5×3 mixed53 (inflates near-zero), 224b asym
  (fragile), ALS direct-factor-opt (noise at int4, can't fix sub-256b fragility).
- **Vector / product quantization vs a corpus-wide codebook** — the main untried lever; amortizes bits →
  potential sub-256b. HIGH overfit-risk → fit on a train-user pool, validate on held-out users (the 400+400
  split + bigger dataset make this feasible now). The codebook is global/fixed (robust to card create/delete).

## Caveats

- **400 dev + 400 val users (6000-6435 / 6436-6999), in `reference_big/`** (the 36 largest were deleted,
  `excluded36.txt`, not evaluated). A winning scheme must help on the aggregate AND be **robust per-user** (no
  user wrecked) on BOTH sets — the per-step re-SVD can produce non-finite components on some power users at
  int2 (NaN-safe sort fix in `lowrank_roundtrip`; if you change the factorization, keep it robust).
- The eval **already models the recurrence** (state re-compressed every step) — that's why it caught the
  4-level bias the single-shot proxy missed. Trust it over your intuition about matrix error.
- `python_fp32` printed by the scorer from the traces may be stale (the traces are weight-independent
  inputs reused across champions) — ignore it; compare `rust_*` rows to `rust_fp32`.

## Files
- `CLAUDE.md` (this), `run_eval.sh` (the loop), `score.py` (scorer, reads a users-file + `RWKV_TRACE_DIR`).
- `research_log_h2k16.md` = **THIS phase's log** (H=2/K=16). `research_log.md` = old H=1/K=32 phase.
- `dev_users.txt` / `val_users.txt` = the 400+400 split (ids). `excluded36.txt` = ids of the 36 largest (DELETED).
- `engine/` — your OWN copy of the Rust inference crate (edit `src/model.rs`; `cargo build --release`).
  `src/fast.rs` = plain-Rust fast forward (`--verify-fast`/`--bench-synth-fast`/`--bench-mt`).
- `scratchpad/detach.ps1` + `run_eval_detached.cmd` = Esc-proof detached launcher; `eval.log` = its output.
- **INPUT vs OUTPUT are SEPARATE folders** (engine reads `RWKV_TRACE_DIR`, writes `RWKV_PRED_DIR`):
  - `reference_big/` — INPUT eval set, **traces only**: 800 `trace_user_*.{json,safetensors}` (weight-
    independent inputs + labels). Read via `RWKV_TRACE_DIR=reference_big`. Never written to by runs.
  - `preds/` — OUTPUT: `rust_pred_{tag}_{u}.json` predictions, written via `RWKV_PRED_DIR=preds`. Scratch
    (regenerated every run); safe to wipe.
- `reference/` — `champ_h2k16.safetensors` (NEW champion) + `champ_decay15.safetensors` (old) + a few
  leftover `trace_user_*` for quick checks. (The 36 largest users were DELETED; ids in `excluded36.txt`.)
