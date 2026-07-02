# research log — card/note state compression (append-only)

Metric = by-user-mean equalized **log-loss penalty** vs fp32, via `run_eval.sh`
(`imm` = immediate-recall head, `ahead` = forgetting-curve head; BOTH matter). Lower = better. Budget =
**≤256 state bits/matrix** (the int2 budget). **WIN = penalty vs fp32 ≤ +0.0025 in BOTH modes** (imm AND
ahead) at ≤256 bits. Clean methods AND heuristics allowed.

> **★ 2026-06-30 — DATASET is now 500 dev + 500 val users** (parent-repo owner; superseded a briefly-
> started 50+50 export that was aborted at 19 users). Develop on DEV (`bash run_eval.sh 6 dev`, users
> 6000-6499); CONFIRM the final candidate on the held-out VAL set (`bash run_eval.sh 6 val`, users
> 6500-6999). A real win clears the gate on **BOTH**. Re-measure the baselines + the current winner
> **i4r1** on dev/val (the +0.0017/-0.0001 below was the original 17-user gate). Traces for 6000-6999 are
> in `reference_big/` (read via `RWKV_TRACE_DIR`; export may still be completing — missing users are
> skipped). **ALSO: the model changed to H=2/K=16** — per-head **16×16** state, not one 32×32, so the
> low-rank bit-budget needs re-derivation (see CLAUDE.md "TRANSITION IN PROGRESS" + "DEV / VAL split").

## Baselines (measured 2026-06-30, 17-user gate, champion = champ_decay15)
| scheme | state bits | imm penalty | ahead penalty | note |
|---|---|---|---|---|
| fp32 (uncompressed) | — | 0 | 0 | reference |
| **int4 rank-2 (THE BAR)** | ~512 | **+0.0005** | +0.0002 | current deploy — FREE |
| int2 rank-2 per-column | ~256 | **+0.0140** | +0.0124 | half size, "dies" — the number to beat |
| int2 shared-scale (no percol) | ~256 | +0.0512 | — | per-column is essential; keep it on |
| **★ rank-1 int4 (`i4r1`)** | **256** | **+0.0017** | **−0.0001** | **WIN** — same bits as int2, ~int4-free |
| rank-1 int8 (`i8r1`, diag) | ~512 | +0.0012 | −0.0003 | pure rank-1 truncation ceiling ≈ FREE |

## Tried — DEAD END (don't redo)
- **4-level 2-bit (`RWKV_LOWRANK_4LEVEL`)** — **+0.0461 imm / +0.0497 ahead** (3× WORSE than plain int2),
  even though its Frobenius reconstruction error was the *best* of the int2 variants (0.75 vs 0.97). Its
  grid {-1.5,-0.5,+0.5,+1.5}·scale has no exact zero → biases the many near-zero state entries → the bias
  compounds through the recurrence. **★ This is THE lesson: Frobenius error is anti-correlated with
  log-loss here. Judge only by log-loss; preserve an exact-zero code.**
- **Packed 5×3 mixed-resolution int2 (`RWKV_LOWRANK_MIXED53`)** — **+0.0443 imm / +0.0493 ahead** (as bad
  as 4-level!). rank-2, comp1 (dominant) at 5-level {-2..2}·(amax/2), comp2 at ternary; both KEEP an exact
  zero, so this *isn't* the no-zero bias. The killer: the 5-level grid lowers the zeroing threshold to
  amax/4 and snaps small entries (|x|∈[0.25,0.5]·amax) UP to amax/2 → **it inflates near-zero entries**.
  Plain ternary aggressively zeros 78% of entries; that aggressive sparsity is a FEATURE, not a bug.
  **★★ SHARPENED LESSON: the real driver is "keep small entries small / preserve sparsity," NOT level
  count.** int4 (+0.0005) beats int2 (+0.014) because its fine grid keeps small entries near zero — not
  merely because it has more levels. Any scheme that inflates near-zero entries dies, even with a zero code.
- **Hadamard / incoherence rotation (`RWKV_LOWRANK_HADAMARD`)** — neutral. +4-level+Hadamard = +0.0140
  (lands back at plain int2: Hadamard cancels 4-level's harm but adds nothing over per-column int2). Per-
  column scaling already absorbs the outlier issue rotation would address. Verified on real states
  (singular-vector coherence μ≈2.6). Don't pursue.

## Tried — KEEP
- **Per-column scaling (`RWKV_LOWRANK_PERCOL`)** — each rank component its own scale. Turns catastrophic
  shared-scale int2 (+0.051) into +0.014. On in the int2 baseline; build on it.

## Open / promising (untested in log-loss)
- ~~**rank-1 int4 vs rank-2 int2 (same 256 bits).**~~ **RESOLVED → rank-1 int4 WINS** (+0.0017 vs +0.014).
  This is the deliverable. See Findings.
- ~~**Mixed bit allocation (σ₁ column gets more bits).**~~ Tried (mixed53 packed 5×3, asym U4V3 224b):
  mixed53 DEAD (+0.044, inflates small entries); 224b asym FRAGILE. comp1 needs the whole 256b at int4.
- ~~**Non-uniform 2-bit / companding.**~~ Companding tried: NOISE at int4 (non-robust to p), HURTS below.
- ~~**Direct factor optimization (ALS with per-iter quantize).**~~ **TRIED** (`RWKV_LOWRANK_ALS`): real but
  limited — helps the mean at int3/rank-2-int2, but NOISE at the int4 win and can't fix sub-256b fragility.
  Doesn't move the deliverable. Full numbers in Findings.
- **Vector / product quantization vs a shared corpus codebook** — the one genuinely-different lever left;
  could amortize bits for a sub-256b win. HIGH overfit risk on the 17-user gate → needs held-out validation.

## Findings (write the deliverable here)

### ★ WIN — rank-1 int4 (`i4r1`): 256 bits, +0.0017 imm / −0.0001 ahead  (2026-06-30)
At a FIXED 256-bit budget, **rank-1 at int4 dominates rank-2 at int2** (+0.0017 vs +0.0140 imm — ~8×
better; ahead is actually negative). It clears the gate (≤+0.0025 BOTH modes) with margin via a
maximally-general mechanism (no fitted constants — just "rank 1, 4 bits"). Honest size: U(32×1)+V(32×1)
= 64 values × 4 bits = **256 bits**, identical deploy size to int2 rank-2 (card 48 B / note 144 B), and
it uses FEWER per-column scales (2 vs 4).

**Why it works (mechanism):** dumped 783 real card states (H=1, 32×32). The 2nd SVD component holds
**17% of Frobenius energy but ≈0 predictive signal** — `i8r1` (rank-1 int8, near-lossless comp1) is
+0.0012/−0.0003, i.e. dropping comp2 is essentially free in log-loss. int2 rank-2 keeps comp2 but is
forced to quantize BOTH components to coarse ternary; the win **drops the predictively-useless comp2 and
spends all 256 bits refining the dominant component to 4-bit resolution.** This is the "Frobenius ⊥
log-loss" lesson again: Frobenius energy ≠ predictive value.

**Robustness (17-user gate):** worst single-user +0.0066 imm (user 121) / +0.0042 ahead; 4 users mildly
over +0.0025; NO user wrecked (cf. int2's worst +0.057 / 14-of-17 over). Spread comparable to the
accepted int4-512 baseline (whose worst is +0.0033 imm / +0.0093 ahead). → general mechanism, should
survive re-validation on more users. Re-validate on the larger held-out set in the parent repo before ship.

### Frontier probes — rank-1 bit/quantizer sweep (2026-06-30)
Companding (`RWKV_LOWRANK_COMPAND=p`, signed power-law `sign(u)|u|^p`, p<1 ⇒ levels dense near zero,
keeps small entries small) + fewer-bit rank-1 (`int3`=192b, `int2`=128b) + asymmetric U/V (`VLEVEL`).

| scheme | bits | imm pen | ahead pen | verdict |
|---|---|---|---|---|
| int4 rank-2 (bar) | 512 | +0.00048 | +0.00023 | reference |
| **i4r1sq** rank-1 int4, sqrt p=.5 | **256** | **+0.00138** | **+0.00009** | **★ best @256b** (Pareto-improves i4r1) |
| i4r1 rank-1 int4 uniform | 256 | +0.00167 | −0.00011 | win, **zero constants** (safest) |
| i3r1 rank-1 int3 uniform | 192 | +0.00303 | +0.00063 | FAIL imm (+.0005 over) **& fragile** (u121 +0.0127, 8/17 over) |
| i3r1sq rank-1 int3 sqrt | 192 | +0.00418 | +0.00231 | FAIL — companding HURTS int3 |
| i2r1 rank-1 int2 uniform | 128 | +0.01215 | +0.00725 | FAIL (ternary on comp1 too coarse) |
| i2r1sq rank-1 int2 sqrt | 128 | +0.02501 | +0.02152 | FAIL hard (companded ternary inflates) |

**Learnings:**
- **Companding is a 256b-refiner, not a bit-saver.** Dense-near-zero HELPS when levels are plentiful
  (int4: +0.00167→+0.00138, ~17% lower imm) but HURTS when scarce (int3/int2): densifying near zero
  leaves too few levels for the large range → large entries coarsened → net loss. The benefit needs
  ≥ ~int4's level count.
- **Sub-256b is hard.** 192b uniform int3 misses imm by only +0.0005 on the mean but is fragile
  (one power user, 121, at +0.0127). Confirms rank-1 needs ~int4 resolution to keep small entries small.
- Per-user (i4r1sq vs i4r1): imm improves on most users; one wrinkle — u165 ahead +0.0037→+0.0065.
  Mean ahead stays ≈0. Need to confirm i4r1sq's gain is robust to p (not a fitted constant) → next batch.

### p-robustness + sub-256b probe (2026-06-30, batch 2) → frontier is settled
| scheme | bits | imm pen | ahead pen | worst-user imm | #>+.0025 | verdict |
|---|---|---|---|---|---|---|
| i4r1 rank-1 int4 uniform | 256 | +0.00167 | −0.00011 | +0.0066 (u121) | 4 | **★ ROBUST WIN** |
| i4r1 compand p=0.4 | 256 | +0.00178 | −0.00026 | — | — | ≈uniform |
| i4r1sq compand p=0.5 | 256 | +0.00138 | +0.00009 | — | — | dips, but… |
| i4r1 compand p=0.6 | 256 | +0.00190 | +0.00023 | — | — | ≈uniform |
| i43r1 asym U4V3 | 224 | +0.00245 | +0.00133 | +0.0103 (u121) | 8 | passes mean by .00004 — **FRAGILE** |
| i34r1 asym U3V4 | 224 | +0.00254 | +0.00097 | — | — | FAILS imm |

**Companding is NOISE at int4, REJECTED.** Response across p is non-monotonic: p=0.5 (+0.00138) dips below
uniform (+0.00167) but BOTH p=0.4 (+0.00178) and p=0.6 (+0.00190) are *worse* than uniform. A real
mechanism would vary smoothly; this is the 17-user gate rewarding a lucky constant — exactly the
"hand-tuned global constant won't generalize" trap. **Keep i4r1 uniform (zero constants).**

**Sub-256b is fragile, not reachable robustly.** 224b (i43r1) passes the *mean* by 0.00004 but wrecks
power users (u121 +0.0103, u107 ahead +0.011, 8/17 over). 192b/128b fail outright.

### ★★ CONCLUSION — rank-1 int4 @256b is at the sweet spot; here is WHY it can't be cheaply beaten
1. **comp2 is on-average predictively inert** → drop it (rank-1). But on hard power users (e.g. 121) comp2
   *does* carry signal (i8r1 rank-1 int8 is still +0.0035 on u121) — that's what caps i4r1's worst-user.
2. **comp1 needs ~int4 (15 levels) to keep small entries small** — the true log-loss driver. int3 (7
   levels) already degrades to +0.0030 and gets fragile; companding can't help (too few levels to spare).
3. **256 bits = exactly rank-1 int4** (64 vals × 4 bits). Refining comp1 to int4 consumes the *entire*
   budget, leaving no room for comp2. Any 256b scheme that keeps comp2 must drop comp1 below int4 — which
   costs more than comp2 buys (rank-2 int2 is +0.014; mixed53 +0.044). So no robust 256b scheme beats it.
4. Net: **i4r1 (256 bits, +0.00167 imm / −0.00011 ahead, zero tuned constants) is the deliverable.** It
   halves note-state memory (vs int4-512) at near-int4-free accuracy. Re-validate on the larger held-out
   user set in the parent repo (margin to +0.0025 is comfortable on the mean; spread ≈ accepted int4-512).

### Direct factor optimization / ALS (`RWKV_LOWRANK_ALS=n`) — TRIED 2026-06-30, does NOT move the win
Alternating least squares with a per-iter quantize step (5 iters): minimize ‖A − U Vᵀ‖_F over quantized
U,V, init from SVD factors, closed-form LS solve for one factor given the other (compensating the other's
quant error), re-quantize. Per-column uniform (keeps exact zero).

| scheme | bits | imm | ahead | vs post-hoc | per-user |
|---|---|---|---|---|---|
| als_i4r1 | 256 | +0.00158 | +0.00030 | i4r1 +0.00167 → **noise** | worst/nbad ≈ i4r1 (just reshuffles users) |
| als_i3r1 | 192 | +0.00250 | +0.00056 | i3r1 +0.00303 → −0.0005 | **still fragile**: u121 +0.0108, 8/17 over |
| als_i2r1 | 128 | +0.01253 | +0.00670 | i2r1 +0.01215 → no help | ternary: Frobenius sacrifices small entries |
| als_i2 (rank-2) | 256 | +0.01143 | +0.00794 | i2 +0.01401 → −0.0026 | helped but still ≫ gate, dominated by i4r1 |

**Nuance (refines the lesson):** the Frobenius⊥log-loss anti-correlation is **NOT absolute** — when the
grid keeps an exact zero, ALS's lower Frobenius DOES help log-loss at moderate depth (int3 −0.0005,
rank-2 int2 −0.0026). But: (1) at **int4** (the win) ALS is **noise** — i4r1 already sits near the rank-1
int8 ceiling, nothing left to optimize; (2) at **sub-256b** ALS lowers the *mean* but **cannot fix
per-user fragility** — every sub-256b scheme (uniform int3, ALS int3, 224b asym) wrecks the SAME power
user (121: +0.011–0.013, 8/17 over). The binding constraint is structural, not reconstruction-quality:
hard users need BOTH comp2 AND ~int4-resolution comp1, which 192 bits can't supply. → ALS doesn't change
the deliverable; **i4r1 @256b stands.**

### Untried (higher-risk, future): corpus-wide **vector/product quantization** of the dominant factor
(amortizes bits → potential sub-256b), and **per-matrix adaptive allocation by σ-ratio** (could target the
u121-type hard users). Both add overfit surface to the 17-user gate; pursue only with held-out validation.
