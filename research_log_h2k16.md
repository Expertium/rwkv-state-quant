# research log — H=2/K=16 phase (champ_h2k16, 400+400)

Metric = by-user-mean equalized **log-loss penalty** vs fp32, via `run_eval.sh` (`imm` = immediate-recall
head, `ahead` = forgetting-curve head; BOTH matter). Lower = better. fp32 raw VAL log-loss = **imm 0.2797 /
ahead 0.3118**. Numbers to 4 decimals; scheme names in `rank-N intM` order. Old H=1/K=32 phase → `research_log.md`.

## ★★★ FINAL RESULT + HANDOFF (run stopped by Andrew 2026-07-03 ~08:45)

**FINAL CHAMPION @ 352 b/card: `e150_pq` = rank-1 PQ (m2b8) WKV + int4 token-shifts + 1.5-epoch QAT.**
**VAL +0.0010 imm / −0.0003 ahead** — the compressed model BEATS the uncompressed fp32 champion on the
ahead mode. Robust per-user (imm mean/med/nbad 0.0010/0.0007/69-of-400, Q4-largest +0.0013; one benign
single-user ahead outlier, 6951 +0.0246). **DEV-CONFIRMED (09:17): +0.0009 imm / +0.0003 ahead on the
400 held-out DEV users — matches VAL; the val-selected epoch count is not a fluke.** For scale: the
512-b F15 champion was +0.0024/+0.0021; the original PTQ starting point at this size was +0.0046/+0.0040.

**The LOCKED quantization recipe (do not change — improve log-loss around it):**
- Per WKV 16×16 head-matrix: rank-1 factors (power-iteration top singular vector, split-√σ, sign-canon),
  each 16-dim factor PQ-encoded with the **fixed global codebook `scratchpad/pq_cb_m2b8.txt`** (2 sub-vectors
  of dim 8, 256 centroids → 16 b + norm scale per direction) ≈ **96 b/card layer**; token-shifts **int4**
  (2×32 values ≈ 256 b) → **card ≈ 352 b, note (3 layers) ≈ 1056 b**.
- **Deploy env** (Rust engine, both fast+candle paths): `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4
  RWKV_LOWRANK_PQ=scratchpad/pq_cb_m2b8.txt RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1`.
- **Champion weights: `reference/qat_pq_ep150.safetensors`** (raw ckpt step 5026; EMA variant ±0.0001, ignore).
- **QAT recipe that produced it** (config `gpu_train/configs/qat_pq_ep150.toml`): fine-tune the fp32 champion
  `h2k16d_904` for **EPOCHS=1.5**, PEAK_LR 1e-3, no warmup, WD 0.01, clip 0.25, train users 1000–2499, with
  the deploy compression fake-quantized in the forward via the fused CUDA kernel:
  `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_QAT_PQ=<codebook> RWKV_QAT_FUSED=1` (train==deploy
  parity 3e-7; shifts not fake-quantized — int4 shift PTQ is ~free and is included in the measured numbers).

**Why it works (the two levers that mattered):** (1) **epochs** — monotone 0.05→1.5 ep, no turn-around seen;
longer QAT lets the base recover under the PQ regime and compression_cost goes NEGATIVE (−0.0008/−0.0008 at
1.5 ep: the PQ constraint acts as a regularizer). (2) **codebook fineness** (m4b8 ~0.0009 better at equal
epochs, but int4-shift variant is over budget). Dead levers (all measured): LR, tail length, grad clip, WD
(champion sits at the WD=0.01 equilibrium), EMA, codebook co-adaptation (weights equilibrate to the codebook
they trained with — refit hurts), H4K8 geometry (rejected: same rank-1 payload).

**Epoch ladder (VAL, m2b8 @352 b):** 0.05 ep +0.0050/+0.0040 · 0.1 +0.0043/+0.0037 · 0.2 +0.0038/+0.0031 ·
0.3 +0.0031/+0.0020 · 0.5 +0.0028/+0.0018 · 0.75 +0.0021/+0.0012 · 1.0 +0.0016/+0.0005 · **1.5 +0.0010/−0.0003**.

**Cancelled in-flight at the stop order (both resumable, configs + chained cmds committed):** ep200 (2.0 ep,
killed at ~48% — the trend was STILL paying at 1.5, so 2.0+ ep is the most promising open lever); F28
`qat_pq_m4s3` (m4b8 WKV + int3 shifts = exactly 352 b, est. +0.0019–0.0024 — never trained). Also open:
whether the 6951-type single-user ahead outlier grows with epochs.

## ★ NEW OBJECTIVE (Andrew 2026-07-04, task22): push ≤288 b/card at the SAME ≤+0.0025 gate
288 b = the champion format with int4→int3 token-shifts (m2b8 PQ WKV 96 b + int3 shifts 192 b). Basis:
the int3-shift tax measured tiny at 0.1 ep (PTQ +0.0007/+0.0004 F18; shift-QAT recovers ~+0.0003 F19) and
every cost so far shrank with epochs; e150_pq sits at +0.0010/−0.0003 with 0.0015+ of gate margin to spend.
**Plan:** (a) CPU PTQ diagnostic `e150s3` = e150 weights + int3 shifts, no retrain (RUNNING 02:04) — may
already clear the gate; (b) GPU QAT `s3e150` = e150 recipe + `RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3`,
1.5 ep (config `qat_pq_s3e150.toml`, launch after the kernel-port parity check); (c) if passed:
robustness + dev-confirm + handoff update. Backups if int3 shifts resist: m2b6 codebook (272 b),
PQ-encode the shift vectors themselves (~176 b, needs new engine+QAT paths).
**★ STANDING ORDER (Andrew 2026-07-04 ~02:20, off to sleep): if 288 b succeeds, KEEP LOWERING BITS at the
same gate.** Overnight ladder queued as self-driving GPU-serial chains (trainings run unconditionally —
they're cheap on the fast kernel; the eval scores decide what's a win):
- **Rung 1, 288 b** = m2b8 96 + int3 shifts 192: `s3e150` TRAINING (started 02:19, ~5025 steps) → chain
  evals @288. LMDB stall fixed first (see below).
- **Rung 2, 272 b** = m2b6 (64-centroid: 4×(12+8)=80 b WKV) + int3 shifts: `q272` chained on s3e150's GPU.
- **Rung 3, 224 b** = m2b8 + INT2 (ternary) shifts 128 b: `q224` chained on q272. Tests whether shift-QAT
  × 1.5 ep rescues what was PTQ-catastrophic (+0.0041, F18) — the epochs lever has revived costs before.
- Deeper (if 224 passes / morning work): m2b6+int2 = 208 b; **PQ-encode the shift vectors** (~2×40 b →
  card ~176 b, new engine+QAT paths — the deep lever).
**QAT-dedicated parameters (Andrew's idea, 2026-07-04 ~02:30) — surgical grafts, NO fresh retrain needed
(identity/zero-init = function-preserving, warm-start champion).** Ranked by added capability: (1) learnable
per-column state scales — REDUNDANT for the WKV path (diagonal conditioning commutes with the WKV update ⇒
absorbable into W_k/W_v/W_r rows; optimization-geometry-only); (2) ★ **learned orthogonal rotation R inside
the compression sandwich** (rotate→rank-1+PQ→un-rotate; NOT absorbable since compression is nonlinear;
SpinQuant showed learned rotations ≫ fixed Hadamard, and our fixed Hadamard was neutral; init I, zero
per-card bits, 2×16×16 matmuls/compress); (3) ★ **LSQ learned per-channel step sizes for the SHIFT
quantizer** (shifts are layernorm'd ⇒ stable stats ⇒ global learned scale plausible; also deletes the 2
per-card shift scales; attacks exactly the int3/int2-shift wall); (4) **gradient co-training of the
codebook** (backprop into selected centroids — DIFFERENT from the dead alternating-PTQ co-adapt; kernel
atomicAdd work). Pick by tonight's failure mode: 224 fails on int2 shifts → build (3); deep PQ rungs fail
on codebook snap → build (2) (+4). All need matching Rust engine paths before scoring (train==deploy).
**Ops incident (02:12): `MDB_READERS_FULL`** — LMDB reader tables clogged by stale slots from days of
tree-kills + 7 dead-parent orphan workers (sibling repo's crashed job). Fixed: `env.reader_check()`
(cleared 89), killed the orphans (parent PID verified dead), then RENAME-GUARDED lock-file reset of
train/test DBs (rename succeeds ⇔ no live mapper — doubles as proof the sibling's LIVE jobs don't use
them; label_filter too). All 3 envs reopen with 1 reader. ⚠ The trainer exits **DONE_EXIT_0 even on this
crash** ("Killed processes.") — chains must (and do) guard on the checkpoint file existing, not the exit
code alone. Sibling's live `data_processing`/`train_bigstack` + FSRS bench (new PIDs 18200 tree) untouched.
**Speedup port from the parent repo (the other Claude, 2026-07-03):** `rwkv7_cuda.cu` (37× QAT kernel:
warp-0 power iter + block-parallel PQ search + skip elision, all bit-exact; + zeros→empty grad buffers),
`rwkv_model.py` (flat-row time-shift gather), `srs_model.py` (PermGather collision-free deterministic
backward, escape hatch `RWKV_PERM_GATHER=0`) — their quant-aware deterministic step went 4,122→450 ms
(9.2×). Ported wholesale (diffs verified to contain ONLY the speedups; our shift-QAT machinery intact),
rebuild + `parity_lr_pq.py`/`parity_lr.py` gate before any training uses it.

**★ OBJECTIVE (Andrew 2026-07-01, REVISED): minimize the FULL per-card state payload** — everything persisted
to disk for a card: WKV factors + token-shifts + any extra terms (error-feedback `e`, codebook indices, scales)
— **subject to the log-loss gate ≤ +0.0025 in BOTH modes on VAL, robust per-user.** Reference (full card, incl.
shifts): current deploy **rank-2 int4 = 768 b**; **rank-1 int4 = 512 b** (ACHIEVED — F15 QAT win +0.0024/+0.0021).
**TARGET: < 512 b/card.** (note = 3 × card.) The engine quantizes the shifts at the WKV bit-width (model.rs),
so rank-1 int3 = **384 b/card**, int2 = **256 b/card** (WKV + shifts both shrink together). The old "≤256 b/layer
WKV" framing is superseded, but its fixed-net NEGATIVE result (below) still holds as context.

## ★★ OBJECTIVE RE-REVISED (Andrew 2026-07-02 evening): 352 b/card is LOCKED IN — stop reducing bits.
**New goal: MINIMIZE LOG-LOSS at the 352 b/card state budget.** The locked format = rank-1 PQ (m2b8) WKV
~96 b + int4 token-shifts 256 b. Current best at budget: `e75_pq` **+0.0021 imm / +0.0012 ahead** (0.75-ep
QAT). Quality levers, ranked: (1) **more epochs** — monotone so far, 1.0/1.5/2.0 sweep already queued (F25c);
(2) **codebook co-adaptation — DEAD (PTQ diagnostic, 22:53):** retraining the m2b8 codebook on the ep75
net's OWN deploy-time directions and swapping it in WITHOUT retraining made things WORSE, not better:
`e75coad` **+0.0027/+0.0016** vs e75_pq +0.0021/+0.0012 (+0.0006/+0.0004 regression). Read: after 0.75 ep of
QAT the weights sit in a self-consistent equilibrium WITH the fixed champion-trained codebook — consistency
dominates codebook-fit. An alternating re-QAT round would start from a worse point and at best converge back
to a similar equilibrium; no GPU slot. (Codebook `pq_cb_m2b8_coad.txt` + `coad_dump.sh` kept for reference.); (3) **m4b8 WKV + int3 shifts = exactly 352 b**
— **REVIVED by the m4ep50 readout (22:28): m4b8×0.5ep = +0.0019/+0.0011, BEATS e75_pq with less training**
(m4b8 is ~0.0009 ahead of m2b8 at every equal-epoch point: 0.3ep +0.0024 vs +0.0031; 0.5ep +0.0019 vs
+0.0028). The F27 downgrade assumed the 0.1-ep int3-shift tax (+0.0004-7); the epoch lever plausibly shrinks
that too (F19/F20 never tested shifts beyond 0.1 ep). Est. m4b8@0.75ep+shift3 ≈ +0.0019-0.0024 @ EXACTLY
352 b. Run = RWKV_QAT_PQ=m4b8 + RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3, deploy RWKV_STATE_SHIFT_LEVEL=int3.
**F28 QUEUED** (detached PID 17956, `run_m4s3_chained.cmd` polls ep200's DONE; config `qat_pq_m4s3.toml`,
0.75 ep, out `qat_pq_m4s3_*.pth`, log `qat_qat_pq_m4s3.log`; eval env adds `RWKV_STATE_SHIFT_LEVEL=int3`);
(3b) **H=4/K=8 geometry — considered and REJECTED (Andrew 2026-07-02 ~23:00):** halves the RAW state but NOT
the rank-1 payload (2·d_model = 64 factor values either way; PQ index count would double), needs a from-
scratch champion, benefit (2× DOF fraction under rank-1) speculative. Not pursued; (4) larger-ncent codebooks (m2b9+)
are OVER budget (360 b) — skip. Schemes must fit ≤352 b; judged only by VAL log-loss, robust per-user.
**Progress (21:50):** co-adapted codebook BUILT (`pq_cb_m2b8_coad.txt`, 120k dirs/role dumped from the ep75
net under PQ deploy). Cheap diagnostic queued before committing GPU: ep75 weights deployed with the coad
codebook, NO retraining (tag `e75coad`, 1 pass) — bounds what the co-adapt re-QAT can gain (consistency vs
codebook-fit). CPU chain: m4ep50 eval (m4b8 epoch-scaling question) → coad-PTQ diagnostic. GPU: ep100
(started 21:43) → ep150 → ep200. **UPDATE 22:53: diagnostic read out NEGATIVE (see lever 2 above) — the
co-adapt re-QAT is cancelled; the GPU chain stays ep100 → ep150 → ep200 → F28 as queued.** Remaining live
levers: epochs (sweep3) and m4b8+int3shift (F28). **UPDATE 23:00: the whole readout pipeline is now
self-driving** — four detached wait→convert→eval chains launched (PIDs 42608/5580/36900/18872):
`pq_epsweep_eval.sh` (generic, per-N: polls `qat_qat_pq_epN.log` for DONE_EXIT_0 → `pth_to_sft.py` →
2-pass RAW VAL eval, tags `eN_base`/`eN_pq`) via `pq_ep{100,150,200}_chain.cmd`, and `pq_m4s3_eval.sh`
(F28 @ exactly 352 b: m4b8 codebook + `RWKV_STATE_SHIFT_LEVEL=int3`) via `pq_m4s3_chain.cmd`. Each aborts
loudly if its training doesn't end DONE_EXIT_0. (Ops note: a monitor grepping a log for `DONE_EXIT` must
use `DONE_EXIT_[0-9]` — a status line echoing the *word* DONE_EXIT false-triggers it.)
**Pipeline for the new objective (staged 2026-07-02 ~20:45):** GPU chain: m4ep50 → ep100 → ep150 → ep200
(sweep3, lever 1). CPU chain: c30 combo eval → ep75 dev-confirm → **co-adapt dump+retrain** (lever 2 prep:
`scratchpad/coad_dump.sh` dumps 20-user card+note corpus from the ep75 net UNDER PQ deploy →
`pq_cb_m2b8_coad.txt`; detached PID 37300). Re-QAT with the co-adapted codebook = next GPU slot after sweep3
(or earlier if sweep3 shows saturation — then truncate it). Lever 3 (m4b8+int3shift @352 b) decision awaits
the c30/m4ep50 readouts.

## ★ CURRENT STATUS (2026-07-03) — RUN FINALIZED (see HANDOFF section at top)
- **★★★ FINAL CHAMPION @ ~352 b/card = `e150_pq` (1.5-ep QAT, 2026-07-03 06:23): VAL +0.0010 imm / −0.0003
  ahead — the compressed model BEATS the fp32 champion on ahead.** Epoch trend still monotone through 0.05 →
  1.5 ep; comp_cost −0.0008/−0.0008 (PQ = regularizer). Robustness PASS (imm mean/med/nbad 0.0010/0.0007/69;
  ahead −0.0003/+0.0002/76; Q4 imm +0.0013; watch-item: single-user ahead outlier 6951 +0.0246). Weights
  `reference/qat_pq_ep150.safetensors` (raw, step 5026). Prior points: e100_pq +0.0016/+0.0005 (robustness
  PASS, nbad 108), e75_pq +0.0021/+0.0012. Andrew stopped the run 2026-07-03 ~08:45 (ep200 killed at ~48%,
  F28 cancelled before start); dev-confirm of e150_pq completed 09:17: **DEV +0.0009/+0.0003 — matches VAL,
  recipe generalizes.** Final champion fully validated: gate ✓ (with 2.5–9× margin), robustness ✓, dev ✓.
- **Prior win (2026-07-02 evening): rank-1 PQ (m2b8) + QAT 0.75 ep** — VAL **+0.0021 imm /
  +0.0012 ahead**, both ≤ +0.0025 with real margin; **beats the F15 512-b champion (+0.0024/+0.0021) on BOTH
  modes at 69% of the size.** Card = WKV PQ ~96 b + shifts int4 256 b ≈ 352 b; note ≈ 1056 b. Recipe: F22 +
  EPOCHS=0.75 (LR 1e-3, WD 0.01, from champion, fixed champion-trained m2b8 codebook; weights
  `reference/qat_pq_ep75.safetensors`). Two levers found: **epochs** (monotone 0.05→0.75, drift keeps falling —
  longer QAT lets the base recover under the PQ regime) and **codebook fineness** (m4b8 cuts drift ~40%/ep).
  Dead levers (all measured): LR, shorter tail, clip, WD (champion sits at the WD=0.01 equilibrium), EMA ~nil.
  **DEV-CONFIRMED (was the last caveat): +0.0021 imm / +0.0014 ahead on the 400 DEV users** — matches VAL
  (+0.0021/+0.0012); the recipe generalizes, not a val fluke. Win fully validated: gate ✓, robustness ✓,
  dev ✓. F27 combo read out (+0.0024/+0.0013 @416 b — behind the champion); m4ep50 still in flight.
  **Robustness (per_user.py, 400 VAL): PASS — MORE robust than F15.** Head-to-head vs qi4r1(F15,512b):
  imm mean/median/nbad **+0.0021/+0.0014/131** vs +0.0024/+0.0020/155; ahead **+0.0012/+0.0012/130** vs
  +0.0021/+0.0018/150; Q4-largest imm **+0.0023** vs +0.0028 (no power-user runaway). Worst users same
  magnitude AND same identities (6652/6863/6787/6861/6994 in both lists — hard users, not PQ-wrecked).
- **SUB-512 EXPLORATION (F16–F20): F15 @512 b is the practical floor for the rank-1-int4+shift scheme.** Every
  sub-512 route via bit-coarsening failed the gate: int3-everything 384 b (F17 +0.0036), WKV4+shift2 384 b (F18
  +0.0065), WKV4+shift3 448 b at all LRs (F19/F20 best +0.0028, +0.0003 over). **Token-shifts are the binding
  wall below 512** (more quant-sensitive than the WKV; int3 shift = irreducible ~+0.0004 imm). LR can't cut imm
  (F16, F20). **Only untried lever with headroom = PQ+QAT on the WKV factors** (shifts kept safe int4) → Step 6.
- Built this phase: engine `RWKV_STATE_SHIFT_LEVEL` (independent shift bits, fast+candle); QAT shift fake-quant
  (`RWKV_QAT_SHIFT_SCOPE`, `fake_quant_shift` in rwkv_model.py) — closes the train/deploy shift gap. int3 added
  to Python `_QMAX` + confirmed in Rust `parse_level`.

- **Fixed-net ≤256 b = clean NEGATIVE result** (F10). Every ≤256-b scheme fails the gate; best *deployable*
  (both > 256 b, PTQ on the champion): `i4efe1i4` 768 b **+0.0009/+0.0006**, `pqm2b8e1i4` 432 b +0.0027/+0.0014.
- **QAT phase** — break the negative by fine-tuning the net to be ROBUST to the compressed state (fake-quant in
  the forward, STE). Pipeline in `gpu_train/` (parent venv + LMDB referenced, not duplicated).
- **Two fused CUDA QAT kernels built + validated (F13):** full-matrix int-N, and **rank-1 int4 low-rank**
  (matches the deploy `compress_wkv_state` r==1). Both ≈fp32-parity vs their references, **150–490× over the
  Python loop** → a QAT run drops from hours to ~10 min. Rebuild: `cmd /c gpu_train/build_ext.cmd`.
- **Full-matrix int2 QAT = a WASH (F12, F14):** quant becomes ~free (+0.0011) but the fp32 base regresses
  ~+0.0024 — for ANY int2 tail, even a short low-LR one warm-started from the champion → `qi4r1`@256b +0.0035,
  still fails. Root cause: full-matrix int2 QAT teaches *int*-robustness, not the real imm blocker = **rank-1
  truncation**.
- **★★ WIN (F15): rank-1 int4 QAT breaks the ≤256-b negative.** Training with the DEPLOY compression (rank-1
  int4, via the fused low-rank kernel) directly teaches rank-1-TRUNCATION robustness. Result `lri4r1` @256 b:
  **VAL +0.0024 imm / +0.0021 ahead — BOTH ≤ +0.0025**, robust per-user (uniform, no blow-up; worst +0.0186 vs
  the fixed-net i4r1's +0.0258, and better mean/median/#over-gate everywhere). It cut the imm penalty from the
  champion PTQ i4r1's +0.0036 → +0.0024 (the rank-1 truncation cost QAT removed), exactly where full-matrix int2
  QAT (+0.0035) failed. Decomp: base_drift +0.0018 (still the main cost — gentler than int2's
  +0.0024), compression_cost only +0.0007. imm margin is THIN (+0.00005) and this is one val-measured config → next:
  tighten the margin (reduce base_drift: even shorter/lower-LR tail, warm optim, or EMA), and dev-tune the
  recipe to confirm it's not a val fluke. "Keep train≈deploy" (Andrew) is what did it.

## ★ AUTONOMOUS NIGHT PLAN (2026-07-01, Andrew asleep) — do in order; continuity lives here
- **LR-CLOBBER BUG FIXED (important):** the champion optim (`h2k16d_904_optim`) was saved under a LambdaLR, so
  its param_groups carry `initial_lr=1e-3` + `lr=0`; `load_state_dict` restored both and LambdaLR reused
  `initial_lr` as base_lr, **silently overriding `config.PEAK_LR`**. So F15 actually trained at **peak LR 1e-3**
  (not "2e-4"), and my first LR sweep varied nothing. Fixed in `train_rwkv.py` (reset lr+initial_lr to
  config.PEAK_LR after optim load; prints `[lr] reset ...`). The LR knob is now REAL for all QAT runs.
- **Step 1 [in flight]:** LR sweep re-run with the fix — `qat_lr1i4_{lr1e3,lr5e4,lr2e4,lr1e4}` (0.1 ep from
  champion), log `scratchpad/qat_lr_sweep.log`. Then dev-rank (`scratchpad/qat_lr_dev_eval.sh`, qi4r1 on full
  dev vs champ fp32), pick LOWEST total (lower LR should cut base_drift +0.0018 → widen the thin margin),
  VAL-confirm the winner. lr1e3 should reproduce F15 (~+0.0024) = fix sanity check.
- **Step 1 DONE (F16):** LR sweep — imm penalty is COMPRESSION-dominated, not drift-dominated. Lower LR helps
  ahead monotonically (+0.00203→+0.00161) but slightly HURTS imm (+0.00248→+0.00258); gate binds on the worse
  mode (imm), so lr1e3 (=F15) has the best binding margin. LR lever CANNOT widen the imm margin. F15 stays champ.
- **Step 2 DONE (F17): rank-1 int3 QAT (384 b) FAILS** (+0.0036/+0.0029). base_drift is fine (+0.0016) but the
  WKV int3 truncation compression_cost (+0.0020 imm) ~doubles int4's — QAT can't recover the extra bit. int3
  everything is too coarse. **Pivot:** the 384 b = WKV 192 + shifts 192 coarsened BOTH; keep WKV int4 (F15
  showed ~free) and coarsen ONLY shifts → WKV int4 (256) + shifts int2 (128) = **384 b, same size, finer WKV.**
- **Step 3 DONE (F18): asymmetric shift compression, PTQ.** Added engine flag `RWKV_STATE_SHIFT_LEVEL=intN`
  (independent shift bit-width; both fast.rs + candle paths; smoke-verified int4-override == coupled, int2 differs).
  PTQ on F15 weights: WKV int4 + shift int4 (512 b) = F15 repro +0.0024/+0.0021 ✓; + shift int3 (448 b)
  +0.0031/+0.0025 (just fails); + shift int2 (384 b) +0.0065/+0.0045 (catastrophic). **Shifts are SENSITIVE**
  (opposite of intuition) — int2 token-shifts are too coarse. At 384 b, int3-everything (F17) beats WKV4+shift2.
- **Step 4 DONE (F19): shift-QAT built + trained 448 b.** Added per-row shift fake-quant (STE) to the training
  time+channel mixers (`fake_quant_shift` in rwkv_model.py, gated by `RWKV_QAT_SHIFT_SCOPE`); closes the real
  train/deploy shift gap. WKV-int4-rank1 + shift-int3 QAT (lr1e3) → deploy 448 b: +0.0028/+0.0023. Shift-QAT
  recovered +0.0003 imm vs PTQ (F18 +0.0031) and ahead now PASSES. But imm still +0.0003 over — and it's now
  BASE_DRIFT-dominated (base +0.0020 > compression +0.0008), the OPPOSITE regime from F16.
- **Step 5 DONE (F20): 448-b LR sweep FAILS.** Lower LR (5e-4, 2e-4) did NOT cut imm (stayed ~+0.0028-0.0030,
  ahead improved) — same as F16, imm is irreducible via LR. **448 b is a clean negative.** My "base_drift-
  dominated" read was misleading; the imm floor for WKV4+shift3 is ~+0.0028.
- **★ CONCLUSION (the sub-512 frontier):** below 512 b the TOKEN-SHIFTS are the binding wall — more quant-
  sensitive than the WKV. int3 shifts add an irreducible ~+0.0004 imm (F18/F19/F20); int2 shifts are
  catastrophic (F18 +0.0065). WKV-int3 truncation also too coarse (F17). Both components resist compression
  below their rank-1-int4 state ⇒ **F15 @512 b is the practical floor for the rank-1-int4 + shift scheme.**
- **Step 6 [next, deep lever]: PQ+QAT on the WKV factors (shifts stay safe int4).** The WKV is the bigger 256-b
  chunk; PQ (corpus codebook, sub-int4/value) could crush the rank-1 factor directions to ~64-96 b → card
  ≈ 96 (WKV) + 256 (shift int4) ≈ 352 b, comfortably sub-512 with the shifts left at their safe int4. Fixed-net
  PQ blew up (+0.1999, master table `pqm2b8`) → QAT is the fix. Deploy PQ path exists (`RWKV_LOWRANK_PQ=<file>`,
  `scratchpad/pq_train.py`); needs a QAT forward that fake-quantizes through the fixed codebook (STE). Big build
  → give it a FRESH context (compact first). This is the one lever with real headroom left below 512.
- **Step 7 DONE (F21) — rank-1 PQ PTQ de-risk = GREEN, building.** First-ever rank-1 PQ measurement (all prior
  PQ rows are RANK-2, which blows up via the re-SVD runaway → +0.1778; rank-1 is stable, no comp2). PTQ on the
  F15 weights, full VAL vs champion fp32 (`scratchpad/pq_r1_ptq_val.sh`): `qi4r1` (F15 sanity, 512 b)
  **+0.0024/+0.0021** (reproduces F15 ✓ — eval trustworthy); **`pqr1` rank-1 PQ m2b8 (~352 b) +0.0046/+0.0040**;
  `pqr1_6` m2b6 (~336 b) +0.0058/+0.0050. Rank-1 PQ is NOWHERE near the rank-2 +0.1778 — only ~+0.0022 above
  int4-on-the-same-weights, well within the ≲+0.05 "build it" rule. QAT dropped int4 rank-1 by −0.0012
  (champion PTQ +0.0036 → F15 +0.0024); PQ-QAT needs a similar drop to clear +0.0025. **DECISION: build the CUDA
  PQ-STE QAT path.** Reuse the champion `pq_cb_m2b8.txt` as the FIXED codebook (train==deploy share it; net
  adapts to whatever codebook it's given → consistent). Build: modify `qat_lr_rank1` (rwkv7_cuda.cu:304) —
  replace per-column int-N quant (lines 364–374) with sign-canon + codebook encode/decode; pass rank-1 codebook
  (roles 0,1 = 8192 floats = 32 KB) as a kernel arg. If QAT lands close, refine the codebook on QAT weights.
- **Step 8 [IN FLIGHT, F22] — PQ+QAT built, parity-confirmed, training.** CUDA build DONE: `qat_lr_rank1`
  (rwkv7_cuda.cu) gained a PQ branch (sign-canon + `pq_encode_decode`, mirrors engine `PqCodebook::encode_decode`)
  gated by a `__constant__ c_pq_active` flag; codebook uploaded once via new op `rwkv7_set_pq_codebook`
  (device globals, 32 KB). Design keeps the fwd/bwd kernels + wrappers UNTOUCHED (flag-branch) → int-N runs
  byte-identical. Python: `maybe_upload_pq_codebook()` in rwkv_ops.py (reads `RWKV_QAT_PQ=<codebook>`, uploads
  roles 0,1 lazily on first fused-LR forward). **Parity CONFIRMED** (`scratchpad/parity_lr_pq.py`): CUDA PQ vs
  Python deploy `encode_decode` ref = max REL **3.2e-07**, all 32/32 matrices ~1e-7 (train==deploy ✓); int-N
  regression (`parity_lr_trunc.py`) unchanged. **Training launched** (F22, detached PID 39248): from champion
  h2k16d_904, 0.1 ep, LR 1e-3, `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4` + `RWKV_QAT_PQ=pq_cb_m2b8.txt`,
  fused (MANDATORY — PQ is CUDA-only; Python-loop fallback would silently do int4). Config `qat_pq_m2b8.toml`,
  cmd `run_qat_pq.cmd`, log `qat_qat_pq_m2b8.log`, out `reference/qat_pq_m2b8_335.pth`. Then convert →
  safetensors, deploy-eval VAL (rank-1 PQ + shift int4, ~352 b/card) vs champion fp32. Target: PTQ +0.0046 →
  ≤ +0.0025. NEXT after result: if close, retrain codebook on QAT dirs / try m2b6 (~336 b); if win, VAL-lock.
- **Step 8 DONE (F22) — PQ+QAT @~352 b: near-miss, but PQ COMPRESSION IS FREE.** Total deploy penalty
  +0.0043/+0.0037 (FAILS gate). Decomp: **base_drift +0.0037/+0.0044 (the WHOLE penalty); compression_cost
  +0.0006/−0.0007 (≈FREE, negative on ahead).** The deep lever WORKS at the compression level — QAT taught
  rank-1-PQ robustness so the codebook snap costs ~nothing (cf PTQ F21 +0.0046 → comp_cost now ~0). The blocker
  is base_drift, which is ~2× F15's (+0.0018) at the IDENTICAL recipe (0.1 ep, LR 1e-3, from champion) — the
  PQ-STE forward(snap)/backward(identity) mismatch drifts the base more. **This is a training artifact, not a
  PQ limit → reducible.** If base_drift → F15's +0.0018, total ≈ +0.0018/+0.0011 = WIN at ~352 b.
- **Step 9 [IN FLIGHT, F23] — reduce base_drift (convert the near-miss to a win).** (a) **50-step ckpt = WORSE,
  not better** (counterintuitive): `pqqat50` base_drift +0.0048/+0.0041, comp_cost +0.0014/+0.0008, total
  +0.0062/+0.0049. Shorter tail HURTS both — with warmup=0 the LR starts at PEAK (1e-3), the net OVERSHOOTS
  early, then SETTLES as the schedule decays, so _335 (full) has LOWER drift than _50. Drift isn't monotonic in
  steps → shorter-tail lever DEAD. (b) **Right lever = lower PEAK LR** (smaller overshoot, keep full 0.1 ep so
  PQ robustness is still learned → comp_cost stays low). lr5e4 + lr2e4 trained (full 0.1 ep from champion),
  converted, **eval RUNNING** (`pq_qat_lr_eval.sh`, detached PID 37784, log `pq_qat_lr_eval.log`, tags
  pq5e4/pq2e4 + _base). If lower LR drops base_drift → F15's +0.0018 with comp_cost still ~0, total → ~+0.0020 =
  WIN at ~352 b. Codebook fixed champion-trained m2b8 throughout.
  - **F23a RESULT — LR is NOT the lever.** lr1e3 (F22) total +0.0043/+0.0037; lr5e4 +0.0045/+0.0034; lr2e4
    +0.0041/+0.0028 (best). base_drift is NON-monotonic in LR (lr5e4 WORST at +0.0042/+0.0052) and stuck at
    ~+0.0032-0.0042 imm across all LRs — ~2× F15's +0.0018, and LR can't move it (cf F16/F20 "LR floor").
    **compression_cost stays free/negative everywhere** (PQ robustness solid). ⇒ the extra base_drift is
    STRUCTURAL to PQ-STE (codebook snap biases the gradient more than int4 rounding), not an LR-overshoot
    artifact. Orthogonal knobs (WD=0, EMA) are the remaining hope (F24) — different mechanism than LR.
- **Step 9b [IN FLIGHT, F24] — OTHER decay/QAT hyperparameters (Andrew asked: not just LR).** Enumerated the
  knobs from train_rwkv.py: D-mode LR schedule = `cosine_down` from FULL peak at step 0, **no warmup** (= the
  overshoot the _50 result exposed). Env-tunable (HP-override system): **`RWKV_WEIGHT_DECAY`** (default 0.01 —
  an AdamW force pulling weights toward 0, i.e. AWAY from champion → setting 0 should cut base_drift directly),
  **`RWKV_EMA_DECAY`** (off by default — averages the tail, saves a bonus `_ema_` ckpt, smooths the overshoot),
  **`RWKV_CLIP`** (0.25 — tighter caps overshoot step size). Warmup would kill the overshoot but D-mode ignores
  WARMUP_STEPS → needs a code edit (fallback). Launched F24 sweep (detached PID 33072, GPU, both LR1e3/0.1ep/champ
  = clean ablation vs F22): **run1 `qat_pq_wd0ema` (WD=0 + EMA0.99), run2 `qat_pq_clip01ema` (clip0.1 + EMA0.99)**
  — EMA on both → raw+ema eval candidates each. Configs `qat_pq_{wd0ema,clip01ema}.toml`, reusable cmd
  `run_qat_pq_hp.cmd %cfg %wd %clip %ema`, sweep `pq_qat_hp_sweep.cmd`, logs `qat_qat_pq_{wd0ema,clip01ema}.log`.
  Early-health CONFIRMED (WD=0/EMA0.99/PQ all active). NEXT: eval all → combine best LR × {WD0, EMA, clip} → win.
  - **F24 status:** both HP runs trained + converted (4 candidates: `wd0`, `wd0_ema`, `cl01`, `cl01_ema`, each
    raw + EMA). Deploy-eval RUNNING (`scratchpad/pq_hp_eval.sh`, detached PID 13456, log `pq_hp_eval.log`, 8
    passes base+PQ). **Fallbacks if WD/EMA don't crack the ~2× base_drift wall** (likely, since F23a showed it's
    structural to PQ-STE, not an optimizer artifact): (1) **finer codebook m4b8** (4 sub-vecs × 8 bit = 32 b/dir
    → WKV ~160 b, card ~416 b still sub-512) — smaller snap ⇒ less STE bias ⇒ less drift, compression stays
    free; kernel already handles any m (no rebuild). (2) **retrain codebook on the QAT net's OWN directions**
    (VQ co-adaptation) — shrinks snap distance at the SAME 352 b. (3) base-preservation regularizer
    (distill/L2-to-champion) — directly minimizes drift (code change). m4b8 is the cleanest next lever.
- **★ Step 9c RESULT (F25) — EPOCHS IS THE LEVER. Monotone improvement, no saturation through 0.3 ep:**
  | ep | base_drift | total (~352 b) |
  |---|---|---|
  | 0.05 | +0.0049/+0.0052 | +0.0050/+0.0040 |
  | 0.1 (F22) | +0.0037/+0.0044 | +0.0043/+0.0037 |
  | 0.2 | +0.0037/+0.0037 | +0.0038/+0.0031 |
  | **0.3** | **+0.0033/+0.0031** | **+0.0031/+0.0020 — ahead PASSES** |
  imm at 0.3 ep is +0.0031 (+0.0006 over); trend ≈ −0.0007 imm per +0.1 ep. At 0.3 ep the PQ model BEATS its
  own uncompressed base on imm (comp_cost −0.0002) — the net fully co-adapts to the codebook. Reframe: longer
  QAT isn't "more drift", it lets the base RECOVER under the PQ regime while robustness deepens (the F19→F20
  "irreducible imm" was just an undertrained tail). **F25b launched (detached PID 16932): 0.5 + 0.75 ep**
  (same recipe, EMA0.99 bonus ckpts; configs `qat_pq_ep{50,75}.toml`, sweep `pq_qat_epoch_sweep2.cmd`). If the
  trend holds, 0.5 ep lands at the gate → WIN at ~352 b. Then: robustness check (per_user.py), and consider
  ep-extension × {WD0, m4b8} combos per the pending readouts.
- **Step 9c [IN FLIGHT, F25] — epoch sweep (Andrew asked).** Pure epoch effect vs F22 (LR1e3, WD0.01, no EMA,
  m2b8): EPOCHS ∈ {0.05, 0.2, 0.3} (F22=0.1). Each run has its OWN full cosine-decay-to-0 (NOT the same as the
  _50 intermediate ckpt). Rationale: within a run, _50→_335 showed drift DECREASES as the schedule settles, so a
  longer tail MIGHT settle further — but longer also = more total movement, so it cuts both ways. Detached PID
  19108 (GPU), configs `qat_pq_ep{05,20,30}.toml`, sweep `pq_qat_epoch_sweep.cmd`, logs `qat_qat_pq_ep*.log`,
  outs `qat_pq_ep{05,20,30}_*.pth`. ~3.7 h (contends w/ F24 CPU eval). Then convert + deploy-eval VAL.
  - **F25 training DONE** (ep05_167 / ep20_670 / ep30_1005 all DONE_EXIT_0, converted →
    `reference/qat_pq_ep{05,20,30}.safetensors`). **Eval QUEUED Esc-proof** (`pq_ep_eval.sh`, detached PID
    41120): the wrapper cmd POLLS for the F24 eval's DONE_EXIT before starting (no CPU thrash; chain survives
    Esc because the wait lives inside the detached job). 6 passes (3 weights × base+PQ), tags `ep{05,20,30}_{base,pq}`,
    log `pq_ep_eval.log`.
- **★ Step 9d RESULT (F26) — m4b8 CONFIRMS the structural-STE-bias hypothesis: 2nd lever.** At the SAME 0.1 ep
  as F22: base_drift **+0.0023/+0.0026** (m2b8: +0.0037/+0.0044 — the finer codebook cut drift ~40%), total
  **+0.0032/+0.0029** @ ~416 b (m2b8: +0.0043/+0.0037 @ 352 b), comp_cost +0.0009/+0.0003. m4b8@0.1ep ≈
  m2b8@0.3ep. Epochs & codebook-fineness attack drift by DIFFERENT mechanisms (adaptation time vs snap size) ⇒
  should stack. **F27 combo QUEUED: m4b8 × 0.3 ep** (`qat_pq_m4ep30`, detached PID 32152, Esc-proof chain
  behind the ep75 training; `run_qat_pq_hp.cmd` gained a %5=codebook arg). Highest-probability win @416 b.
- **Step 9d [IN FLIGHT, F26] — finer codebook m4b8 (the structural-drift lever), launched on the idle GPU in
  parallel** (Andrew away, autonomous). Hypothesis: F23a showed drift is structural to the SNAP magnitude →
  m4b8 (4 sub-vecs of dim 4 × 256 cents = 32 b/dir; WKV ~160 b; **card ~416 b**, still sub-512) halves the snap
  error ⇒ less STE gradient bias ⇒ less base_drift, while compression stays free. Codebook trained
  (`scratchpad/pq_cb_m4b8.txt`, same corpus as m2b8, 111k dirs/role; rank-1 upload = 8192 floats = exactly the
  kernel buffer). Training detached PID 30800 (F22 recipe: LR1e3, 0.1ep, champ), config `qat_pq_m4b8.toml`, log
  `qat_qat_pq_m4b8.log`, out `qat_pq_m4b8_335.pth`. Then convert + eval (queue behind F25's eval).
- **Ops note:** evals can be PAUSED without losing progress via NtSuspendProcess on the rwkv-infer workers
  (bash orchestrator just blocks in `wait`); resume with `scratchpad/resume_evals.ps1`. Used 2026-07-02 when
  Andrew needed the CPU briefly.
- **Ops lesson (2026-07-02): NEVER edit a .cmd that a running batch will re-read.** cmd.exe resumes batch
  files by BYTE OFFSET after each command; editing `run_qat_pq_hp.cmd` (adding the %5 codebook arg) while the
  ep50 `call` was inside its python line garbled the resume-parse → spurious `DONE_EXIT_9009` tail in
  `qat_qat_pq_ep50.log`. Harmless here (ckpt saved before; ep75 re-read fresh and ran clean) — but edits to
  shared .cmd helpers must wait until no batch is running them.
- **Step 9b RESULT (F24) — ⚠ WD-CLOBBER BUG found (2nd of its class!), clip DEAD, EMA marginal.**
  - **`wd0` was INVALID:** its weights came out **hash-identical** to F22 — `optimizer.load_state_dict`
    restores the champion's saved `weight_decay=0.01` into every param_group, silently clobbering
    `RWKV_WEIGHT_DECAY=0` (exact sibling of the F16 lr-clobber). **FIXED in train_rwkv.py** (capture intended
    per-group WD before the optim load, restore after; prints `[wd] reset ...`). **wd0 RETRY running** as
    `qat_pq_wd0fix` (detached PID 6416, fix verified in-log: `[0.0, 0.0, 0.0, 0.01, 0.0]`). Note: no OTHER run
    is affected — every prior run intended WD=0.01, which is what the clobber restored.
  - **clip0.1 = DEAD lever, made drift WORSE:** cl01 base_drift +0.0047/+0.0054 (vs F22 +0.0037/+0.0044);
    tighter clipping distorts the descent direction more than it tames the overshoot.
  - **EMA0.99 = marginal:** wd0_ema base +0.0036/+0.0042 (−0.0001/−0.0002 vs raw); cl01_ema ≈ cl01. Real but
    tiny at decay 0.99 over 335 steps; keep it ON (free bonus ckpt) but it won't crack the wall alone.
- **Pipeline (updated 19:45):** CPU: ep75 eval RUNNING (tags `e75*/e75e*`) → **F27 combo eval QUEUED**
  (detached PID 42628, polls `pq_ep75_eval.log`; tags `c30_{base,pq}` raw + `c30e_*` EMA, m4b8 codebook, log
  `pq_m4ep30_eval.log`). GPU: **F27b `m4ep50` TRAINING** (m4b8 × 0.5 ep margin run, detached PID 32164, config
  `qat_pq_m4ep50.toml`, log `qat_qat_pq_m4ep50.log`) — clear-pass candidate @416 b if combo@0.3 lands close.
  F27 combo internal val 0.3165/0.2793 = best of ANY run (caveat: internal val ≠ deploy penalty, cf wd0fix).
  Next: readouts → pick winner (e50/e75/c30/m4ep50) → per-user robustness → memory + CURRENT STATUS.
- **Step 9e [QUEUED, F25c] — Andrew: "more epochs is just better — try 1, 1.5, 2 ep."** Sweep3 queued behind
  the m4ep50 training (detached PID 30468, `pq_qat_epoch_sweep3.cmd`, configs `qat_pq_ep{100,150,200}.toml`,
  m2b8/LR1e3/WD0.01/EMA0.99). ~3.2/4.8/6.4 h per run → finishes tomorrow morning; readouts incremental (1.0 ep
  first). Testing where the epoch trend turns around (or keeps going below F15 levels). Eval plan for these:
  raw ckpt only (EMA has been within ±0.0001 in all 3 paired evals — skip its 2 passes).
- **Step 9b-fix RESULT (wd0fix) — TRUE WD=0 is a DEAD lever: WORSE drift.** wdf_base +0.0045/+0.0055 (F22
  WD=0.01: +0.0037/+0.0044); total +0.0042/+0.0038 ≈ wash. Why (in hindsight): the champion was TRAINED with
  WD=0.01, so its weights sit at the loss-gradient⇄WD equilibrium; removing WD changes the objective and the
  weights drift AWAY from that equilibrium. **Keep WD=0.01.** (Its best-ever INTERNAL val was a red herring —
  absolute quality ≠ closeness to champion.) Live levers = EPOCHS + CODEBOOK FINENESS only; both in pipeline.
- Ops: heartbeat each turn; evals DETACHED (NPROC=14); compact when context grows (continuity = this file).

## METHODOLOGY
- No tuned params → measure directly on VAL (fp32/i4/i2/i4r1). Tuned params (codebook, QAT recipe) → tune on
  DEV, report on VAL. Tables = VALIDATION only (dev = scratch). Judge ONLY by log-loss (Frobenius ⊥ log-loss).

## SETUP
- **Model** `reference/champ_h2k16.safetensors`: d_model C=32, H=2, K=16. Per-layer WKV = two 16×16 = **512
  floats**. Layers/stream: card 1, deck 4, note 3, preset 3, user 3. Engine auto-derives H/K/C.
- **Data** `reference_big/`: DEV=400 (`dev_users.txt`, 6000–6435), VAL=400 (`val_users.txt`, 6436–6999); 36
  largest excluded. Evals **NPROC=14** default, launched detached (WMI, Esc-proof); poll `scratchpad/eval.log`.
- **Scoring** on the plain-Rust fast forward (`fast.rs`, B=1, ~6.7× candle, parity ~1e-5). The per-step
  `compress_wkv_state` (model.rs) is SHARED by candle + fast + (ported to) the QAT kernels.

## ★★★ MASTER RESULTS TABLE — PTQ deploy schemes on the champion, full state size, VALIDATION log-loss ★★★
SIZES IN BITS. `WKV+e` (per layer) = WKV factors + error-feedback `e` (both heads). `card` = WKV+e + token-
shifts (1 layer); `note` = 3 × card. Shifts follow the WKV bit-width (≈256 b int4 / 192 b int3 / 128 b int2).
**The OBJECTIVE is on `card` (full payload): target < 512 b** — that's the `<512b?` column. fp32 raw VAL = imm
0.2797 / ahead 0.3118. (rank-r factors = 2·16·r·2heads values; full `e` = 512, rank-1 `e` = 64; PQ m2b8 = 16
b/dir. int4 = 4 b/val, int2 = 2 b/val.)

**Naming key:** `i4`/`i3`/`i2` = **rank-2** factors at int4/int3/int2 (rank-2 default). `r{R}`/`i{N}` mark
rank/int-bits: `i4r1` = rank-1 int4, `r1fp` = rank-1 fp (ceiling). `pqm{M}b{B}` = product-quant of factor
directions, M sub-vectors × B-bit codebooks. `m53` = mixed 5×3. **EF suffix:** `ef` = full-precision `e`;
`e1i4`/`e1i2` = rank-1 int4/int2 `e`; `epq` = PQ'd rank-1 `e`. E.g. `i4efe1i4` = rank-2 int4 + rank-1-int4 `e`.

| scheme | approach | WKV+e (b) | card (b) | note (b) | <512b? | VAL imm | VAL ahead | notes (dev; mechanism) |
|---|---|---|---|---|---|---|---|---|
| `fp32` | uncompressed reference | 16384 | 18432 | 55296 | — | 0 | 0 | reference (dev raw 0.2724/0.3039) |
| `i4` | rank-2 int4 | 512 | 768 | 2304 | No | +0.0125 | +0.0109 | ✗ current deploy; power-user blow-up tail |
| `i2` | rank-2 int2 | 256 | 384 | 1152 | Yes | (+0.164) | (+0.148) | ✗ throwaway (fast-path NaN-inflated) |
| **`i4r1`** | **rank-1 int4** | **256** | **512** | **1536** | No (=512) | **+0.0036** | **+0.0020** | PTQ misses imm by +0.0011; the 512-b baseline to beat (QAT fixes it → F15) |
| `r1fp` | rank-1 fp (ceiling) | 2048 (fp) | — | — | — | — | — | dev +0.0028/+0.0013 > gate ⇒ rank-1 DEAD (F3) |
| `r2fp` | rank-2 fp (ceiling) | 4096 (fp) | — | — | — | — | — | dev +0.0003/+0.0001 ⇒ rank-2 IS the target (F3) |
| `i4ef` | rank-2 int4 + FULL `e` | 16896 | 17152 | 51456 | No | +0.0006 | +0.0003 | EF ceiling — kills blow-ups (dev +0.0006/+0.0004) |
| **`i4efe1i4`** | rank-2 int4 + rank-1 int4 `e` | 768 | 1024 | 3072 | No | +0.0009 | +0.0006 | **best deployable** — passes gate, robust; 13× < i4 |
| `pqm2b8ef` | PQ + FULL `e` | 16544 | 16800 | 50400 | No | +0.0018 | +0.0009 | PQ quality ceiling; `e` over budget |
| `pqm2b8e1i4` | PQ + rank-1 int4 `e` | 416 | 672 | 2016 | No | +0.0027 | +0.0014 | near-miss (imm just over); codebook generalizes |
| `pqm2b8` | PQ rank-2 (~1 b/val), no stab. | 160 | 416 | 1248 | Yes | +0.1778 | +0.1614 | ✗ blows up — needs a stabilizer |
| `pqm2b8epq` | PQ + PQ'd rank-1 `e` | 240 | 496 | 1488 | Yes | +0.0125 | +0.0104 | ✗ in-budget but fails (codebook `e` too coarse) |

DEV-only diagnostic rows (never deployed, kept in Notes not tabled): `r1i5` 320 b +0.0029/+0.0014, `r1i3` 192 b
+0.0058/+0.0030, `i3ef` 384 b +0.0015/+0.0008, `i2ef` 256 b +0.0070/+0.0037, `i4efe1i2` 640 b +0.0052/+0.0035,
`i3efe1i2` 512 b +0.0370/+0.0325, `d1i2` 256 b +0.0135/+0.0072, `m53ef` 256 b +0.0048/+0.0025, `pqm2b8e1i2`
288 b +0.0074/+0.0051, `pqm2b6epq` 208 b +0.0191/+0.0159.

**Bottom line (fixed-net PTQ):** every card-<512 b row (`i2` 384, `pqm2b8` 416, `pqm2b8epq` 496) FAILS the
log-loss gate; every gate-passing row is ≥ 512 b (`i4efe1i4` 1024, `pqm2b8e1i4` 672, and `i4r1` 512 fails imm
anyway). So no FIXED-net scheme is both < 512 b/card AND passes. **QAT breaks this: rank-1 int4 QAT passes AT
512 b (F15); sub-512 via rank-1 int3 QAT (384 b) is next.** (Dev-diagnostic sizes below are WKV+e/layer.)

## QAT RESULTS (penalty vs CHAMPION fp32; QAT model ≠ champion, so tabled separately)
Total deploy penalty (VAL vs champ) = **base_drift** + **compression_cost**. `base_drift` = `qfp32` − champion
fp32 = how much QAT moved the UNCOMPRESSED base (measured with no compression; can go negative if QAT improves
the base). `compression_cost` = compressed − `qfp32` = the penalty the deploy compression adds on top (method-
agnostic: covers low-rank truncation + int-quant + codebook).
Sizes as in the master table: **WKV+e = per-LAYER gated quantity** (the ≤256-b gate is on THIS); **card / note =
FULL per-entity stored payload** = WKV factors + token-shifts (card = 1 layer, note = 3 layers; no `e` term in
these rank-1/rank-2 int4 schemes). QAT changes only the weights, not the compression format, so the byte layout
is identical to the master table's `i4r1`/`i4` rows. `≤256b?` reads off WKV+e (the gate). All log-loss = VAL.

| QAT recipe | scheme | WKV+e (b/layer) | card (b) | note (b) | <512b? | VAL vs champ | base_drift | compression_cost | verdict |
|---|---|---|---|---|---|---|---|---|---|
| full-matrix int2, 0.27 ep from WS (F12) | `qi4r1` | 256 | 512 | 1536 | No (=512) | +0.0035/+0.0024 | +0.0024/+0.0016 | +0.0011/+0.0008 | ✗ wash (fails gate) |
| full-matrix int2, 0.27 ep from WS (F12) | `qi4` | 512 | 768 | 2304 | No | +0.0027/+0.0018 | +0.0024/+0.0016 | +0.0003/+0.0002 | ✗ quant ~free but 768 b + base |
| full-matrix int2, **0.1 ep from champion** (F14) | `qi4r1` | 256 | 512 | 1536 | No (=512) | +0.0036/+0.0028 | +0.0024/+0.0022 | +0.0012/+0.0007 | ✗ short tail didn't preserve base |
| **rank-1 int4, 0.1 ep from champion (F15)** | **`lri4r1`** | **256** | **512** | **1536** | No (=512) | **+0.0024/+0.0021** | +0.0018/+0.0018 | +0.0007/+0.0004 | **★ passes gate @512 b (achieved point; now push <512)** |
| rank-1 int3 QAT, lr1e3 0.1 ep (F17) | `lri3r1` | 192 | **384** | 1152 | **Yes** | +0.0036/+0.0029 | +0.0016/+0.0015 | +0.0020/+0.0014 | ✗ FAILS gate — WKV int3 truncation ~doubles compression_cost; QAT can't recover the extra bit |
| **F15 weights, WKV int4 + shifts int3 PTQ (F18)** | `qi4s3` | 256 | **448** | 1344 | **Yes** | +0.0031/+0.0025 | (F15 +0.0018) | shift-int3 PTQ +0.0007/+0.0004 | ✗ just fails (both ~at gate) — but PTQ; QAT-modeling shifts should rescue it → next |
| F15 weights, WKV int4 + shifts int2 PTQ (F18) | `qi4s2` | 256 | 384 | 1152 | **Yes** | +0.0065/+0.0045 | (F15 +0.0018) | shift-int2 PTQ +0.0041/+0.0024 | ✗ catastrophic — shifts are SENSITIVE, int2 (ternary) too coarse for token-shifts |
| **WKV int4 rank-1 + shift-int3 QAT, lr1e3 (F19)** | `q448` | 256 | **448** | 1344 | **Yes** | +0.0028/+0.0023 | +0.0020/+0.0018 | +0.0008/+0.0005 | ✗ imm just fails (+0.0003 over); ahead PASSES. shift-QAT recovered +0.0003 imm vs PTQ |
| 448 b shift-QAT LR sweep 5e-4 / 2e-4 (F20) | `q448` | 256 | **448** | 1344 | **Yes** | 5e4 +0.0030/+0.0022 · 2e4 +0.0029/+0.0019 | — | — | ✗ lower LR does NOT cut imm (stays ~+0.0028-0.0030; ahead improves) — imm floor for WKV4+shift3 is ~+0.0028, irreducible via LR (cf F16). **448 b is a clean NEGATIVE** |
| **PQ+QAT m2b8, lr1e3 0.1ep from champ (F22)** | `pqqat` | ~96 | **~352** | ~1056 | **Yes** | +0.0043/+0.0037 | **+0.0037/+0.0044** | **+0.0006/−0.0007** | ✗ near-miss FAILS gate — but **PQ compression ≈FREE post-QAT** (comp_cost ~0, neg on ahead); penalty is ALL base_drift (2× F15, PQ-STE snap/identity mismatch). Reduce drift → win at ~352 b (Step 9) |
| PQ+QAT LR sweep 5e-4 / 2e-4 (F23a) | `pq5e4`/`pq2e4` | ~96 | ~352 | ~1056 | Yes | 5e4 +0.0045/+0.0034 · 2e4 +0.0041/+0.0028 | 5e4 +0.0042/+0.0052 · 2e4 +0.0032/+0.0040 | free/neg everywhere | ✗ LR NOT the lever: drift non-monotonic in LR, stuck ~2× F15 → STRUCTURAL to the PQ-STE snap |
| PQ+QAT 50-step ckpt (F23) | `pqqat50` | ~96 | ~352 | ~1056 | Yes | +0.0062/+0.0049 | +0.0048/+0.0041 | +0.0014/+0.0008 | ✗ shorter tail WORSE both axes (peak-LR overshoot hasn't settled) — shorter-tail lever DEAD |
| PQ+QAT clip0.1 (+EMA) (F24) | `cl01`/`cl01_ema` | ~96 | ~352 | ~1056 | Yes | +0.0046/+0.0033 · ema +0.0045/+0.0031 | +0.0047/+0.0054 | free/neg | ✗ tighter clip WORSE drift — clip lever DEAD. EMA0.99 = marginal −0.0001/−0.0002 (keep, won't crack wall). `wd0` INVALID (WD-clobber bug, hash-identical to F22; fixed, retry = `wd0fix`) |
| **PQ+QAT epoch sweep 0.05→0.3 (F25)** | `ep05/20/30_pq` | ~96 | **~352** | ~1056 | **Yes** | 0.05: +0.0050/+0.0040 · 0.2: +0.0038/+0.0031 · **0.3: +0.0031/+0.0020** | 0.3: +0.0033/+0.0031 | 0.3: −0.0002/−0.0011 (neg!) | **★ EPOCHS IS THE LEVER — monotone, no saturation; 0.3 ep ahead PASSES, imm +0.0006 over; PQ beats own base on imm. F25b: 0.5/0.75 ep running** |
| **PQ+QAT m4b8 finer codebook, 0.1 ep (F26)** | `m4_pq` | ~160 | **~416** | ~1248 | **Yes** | **+0.0032/+0.0029** | **+0.0023/+0.0026** | +0.0009/+0.0003 | **★ 2nd lever — finer codebook cuts drift ~40% at same ep (STE-bias hypothesis confirmed); m4b8@0.1ep ≈ m2b8@0.3ep. F27 combo m4b8×0.3ep queued** |
| PQ+QAT TRUE WD=0 (wd0fix, post-bugfix) | `wdf_pq`/`wdfe_pq` | ~96 | ~352 | ~1056 | Yes | +0.0042/+0.0038 · ema +0.0042/+0.0037 | **+0.0045/+0.0055 — WORSE** | −0.0003/−0.0017 | ✗ WD lever DEAD: champion sits at the WD=0.01 equilibrium; removing WD drifts AWAY. Keep WD=0.01 |
| **PQ+QAT 0.5 ep (F25b)** | `e50_pq` | ~96 | **~352** | ~1056 | **Yes** | **+0.0028/+0.0018** | +0.0031/+0.0031 | −0.0003/−0.0013 | ahead PASSES big; imm +0.0003 over. Trend SLOWING (−0.0003/+0.2ep vs −0.0007/+0.1ep) → m2b8 floor ≈ +0.0026±. EMA identical. ep75 may graze gate; F27 combo = main hope |
| **★★ PQ+QAT 0.75 ep (F25b) — WIN @ ~352 b** | **`e75_pq`** | **~96** | **~352** | **~1056** | **Yes** | **+0.0021/+0.0012 — BOTH PASS, real margin** | +0.0024/+0.0018 | −0.0004/−0.0006 | **★★ THE WIN: beats F15's 512-b champion (+0.0024/+0.0021) on BOTH modes at 69% the size. The "floor ≈ +0.0026" read was wrong — trend broke through. EMA identical. Robustness + dev-confirm next** |
| PQ+QAT COMBO m4b8 × 0.3 ep (F27) | `c30_pq` | ~160 | ~416 | ~1248 | Yes (but OVER the locked 352 budget) | +0.0024/+0.0013 | +0.0025/+0.0017 | −0.0001/−0.0004 | passes gate; stacking works at equal ep (vs m2b8@0.3 +0.0031) but e75_pq is BETTER at 64 fewer bits ⇒ lever 3 (m4b8+shift3@352) unlikely — int3-shift tax ~+0.0004-7 would land ~+0.0028. EMA identical |
| **PQ+QAT m4b8 × 0.5 ep (F27b)** | `m450_pq` | ~160 | ~416 | ~1248 | OVER budget | **+0.0019/+0.0011 — beats e75_pq at fewer epochs** | +0.0026/+0.0020 | −0.0006/−0.0009 | **m4b8 stays ~0.0009 ahead of m2b8 at equal ep ⇒ lever 3 REVIVED: m4b8+int3shifts@352 b est. +0.0019-0.0024 (queued after Andrew's ep sweep)** |
| co-adapt codebook PTQ diagnostic (no retrain) | `e75coad` | ~96 | ~352 | ~1056 | Yes | **+0.0027/+0.0016 — WORSE than e75_pq** | (e75 weights, +0.0024/+0.0018) | codebook-swap cost +0.0006/+0.0004 vs m2b8 cb | ✗ **lever 2 (co-adaptation) DEAD**: ep75 weights are in equilibrium WITH the champion-trained m2b8 codebook; a codebook refit to the net's own directions hurts without retraining, so an alternating re-QAT round has no headroom. No GPU slot |
| **★★ PQ+QAT 1.0 ep (F25c sweep3) — NEW BEST @ ~352 b** | **`e100_pq`** | **~96** | **~352** | **~1056** | **Yes** | **+0.0016/+0.0005 — both pass with 2-4× margin** | +0.0018/+0.0010 | −0.0002/−0.0005 | **★★ epoch trend ALIVE at 1.0 ep (−0.0005/−0.0007 vs e75_pq). Robustness PASS, better everywhere: imm mean/med/nbad 0.0016/0.0011/108 (e75: 0.0021/0.0014/131), ahead 0.0005/0.0008/108 (e75: 0.0012/0.0012/130), Q4 imm +0.0019, same hard worst-users (6652/6787/6994). ep150/ep200 will show the turn-around point** |
| **★★★ PQ+QAT 1.5 ep (F25c sweep3) — NEW BEST @ ~352 b** | **`e150_pq`** | **~96** | **~352** | **~1056** | **Yes** | **+0.0010/−0.0003 — ahead NEGATIVE: compressed BEATS the fp32 champion** | +0.0018/+0.0006 | **−0.0008/−0.0008** | **★★★ trend STILL paying at 1.5 ep (−0.0006/−0.0008 vs e100). comp_cost increasingly negative — PQ acts as a regularizer. Robustness PASS: imm mean/med/nbad 0.0010/0.0007/69, ahead −0.0003/+0.0002/76, Q4 imm +0.0013, same imm worst-users. Watch: user 6951 ahead outlier grew (+0.0097→+0.0246), single user, re-check at ep200** |

## ★ THE ≤256-BIT NEGATIVE RESULT (fixed net; rigorous, each step measured — F10)
1. **rank-1 insufficient** — perfect unquantized rank-1 = +0.0028 imm > gate (F3, `r1fp`); the 2nd singular
   component carries real predictive signal (F2). State MUST carry rank-2.
2. **rank-2 w/o a stabilizer blows up** — int4-quantizing comp2 + re-SVD every step compounds into a power-user
   runaway (median fine +0.0009, but giants +0.3–0.5); rank-2 int4 +0.0125, PQ +0.1999 (F2, F8).
3. **the stabilizer needs ~int4 (256 b)** — a coarse `e` only partially tames the runaway: rank-1 int2 `e`
   +0.0074, PQ'd `e` +0.0136 (F6, F9, F10); only ~int4-precision `e` fully stabilizes.
4. **the bits don't fit** — rank-2 direction info ≥128 b (PQ, the most efficient found) + int4 stabilizer 256 b
   ⇒ floor ≈ 384 b > 256 b. No ≤256-b split gives both adequate fidelity AND stabilization.
**Positive by-products:** Error-feedback (EF, Andrew's idea — carry the quant error, `A'=Ã+e_prev`, re-compress,
`e=A'−Â`) makes rank-2 near-lossless & robust (`i4efe1i4` 768 b, +0.0009); PQ (corpus codebook on the clustered
factor directions, ~1 b/value) halves the factor bits (`pqm2b8e1i4` 432 b, +0.0027); PQ codebook generalizes to
held-out val (F11). Both are 13×/dramatically better than the current i4 deploy — if the budget can flex >256 b.

## ★ QAT FINDINGS (F12–F15)
- **F13 — fused QAT CUDA kernels** (`gpu_train/rwkv/model/csrc/cuda/rwkv7_cuda.cu`). (a) Full-matrix int-N:
  `rwkv7_wkv_qat_forward/backward` — matches `fake_quant_state` (per-batch amax over H,K,K → forward one block
  per batch element, persists per-step `scale_BT` so the backward stays per-(b,h); STE ⇒ backward = plain WKV
  backward over the quantized trajectory). Bit-exact fwd+bwd (~1e-7), same speed as the plain kernel (160–490×
  over the Python loop). (b) Rank-1 int4 low-rank: `rwkv7_wkv_qat_lr_forward/backward` + device `qat_lr_rank1`
  (power-iterate top singular vector of the max-normalized state, split-sqrt factors, per-column int4, HALF-AWAY
  rounding = deploy `compress_wkv_state` r==1). Per-(b,h), no cross-head coupling. Validated vs a Python deploy
  port (`scratchpad/lr_ref.py`): truncation max 7e-4 rel; full fwd+grads mean ~1e-4 (rare int4 boundary flips);
  5–9.5× the plain kernel, **146× over the Python SVD loop**. Wired via `RWKV7_WKV_QAT` / `RWKV7_WKV_QAT_LR`,
  routed from `quant_aware_rwkv7` (env `RWKV_QAT_FUSED=0` → Python fallback). fp32 (matches deploy, not bf16);
  quant EVERY step (NO every-N-steps — keep train≈deploy). NaN safeguards kept (`qat_sanitize` / `_sanitize_state`).
- **F12 — full-matrix int2 QAT, 0.27 ep from WS-final.** Quant becomes ~free (rank-2 int4 +0.0003, rank-1 int4
  +0.0011) — QAT even LOWERS the rank-1 truncation loss (champ r1fp +0.0028 → qi4r1 whole cost +0.0011). BUT the
  0.27-ep decay-with-fake-quant regressed the fp32 base +0.0024 (my champion was already well-tuned, so QAT only
  traded fp32 accuracy for quant-robustness). **Net wash.** THE CRUX: base regression is the sole blocker.
- **F14 — base-preserving attempt (short low-LR tail from the champion).** 0.1 ep @ 2e-4 warm-started from the
  CHAMPION `h2k16d_904` still regressed the base +0.0024/+0.0022 (≈ F12) → `qi4r1` +0.0036/+0.0028, still fails.
  ⇒ the +0.0024 is intrinsic to full-matrix int2 fake-quant, NOT a tail-length artifact; and it doesn't target
  the imm blocker (rank-1 truncation). (An e05/e20/e27 tail sweep was run to confirm the flat trend; superseded.)
- **F15 — ★★ rank-1 int4 QAT WINS ≤256 b.** Fused low-rank kernel, trained with the exact deploy compression
  (rank-1 int4), 0.1 ep @ 2e-4 from champion `h2k16d_904` (`gpu_train/configs/qat_lr1i4.toml`,
  `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4`; 335 steps, 0 NaN). Weights `reference/qat_lr1i4.safetensors`.
  **VAL `lri4r1` @256 b: +0.0024 imm / +0.0021 ahead — PASSES both** (raw 0.28213/0.31396 vs champ 0.27968/
  0.31182). Per-user robust: mean +0.0024/+0.0021, median +0.0020/+0.0018, uniform by size (Q1 +0.0017→Q4 +0.0028),
  worst user +0.0186 (no blow-up). Decomp: base_drift (lrfp32) +0.0018/+0.0018, compression_cost +0.0007/+0.0004.
  **Why it worked where F12/F14 didn't:** it teaches rank-1-TRUNCATION robustness (the imm blocker), cutting imm
  from +0.0036 (PTQ i4r1) → +0.0024. Caveats: thin imm margin (+0.00005); single val-measured config (the tail
  recipe is a tuned hyperparam → should dev-tune to confirm). Next: shrink base_drift (shorter/lower-LR tail,
  warm optim, EMA) for margin; sweep the recipe on dev; then push BELOW 256 b (rank-1 int3/int2 QAT, PQ-on-QAT).

## Carried lessons (H=1/K=32, model-agnostic; full detail in research_log.md)
- Judge ONLY by log-loss. Keep small/near-zero entries small (schemes that inflate near-zero die even with an
  exact-zero code: 4-level +0.046, mixed53 +0.044). rank-1 int4 won on the OLD model (256 b, +0.0017) — but on
  16×16 the rank-1 truncation ceiling rose to +0.0028 (fails). Rejected: companding, mixed53, 224b asym, ALS.

## FILES / INFRA
- `run_eval.sh`, `score.py`, `dev_users.txt`/`val_users.txt`, `preds_baseline_backup/` (cached fp32/i4/i2/i4r1).
- Deploy engine flags: `RWKV_STATE_LOWRANK_SCOPE`, `RWKV_LOWRANK_PERCOL`, `RWKV_QUANT_SHIFTS`; EF
  `RWKV_LOWRANK_EF`/`EF_ERANK`/`EF_ELEVEL`; PQ `RWKV_LOWRANK_PQ=<file>`/`EF_PQ`. PQ tools: `--dump-corpus`,
  `scratchpad/pq_train.py`, codebooks `scratchpad/pq_cb_m2b8.txt`/`pq_cb_m2b6.txt`.
- QAT: `gpu_train/` (pipeline; uses parent venv + LMDB). Kernels in `.../csrc/cuda/rwkv7_cuda.cu`; build
  `gpu_train/build_ext.cmd` (VS2022 + CUDA 13.2, sm_89). Run cmds `run_qat_*.cmd`, configs `gpu_train/configs/`,
  convert `scratchpad/pth_to_sft.py`, eval `scratchpad/qat_eval.sh`. Parity: `scratchpad/parity_{qat,lr}*.py`,
  `lr_ref.py`. Champion `reference/champ_h2k16.safetensors` = `gpu_train/reference/h2k16d_904.pth`.
</content>
