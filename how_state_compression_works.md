# How the WKV state gets squeezed from 2 KB to 352 bits
### The five levels of compression used in this project, for a STEM reader who hasn't done quantization

This note explains, from the ground up, how this project compresses an RWKV-7 recurrent "WKV state"
for storage. It assumes you're comfortable with vectors and matrices (we recap the **SVD** in a short
interlude before we use it), but **not** that you know anything about quantization. Every mechanism
below is actually used in the deployed scheme; the numbers come from a **real** RWKV state — "real" in
the deployment sense, and (since it's a matrix of real numbers) in the mathematical sense too — unless
labelled "toy".

---

## 0. The problem in one paragraph

The network keeps a little memory matrix for **every flashcard** (and every note). For one card, in
one layer, that memory is **two 16×16 matrices** (one per "head") of 32-bit floats, plus two small
"token-shift" vectors. Raw size: `2 × 16 × 16 × 32 bits ≈ 16 kbit ≈ 2 KB` per card. A power user has
~1,000,000 cards, so raw states are gigabytes. The final scheme stores each card in **352 bits** —
a ~47× reduction — while predicting recall *essentially as well as the uncompressed model* (log-loss
degradation +0.0021/+0.0012, within the ≤ +0.0025 gate). The rest of this note climbs
the five levels that get there.

---

## Level 1 — Quantization: store integers, not floats

"Quantizing" means: replace each real number with the nearest value from a small, evenly-spaced grid,
and remember only *which grid point* you landed on (a small integer).

To quantize a vector `x` symmetrically to "int-N":

1. Find the largest magnitude: `amax = max(|x_i|)`.
2. Decide the integer range. "int4" means integers in `[-7, +7]` (call the limit `qmax = 7`);
   "int2" means `[-1, 0, +1]`.
3. Compute one shared **scale** `s = amax / qmax`.
4. Store, for each entry, the integer `q_i = round(x_i / s)` (clamped to the range).
5. To decompress:  **`x_i ≈ q_i × s`**.

**Worked example (int4).** Take `x = [0.42, -0.11, 0.03, -0.40]`.
- `amax = 0.42`, so `s = 0.42 / 7 = 0.06`.
- `x/s = [7.0, -1.83, 0.5, -6.67]` → round → `q = [7, -2, 1, -7]` (0.5 rounds "half away from zero").
- Decompressed: `q×s = [0.42, -0.12, 0.06, -0.42]`. Close — but note `0.03` became `0.06`: the grid
  is coarse near zero.

Cost: **N bits per number + one scale for the whole vector.** In the final scheme, Level 1 is exactly
how the two token-shift vectors are stored (int4: 64 values × 4 bits = 256 bits).

> Pedantic-but-critical detail: ties round "half away from zero" everywhere (Rust `f64::round`, CUDA
> `roundf`), because training and deployment must do the *bit-identical* operation — see the QAT section.

---

## Level 2 — Per-column scaling: one scale is not enough

Level 1 uses a single scale for everything it quantizes. That has a failure mode: one large entry
inflates `amax`, which inflates `s`, which crushes every small entry to zero. If a matrix has columns
(or sub-vectors) living on very different magnitudes, they cannot share a scale gracefully.

The fix: give **each column its own scale** `s_j = amax(column_j) / qmax`. A few extra stored scalars
buy a much finer grid where the data is small.

This is not cosmetic. In early experiments, rank-2 int2 with one global scale was catastrophic
(+0.051 log-loss); per-column scaling alone brought it to +0.014. The general principle — **more,
smaller scales = less damage, as long as you can afford to store them** — echoes through everything
below: at Level 5, each stored *direction* carries its own norm, which is this same idea taken to its
limit (one "scale" per 16 numbers).

---

## Interlude — the SVD in five minutes

Levels 3–5 lean entirely on the **singular value decomposition (SVD)**, so here's the whole idea.

**The statement.** *Any* real matrix `A` (here 16×16, but it works for any shape) can be written as

```
A  =  U Σ Vᵀ  =  σ₁·u₁v₁ᵀ  +  σ₂·u₂v₂ᵀ  +  σ₃·u₃v₃ᵀ  +  …
```

- The `uᵢ` (columns of `U`) are **orthonormal** unit vectors — the "output directions".
- The `vᵢ` (columns of `V`) are **orthonormal** unit vectors — the "input directions".
- The `σᵢ ≥ 0` are the **singular values**, sorted **largest first** (`σ₁ ≥ σ₂ ≥ … ≥ 0`).

Read the second form out loud: **every matrix is a sum of rank-1 outer products `uᵢ vᵢᵀ`, each scaled
by its singular value.** The big σ's are the "important" pieces; the tiny ones are almost noise.

**Geometric picture.** A matrix is a linear map. Feed it every point on the unit sphere and the
outputs trace an **ellipsoid**. The SVD reads that ellipsoid off directly:

```
   input space                         output space
      v₂                                     u₁ (length σ₁ = 3)
       │        A = [[2,1],[1,2]]        ／
   ────┼────  ───────────────▶      ●─────────────  (the unit circle becomes an ellipse)
       │                              ＼
      v₁                                 u₂ (length σ₂ = 1)
   (unit circle)                      (ellipse: semi-axes σ₁, σ₂)
```

`A` sends input direction `vᵢ` to output direction `uᵢ`, stretched by `σᵢ`. The singular values are
literally the **lengths of the ellipsoid's axes**; a "nearly rank-1" matrix is one whose ellipsoid is
a long thin cigar — almost all the action along one axis.

**Tiny worked example.** For the symmetric `A = [[2, 1], [1, 2]]`:
```
σ₁ = 3,  u₁ = v₁ = [0.707,  0.707]
σ₂ = 1,  u₂ = v₂ = [0.707, -0.707]

A =  3·[[0.5, 0.5],     +   1·[[ 0.5, -0.5],
        [0.5, 0.5]]             [-0.5,  0.5]]
```
Check: `[[1.5,1.5],[1.5,1.5]] + [[0.5,-0.5],[-0.5,0.5]] = [[2,1],[1,2]]` ✓. The **rank-1
approximation** keeps just the first (bigger) term: `Â = [[1.5, 1.5], [1.5, 1.5]]`.

**Why keeping the biggest σ's is the right move (Eckart–Young).** Of *all* rank-`r` matrices, the one
closest to `A` (in least-squares error) is exactly the truncated SVD — keep the top `r` terms, drop
the rest. Not a heuristic; a theorem. (Caveat for this project: "closest matrix" is *not* the same as
"best log-loss" — see the Frobenius section.)

**The `A Aᵀ` connection (why the code power-iterates).** From `A = UΣVᵀ`: `A Aᵀ = U Σ² Uᵀ` — an
ordinary eigendecomposition. The `uᵢ` are eigenvectors of the symmetric `A Aᵀ` with eigenvalues `σᵢ²`.
Repeatedly multiplying any starting vector by `A Aᵀ` amplifies the biggest-σ² direction fastest, so a
few dozen multiply-and-normalize rounds isolate `u₁`. That's **power iteration** — how the deploy code
finds the top direction cheaply, without a full SVD. Given `u₁`, the rest falls out: `v₁ ∝ Aᵀu₁`, and
`σ₁ = ‖Aᵀu₁‖`.

---

## Level 3 — Low-rank: the state is almost a single outer product

Here's the structural fact everything hinges on. Take a **real** 16×16 head matrix `A` (an actual WKV
state pulled from a deployment run) and look at its singular values:

```
σ:  0.439  0.217  0.100  0.044  0.025  0.021  0.008  0.002 ...
    █████████████████████  σ₁ = 0.439   (76.2% of the energy)
    ██████████             σ₂ = 0.217   (rank-2 → 94.8% cumulative)
    ████                   σ₃ = 0.100
    ██                     σ₄ = 0.044
    ▌                      σ₅ ...
```

The energy concentrates in the first component or two. So instead of storing all 256 numbers, store
the **rank-1 truncation**:

```
A  ≈  σ₁ · u v ᵀ        — one left direction u, one right direction v, one magnitude σ₁
```

That's `16 + 16 + 1 = 33` numbers instead of 256, before any quantization. (In practice the code folds
`√σ` into each side — `uf = u·√σ`, `vf = v·√σ`, `A ≈ uf·vfᵀ` — so both stored factors carry equal
magnitude, which quantizes more gracefully. And it finds `u` by power iteration, per the Interlude.)

Does throwing away 24% of the energy hurt? By matrix-distance standards it's terrible (49% Frobenius
error on the state above!). By *log-loss* standards — after the network is trained to expect it (QAT
section) — it costs almost nothing. Which is why the project measures only log-loss.

**Hands-on (PyTorch).** A rank-1 matrix *is* an outer product — build one and count what you stored:

```python
import torch
U = torch.tensor([[1.],
                  [2.],
                  [3.]])

V = torch.tensor([[10.],
                  [20.],
                  [30.]])

W = U @ V.T
print(W)
# tensor([[10., 20., 30.],
#         [20., 40., 60.],
#         [30., 60., 90.]])
```

A 3×3 matrix (9 values) decomposes into just **6** stored numbers (U's 3 + V's 3) — and every row of
`W` is a multiple of `Vᵀ`, every column a multiple of `U`. That perfect redundancy is what "rank 1"
means.

**Rank 2 = two columns per factor.** Give `U` and `V` a second column and the product becomes the
*sum of two* outer products (exactly the SVD's sum form from the Interlude):

```python
U = torch.tensor([[1.,  0.],
                  [2.,  1.],
                  [3., -1.]])

V = torch.tensor([[10., 1.],
                  [20., 2.],
                  [30., 3.]])

W = U @ V.T          # = U[:,0] ⊗ V[:,0]  +  U[:,1] ⊗ V[:,1]
print(W)
# tensor([[10., 20., 30.],
#         [21., 42., 63.],
#         [29., 58., 87.]])
```

Storage: `3×2 + 3×2 = 12` numbers. Note that at 3×3 that's *more* than the 9 you started with —
low-rank only pays once the matrix is big relative to the rank: rank-`r` factors of a `K×K` matrix
cost `2·K·r` values vs `K²`. For our 16×16 matrices, rank-1 = 32 vs 256 (8×) and rank-2 = 64 vs 256 (4×);
for a 3×3, rank 2 is already not worth it.

---

## Level 4 — Low-rank AND quantized: 512 bits

Levels 1–3 compose: take the rank-1 factors from Level 3 and int4-quantize them with per-column scales
from Levels 1–2.

Per layer that is: 2 heads × 2 factors × 16 numbers = **64 numbers** at 4 bits = **256 bits** for the
WKV, plus 256 bits of int4 token-shifts → **card = 512 bits**. With QAT this scheme reached
+0.0024/+0.0021 log-loss degradation — the project's first accepted win, at 512 b/card.

But look where the bits go: each *direction* (a 16-dim unit vector) costs 16 × 4 = **64 bits**. The
magnitude is one cheap scalar; the direction is what's expensive. Level 5 attacks exactly that.

---

## Level 5 — Product Quantization: a direction becomes a catalog number

### 5a. The idea, with no math

Suppose you need to tell a paint shop the exact color of your wall. You could dictate its
red/green/blue values digit by digit — precise, but long. Or you could both hold the *same* printed
palette of 256 numbered swatches, and you just say **"color #137"**. One byte, done. The palette was
printed once, in advance, for everyone; only the *number* travels.

PQ does this to **directions**. A factor from Level 3/4 is really two things:
- a **magnitude** (its norm) — one scalar, cheap, keep it as a scale (Level 2's principle);
- a **direction** (a 16-dim unit vector) — the expensive part.

And here's the empirical gift: the directions that actually occur in real WKV states are **not
arbitrary**. States are shaped by the network's fixed weights, so their top directions cluster hard
around a modest family of recurring patterns. Random 16-dim directions would be spread hopelessly
thin — these are not random. So:

> Build a **catalog of typical directions once, offline** (the "codebook"). Ship it inside the app —
> the same fixed catalog for every user and every card. Per card, store only **which catalog entry is
> closest** — a small integer — plus the norm.

A direction that cost 64 bits at Level 4 becomes a ~16-bit catalog reference.

### 5b. Why "product": split the vector, multiply the options

One catalog for whole 16-dim directions would need to be enormous to cover them well. The trick that
makes PQ practical is to **chop the direction into `m` sub-vectors and give each chunk its own
independent catalog**:

```
u (16-dim, unit)  =  [ ─── first 8 numbers ─── | ─── last 8 numbers ─── ]
                              │                            │
                     catalog A (256 entries,       catalog B (256 entries,
                      each an 8-dim chunk)          each an 8-dim chunk)
                              │                            │
                        nearest = #137                nearest = #52
                              ▼                            ▼
                     STORED:  the pair (137, 52)  =  8 + 8  =  16 bits
```

Decoding = look up entry #137 in catalog A, entry #52 in catalog B, concatenate, multiply by the
stored norm.

Why is the split so powerful? **Combinations multiply.** Each half independently picks one of 256
chunks, so the pair can represent `256 × 256 = 65,536` distinct full directions — while you only ever
*store* two catalogs of 256 chunks and *search* 2 × 256 = 512 entries. You get the coverage of a
65,536-entry catalog for the memory and search cost of 512. (That's the "product" in the name: the
representable set is the Cartesian product of the per-chunk catalogs.) One whole-vector catalog with
65,536 entries would need 128× more storage and 128× more search work for the same nominal coverage.

The scheme here uses `m = 2` chunks of dimension 8, with 256 entries per catalog ("m2b8"):
**16 bits per direction**, versus 64 bits at Level 4.

### 5c. A full worked example (toy, 4-dim)

Encode the unit direction `d = [0.30, 0.62, 0.10, -0.72]` against a tiny 4-entry catalog (so indices
need just 2 bits). Compute squared distance to every entry:

```
entry 0: [ 0.50  0.50  0.50  0.50]    ‖d − e₀‖² = 1.70
entry 1: [ 0.26  0.53  0.09 -0.80]    ‖d − e₁‖² = 0.016   ◀ nearest
entry 2: [-0.70  0.10  0.70  0.00]    ‖d − e₂‖² = 2.16
entry 3: [ 0.00  0.71  0.00 -0.71]    ‖d − e₃‖² = 0.11
```

For entry 1 explicitly: `(0.30−0.26)² + (0.62−0.53)² + (0.10−0.09)² + (−0.72−(−0.80))² =
0.0016 + 0.0081 + 0.0001 + 0.0064 ≈ 0.016`. Smallest → **store "1"**. Decoding returns
`[0.26, 0.53, 0.09, -0.80] × ‖d‖` — a good stand-in for `d`, at the cost of a 2-bit index plus the
norm. The reconstruction error (that 0.016) is the price; the codebook is built to make that price
small *for the directions that actually occur*.

### 5d. Where the catalog comes from: k-means, once, offline

Dump a large corpus of real WKV states (millions of directions from many users' card histories),
factor each into its `u, v` directions, split into chunks, and run **k-means** on each chunk
position: start with 256 tentative "centers", repeatedly (1) assign every chunk in the corpus to its
nearest center, (2) move each center to the average of the chunks assigned to it. After a few dozen
rounds the 256 centers settle where the data is densest — those centers *are* the catalog
("centroids"). Separate catalogs are trained for `u`-directions vs `v`-directions and for each chunk
position, because their distributions differ.

Three properties matter for deployment:
- **Global and fixed.** One codebook, baked into the app, shared by all users and cards — so its size
  (a few KB) does **not** count against the per-card budget. Only indices + norms are per-card.
- **Robust to churn.** Cards being added or deleted change nothing — there's no per-user fitting.
- **Generalizes.** The codebook is trained on one pool of users and validated on held-out users; the
  clustering structure of state directions carries over.

One free trick rides along: **sign canonicalization**. Since `(−u)(−v)ᵀ = u vᵀ`, every factor pair is
flipped (before encoding) so that `u`'s largest-magnitude entry is positive. All directions land in
the same "hemisphere", so no catalog entries are wasted on mirror images — effectively doubling
resolution for free.

### 5e. The final 352-bit card

| piece | how | bits |
|---|---|---|
| 4 WKV directions (u, v × 2 heads) | PQ: 2 indices × 8 bits each | 64 |
| 4 norms | small scalars (scales) | ~32 |
| token-shifts (2 × 32 values) | Level 1, int4 | 256 |
| **card total** | | **≈ 352** |

(A note stores 3 layers ≈ 1056 bits. The catalog itself: 2 roles × 2 positions × 256 centroids ×
8 dims ≈ 8K floats, shipped once in the app.)

---

## The twist that makes all of this hard: it happens *every step*

This is not one-shot "compress the final state and store it." The deployed engine **re-compresses the
state after every single review**, feeding the compressed state back into the recurrence:

```
 raw state ─▶ compress ─▶ feed back ─▶ next review ─▶ compress ─▶ feed back ─▶ ...
```

So a small *bias* — say, a grid with no exact zero that nudges every near-zero entry slightly — does
not average out; it **compounds** over hundreds of steps. Any scheme must be judged with the
recurrence in the loop.

---

## Judge by log-loss, not by "how close the matrix looks"

The most important lesson of the whole project:

> **Reconstruction error (Frobenius `‖Â − A‖`) is a lie here — it is anti-correlated with what we
> care about.**

Cautionary tale: a "4-level 2-bit" grid once achieved the *best* Frobenius error of all int2 schemes
(0.75 vs 0.97) and the *worst* log-loss (+0.046, ~3× worse than plain int2). Its grid had no exact
zero, so it biased the many near-zero entries, and the bias compounded through the recurrence.
Conversely, our rank-1 approximation has a scary-looking **49%** Frobenius error and costs almost
nothing in log-loss. So: every scheme is judged **only** by real recall-prediction log-loss on
held-out users. Matrix-distance intuition actively misleads.

---

## QAT: teach the network to expect the compression

Compressing the states of a finished network ("post-training quantization", PTQ) works poorly — the
network never agreed to any of this. **Quantization-aware training (QAT)** fixes it: during a
fine-tune from the champion weights, the forward pass applies the *exact* deploy compression (the
power iteration, the codebook snap, the int4 shift quant — bit-identical) to the state at every step,
so the network learns to route information through representations that survive it.

The gradient trick is the **straight-through estimator (STE)**: forward uses the compressed state;
backward pretends compression was the identity (gradients pass through the snap unchanged). Forward =
real deploy behavior; backward = smooth enough to train.

```
forward :  state ─▶ [rank-1 + codebook snap] ─▶ rest of net ─▶ loss
backward:  grad  ◀────── (straight through) ──────────────◀ grad
```

Two findings made the 352-bit scheme work:

1. **Train ≈ deploy, exactly.** Fake-quantizing something *similar* to deployment teaches the wrong
   robustness. The compression in the QAT forward is verified bit-compatible with the Rust deploy
   path to ~1e-7.
2. **Train long enough.** With a short fine-tune, PQ passed most of its cost as a drifted base model.
   Training longer let the base *recover under the compressed regime* — the degradation fell monotonically
   with fine-tune length, to the point where the PQ-compressed model slightly **beats its own
   uncompressed weights** (the compression acts as a familiar, trained-for representation, not an
   injury). Final numbers at 352 b: PTQ +0.0046 → QAT +0.0021/+0.0012.

---

## The whole pipeline, end to end

```
                 ┌────────────────────  PER CARD, PER LAYER, PER HEAD, EVERY REVIEW  ────────────────────┐
  16×16 WKV state A
        │
        ▼
  power iteration  →  top direction u, then v ∝ Aᵀu, magnitude σ        (Level 3 + Interlude)
        │
        ▼
  split-√σ factors  uf = u·√σ,  vf = v·√σ
        │
        ▼
  sign-canonicalize  (flip pair so uf's dominant entry > 0)              (Level 5d)
        │
        ▼
  PQ-encode each direction: chop in 2 → nearest centroid in each        (Level 5b)
  catalog → store (index, index) + norm                                  16 b + norm per direction
        │
        ▼
  reconstruct  Â = uf·vfᵀ  →  feed back into the recurrence              (every step!)

  …and separately: token-shift vectors → int4 with per-vector scales     (Levels 1–2, 256 b)
```

| scheme | card bits | log-loss degradation (imm / ahead) |
|---|---|---|
| raw fp32 | ~18,432 | 0 (reference) |
| Level 4: rank-1 int4 + QAT | 512 | +0.0024 / +0.0021 |
| **Level 5: rank-1 PQ + QAT (deployed)** | **≈ 352** | **+0.0021 / +0.0012** |

Both pass the ≤ +0.0025 gate; the 352-bit scheme is the better *and* smaller one,
and it is at least as robust per-user (no user gets wrecked; the hardest users are equally hard for
every scheme).

---

## Cheat-sheet / glossary

- **Quantize (Level 1)** — snap numbers to an integer grid; store integers + a scale; `x ≈ q·s`.
- **qmax / int-N** — grid limit: int4 = ±7, int2 = ±1.
- **Per-column scaling (Level 2)** — one scale per column/sub-vector instead of one global; "more,
  smaller scales = less damage".
- **SVD** — any matrix = sum of rank-1 pieces `σᵢ·uᵢvᵢᵀ`, biggest σ first (Interlude).
- **Singular value / vector** — σᵢ = stretch along axis i; uᵢ/vᵢ = output/input directions of that axis.
- **Eckart–Young** — truncated SVD is the *best possible* rank-r approximation (in least squares).
- **Power iteration** — repeat `u ← normalize(A Aᵀ u)` to get the top direction cheaply.
- **Low-rank (Level 3)** — store `σ·u·vᵀ` (two directions + magnitude) instead of the full matrix.
- **Level 4** — low-rank factors, int4-quantized per column: 512 b/card.
- **PQ / product quantization (Level 5)** — chop a unit direction into chunks; replace each chunk by
  its nearest catalog entry; store the indices + the norm. Combinations multiply: two 256-entry
  catalogs cover 65,536 direction combinations.
- **Codebook / centroid** — the fixed, global catalog of typical chunks, built once by k-means on a
  corpus of real states; ships in the app, costs no per-card bits.
- **Sign canonicalization** — flip `(u, v)` together so u's dominant entry is positive; `u vᵀ`
  unchanged, catalog coverage doubled.
- **PTQ vs QAT** — compress after training vs fine-tune with the exact compression in the forward pass.
- **STE** — forward uses the snapped value, backward passes gradients through unchanged.
- **Log-loss, not Frobenius** — judge only by recall-prediction loss; matrix distance anti-correlates.

---

*Grounded in the deployed code: `engine/src/model.rs` (`compress_wkv_state`, `PqCodebook`),
`scratchpad/pq_train.py` (codebook k-means), and the QAT kernel in
`gpu_train/rwkv/model/csrc/cuda/rwkv7_cuda.cu` (`qat_lr_rank1`). Real numbers computed by
`scratchpad/pq_explainer_numbers.py` on a real WKV state.*
