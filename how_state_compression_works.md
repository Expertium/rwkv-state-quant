# How the WKV state gets squeezed from 2.25 KB to 64 bits
### The nine levels of compression used in this project, explained from scratch

This note explains, from the ground up, how this project compresses an RWKV-7 recurrent "WKV state"
for storage. It assumes **no linear algebra and no quantization background** — every concept is built
up from ordinary arithmetic, and everything gets a worked numeric example. If you can read a
multiplication table, you have the prerequisites. Every mechanism below is actually used in the
deployed scheme; the numbers come from a **real** RWKV state pulled from a deployment run, unless
labelled "toy".

---

## 0. The problem in one paragraph

The network keeps a little memory matrix for **every flashcard** (and every note). For one card, in
one layer, that memory is **two 16×16 matrices** (one per "head") of 32-bit floats, plus two 32-number
"token-shift" vectors (explained in their own section below). Raw size, counted exactly:
`(2×16×16 + 2×32) × 32 bits = 18,432 bits = 2.25 KB` per card. A power user has ~1,000,000 cards, so
raw states are gigabytes. The comfortable champion stores each card in **72 bits — nine bytes, a 256×
reduction** — at a log-loss degradation of **+0.0018 / +0.0016** (immediate-recall head /
forgetting-curve head), well inside the project's ≤ +0.0025 acceptance gate; pushing to the very
boundary of that gate, **64 bits — one 8-byte machine word per card, 288×** — also passes, with a
razor-thin margin (+0.002492 / +0.0012). An earlier milestone on the
way, the 352-bit card, is even *cheaper* than free on one head: **+0.0010 / −0.0003** (yes, negative —
that compressed model predicts marginally *better* than the uncompressed one on the forgetting-curve
head; the QAT section explains how that is possible). The rest of this note climbs the nine levels
that get there: Levels 1–5 build the 352-bit card, Levels 6–9 take the same card down to 64 bits.

What is that "memory matrix", concretely? Just a **grid of numbers** — 16 rows × 16 columns, 256
ordinary decimal numbers like `0.31` and `-0.007` — that the network reads and updates every time you
review the card. It is not code and not text; its values summarize everything the network has learned
about that card's history. Our whole job is to store that matrix (plus two shorter lists of numbers)
in as few bits as possible without changing what the network predicts.

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
how the two token-shift vectors are stored (int4: 64 values × 4 bits = 256 bits, plus one scale per
vector — what those vectors *are* gets its own section after Level 4).

> Pedantic-but-critical detail: ties round "half away from zero" everywhere (Rust `f64::round`, CUDA
> `roundf`), because training and deployment must do the *bit-identical* operation — see the QAT section.

---

## Level 2 — Per-column scaling: one scale is not enough

Level 1 uses a single scale for everything it quantizes. That has a failure mode: one large entry
inflates `amax`, which inflates `s`, which crushes every small entry to zero. If a matrix has columns
(or sub-vectors) living on very different magnitudes, they cannot share a scale gracefully.

The fix: give **each column its own scale** `s_j = amax(column_j) / qmax`. A few extra stored scalars
buy a much finer grid where the data is small.

**Worked example (int2, so the effect is stark).** int2 allows only the integers −1, 0, +1. Say one
column of a table holds `[0.90, 0.60]` and another holds `[0.08, 0.05]`.
- *One shared scale:* `amax = 0.90`, `s = 0.90/1 = 0.90`. The small column becomes
  `round(0.08/0.90) = 0` and `round(0.05/0.90) = 0` — **wiped out entirely**.
- *Per-column scales:* the small column gets its own `s₂ = 0.08`, so it stores `[1, 1]` and
  decompresses to `[0.08, 0.08]` — not perfect, but alive.

This is not cosmetic. In early experiments, rank-2 int2 with one global scale was catastrophic
(+0.051 log-loss); per-column scaling alone brought it to +0.014. The general principle — **more,
smaller scales = less damage, as long as you can afford to store them** — echoes through everything
below: at Level 5, each stored *direction* carries its own norm, which is this same idea taken to its
limit (one "scale" per 16 numbers).

---

## Interlude — vectors, rank-1 matrices, and "important directions" (no prior math assumed)

Levels 3–5 all rest on three small ideas. Here they are, built from ordinary arithmetic. First, two
words used throughout: a **vector** is just a list of numbers, like `[3, 4]`; a **matrix** is just a
rectangular grid of numbers, like the 16×16 memory matrix from section 0.

### Idea 1: rank-1 matrices — 256 numbers that are secretly 32

Some matrices have a special structure: every entry is the product of a number attached to its row
and a number attached to its column. Take the vectors `u = [1, 2, 3]` and `v = [10, 20, 30]`:

```
        │ 10   20   30      ← v's entries, one per column
   ─────┼──────────────
    1   │ 10   20   30
    2   │ 20   40   60
    3   │ 30   60   90
    ↑ u's entries, one per row
```

The bottom-right `90` is `3 × 30`; every entry is (row's number) × (column's number) — the whole
thing works like a multiplication table from school. A matrix with this structure is called a
**rank-1 matrix**, the two vectors are its **factors**, and the operation "build the matrix from the
two factors" is called the **outer product**. The payoff: all **9** entries are reproducible from the
**6** factor numbers — and the bigger the matrix, the better the deal. A rank-1 16×16 matrix has 256
entries but only 16 + 16 = 32 factor numbers. **Eight entries per number stored.** This matters
because of an empirical miracle at Level 3: *the network's memory matrices are almost rank-1.*

### Idea 2: a vector has a length

A vector like `v = [3, 4]` can be pictured as an arrow: go 3 across, 4 up. Its **length** (the term
of art is **norm**) comes from the Pythagorean theorem, exactly like the long side of a right
triangle:

```
length of [3, 4]  =  √(3² + 4²)  =  √25  =  5
```

Same recipe for any vector, however long: square every entry, add, take the square root. Two derived
notions we'll use constantly:
- A **unit vector** (or "direction") is a vector whose length is exactly 1. Any vector splits
  cleanly into *length × direction*: `[3, 4] = 5 × [0.6, 0.8]` — one number saying "how big", one
  unit vector saying "which way". Check: `√(0.6² + 0.8²) = √1 = 1` ✓.
- The **distance** between two vectors = the length of their difference (subtract entry by entry,
  then apply the length recipe). This is how "nearest" is defined at Level 5.

### Idea 3: every matrix is a sum of a few rank-1 pieces — biggest first

Most matrices are not rank-1. But here is a remarkable theorem (its formal name is the **singular
value decomposition, "SVD"**): **any matrix whatsoever can be written as rank-1 piece #1 + rank-1
piece #2 + …, with the pieces sorted so that #1 carries the most of the matrix, #2 the next most,
and so on.** Each piece's importance is a single number called its **singular value** (written σ);
the tail pieces usually carry near-invisible dust.

**Tiny worked example.** The matrix `[[2, 1], [1, 2]]` is not rank-1 (a rank-1 matrix must satisfy
top-left × bottom-right = top-right × bottom-left, and `2×2 ≠ 1×1`). But it *is* the sum of two
rank-1 pieces:

```
[[2, 1],     =     [[1.5, 1.5],     +     [[ 0.5, -0.5],
 [1, 2]]            [1.5, 1.5]]            [-0.5,  0.5]]
                  piece #1 (σ₁ = 3)       piece #2 (σ₂ = 1)
```

Check the top-left entry: `1.5 + 0.5 = 2` ✓. Piece #1 is three times more important than piece #2
(σ 3 vs 1) — so if you may keep only one, keep #1. The matrix rebuilt from piece #1 alone
(`[[1.5, 1.5], [1.5, 1.5]]`) is the **best possible rank-1 approximation** of the original — that's
a theorem (Eckart–Young), not a heuristic. Keeping the top `r` pieces is likewise the best rank-`r`
approximation. (Caveat for this project: "closest matrix" is *not* the same as "best predictions" —
see the Frobenius section near the end.)

### How the code finds the biggest piece: power iteration

The deployed engine can't afford a textbook SVD on every card at every review. It uses a beautifully
cheap trick instead. A matrix, used as a machine, eats a vector and produces a vector (multiply each
row by the input entry-by-entry and add up). That machine *amplifies* some directions more than
others — and the direction it amplifies most is exactly piece #1's factor direction. So: feed the
machine **any** starting vector, rescale the output to length 1, feed it back in, repeat. Every
pass, the favored direction grows relative to everything else, so the vector swings toward it and
settles.

Watch it happen on `[[2, 1], [1, 2]]`, starting from the (wrong) guess `[1, 0]`:

```
pass 1:  matrix eats [1, 0]       → [2, 1]       → rescale → [0.89, 0.45]
pass 2:  matrix eats [0.89, 0.45] → [2.24, 1.79] → rescale → [0.78, 0.62]
pass 3:  → [0.75, 0.66]    pass 4: → [0.72, 0.69]   …settling on [0.71, 0.71]
```

It converges to `[0.707, 0.707]` — piece #1's factor direction. A few dozen passes of
multiply-and-rescale (microseconds for a 16×16 matrix) replace a full decomposition. Given that one
direction, the other factor and the singular value fall out with one more multiply.

---

## Level 3 — Low-rank: the state is almost a single rank-1 matrix

Here's the structural fact everything hinges on. Take a **real** 16×16 memory matrix (an actual WKV
state pulled from a deployment run) and decompose it into its sorted sum of rank-1 pieces
(Interlude, Idea 3). The singular values:

```
σ:  0.439  0.217  0.100  0.044  0.025  0.021  0.008  0.002 ...
    █████████████████████  piece #1: σ₁ = 0.439   (76.2% of the matrix's "energy")
    ██████████             piece #2: σ₂ = 0.217   (#1+#2 together: 94.8%)
    ████                   piece #3: σ₃ = 0.100
    ██                     piece #4: σ₄ = 0.044
    ▌                      piece #5 ...
```

("Energy" = sum of squares, the same quantity the length recipe computes — the standard way to say
how much of the matrix a piece accounts for.) The matrix is three-quarters piece #1, and pieces #3
onward are dust. So instead of storing all 256 numbers, store **just piece #1's two factors**: one
16-number vector, another 16-number vector, one singular value — `16 + 16 + 1 = 33` numbers instead
of 256, before any bit-squeezing. This is the **rank-1 approximation**.

Two practical details of how the code stores it. First, the singular value is not kept as a separate
number — its square root is folded into *each* factor (`uf = u·√σ`, `vf = v·√σ`; the outer product
of the two scaled factors rebuilds the same piece, since `√σ × √σ = σ`), which makes both stored
vectors equal in magnitude and therefore friendlier to quantize. With easy numbers: a piece with
unit factors and σ = 9 becomes two factors each scaled by √9 = 3 — and note both scaled factors now
have **length 3, the same number**, a fact that becomes a free lunch at Level 9. Second, the factor
direction is found by power iteration (Interlude), not a full decomposition.

Does throwing away 24% of the matrix hurt? By matrix-distance standards it's terrible (49% error on
the state above!). By *log-loss* standards — after the network is trained to expect it (QAT section)
— it costs almost nothing. Which is why the project measures only log-loss.

**Hands-on (PyTorch).** `U @ V.T` is the outer product — build a rank-1 matrix and count what you
stored:

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

A 3×3 matrix (9 values) decomposes into just **6** stored numbers (U's 3 + V's 3) — it's the
Interlude's rank-1 example, and every row of `W` is a multiple of `V`, every column a multiple of
`U`. That perfect redundancy is what "rank 1" means.

**Rank 2 = two pieces.** Give `U` and `V` a second column each and the product becomes the *sum of
two* rank-1 pieces (exactly the Interlude's sorted-sum form):

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
cost `2·K·r` values vs `K²`. For our 16×16 matrices, rank-1 = 32 vs 256 (8×) and rank-2 = 64 vs
256 (4×); for a 3×3, rank 2 is already not worth it.

---

## Level 4 — Low-rank AND quantized: 512 bits

Levels 1–3 compose: take the two factor vectors from Level 3 and int4-quantize them (Level 1) with
per-factor scales (Level 2).

**Counting it through:** each memory matrix keeps 2 factors × 16 numbers; a card has 2 matrices
(one per "head"), so 64 numbers total. At 4 bits each that is 64 × 4 = **256 bits** for the WKV side,
plus 256 bits of int4 token-shifts (next section) → **card = 512 bits**. With QAT this scheme reached
+0.0024/+0.0021 log-loss degradation — the project's first accepted win, at 512 b/card.

But look where the bits go. Split each factor into *length × direction* (Interlude, Idea 2): the
length is one cheap number, but the direction — the 16-entry unit vector — costs 16 × 4 = **64
bits**. The direction is what's expensive. Level 5 attacks exactly that.

---

## The other piece of the state: token-shift vectors

The WKV matrices are only part of what must be persisted per card. The other part — 256 of the 352
bits! — is the **token-shift state**, so it deserves an actual explanation.

**What they are.** An RWKV block does not look only at the current review. Each block works on a
learned *blend of the current step's numbers and the previous step's numbers* — for every one of its
32 channels it computes

```
used = (1 − μ) × current + μ × previous        (μ = a learned mixing weight, one per channel)
```

**Blend example:** if some channel reads `current = 4.0`, remembered `previous = 2.0`, and its
learned mix is `μ = 0.25`, the block actually uses `0.75×4.0 + 0.25×2.0 = 3.5`. That one-step
lookback is the "token shift". It gives every block a cheap short-range memory (features spanning
two consecutive reviews) without spending the long-term WKV memory on them.

**Why they must be stored.** The blend needs `previous`. When a card comes back for its next review
— possibly months later — the computation resumes exactly where it left off, so the engine must
remember *that card's last set of channel values* for every block. Per layer that is two lists of
32 numbers: one for the time-mix block (the one that owns the WKV matrices) and one for the channel-mix
block (the other half of the layer). Drop them and the first review after loading is computed from a
wrong blend — and because each review's result feeds the next, the error then propagates through the
whole rest of the history.

**How they're stored (in the 352-bit era).** They are plain vectors — not matrices, so there is no
rank-1 structure to exploit. They get the Level 1–2 treatment: **int4 codes with one scale per
vector**. Counted exactly: 2 vectors × 32 values × 4 bits = **256 bits** of codes, plus 2 per-vector
scales.

**Why int4 and not fewer — in the 352-bit era.** This was measured, and it was the sharpest empirical
fact of the first descent: shifts are *more* quantization-sensitive than the WKV matrices. Taking the
shifts to int3 costs ~+0.0004 imm log-loss even with QAT; int2 is nearly catastrophic. The sub-512-bit
exploration concluded "token-shifts are the binding wall": the WKV side could be squeezed all the way
down to PQ catalog numbers, but *integer-grid* coding of the shifts stalls at int4. (Foreshadowing:
Level 7 eventually breaks this wall by giving the shifts the same catalog treatment as the WKV
directions — the wall was a property of integer grids, not of the shifts themselves.)

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

A direction that cost 64 bits at Level 4 becomes a 16-bit catalog reference (the one-scalar norm is
paid identically at both levels, so the comparison is clean: 64 → 16 bits per direction).

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
("centroids").

**K-means in one dimension, so you can watch it work.** Data: `0.9, 1.0, 1.1, 5.0, 5.2`; ask for 2
centers, starting from bad guesses `0` and `9`:
- Assign each point to its nearest center: `0.9, 1.0, 1.1` are closer to `0`; `5.0, 5.2` are closer
  to `9`.
- Move each center to the average of its points: center A → `(0.9+1.0+1.1)/3 = 1.0`; center B →
  `(5.0+5.2)/2 = 5.1`.
- Assign again: nothing changes — every point already sits with its nearest center. Done. The two
  centroids `1.0` and `5.1` have found the two clumps in the data, despite the terrible starting
  guesses. The real thing is identical, just with 8-dimensional chunks instead of single numbers
  and 256 centers instead of 2.

Separate catalogs are trained for `u`-directions vs `v`-directions and for each chunk position,
because their distributions differ.

Three properties matter for deployment:
- **Global and fixed.** One codebook, baked into the app, shared by all users and cards — so its size
  (a few KB) does **not** count against the per-card budget. Only indices + norms are per-card.
- **Robust to churn.** Cards being added or deleted change nothing — there's no per-user fitting.
- **Generalizes.** The codebook is trained on one pool of users and validated on held-out users; the
  clustering structure of state directions carries over.

One free trick rides along: **sign canonicalization**. Flipping the sign of *both* factors leaves
the matrix unchanged, because the two minus signs cancel in every entry. Concretely: `u = [1, −2]`,
`v = [3, 1]` builds `[[3, 1], [−6, −2]]`; the flipped pair `u = [−1, 2]`, `v = [−3, −1]` builds
`(−1)(−3) = 3`, `(−1)(−1) = 1`, `(2)(−3) = −6`, `(2)(−1) = −2` — the identical matrix. So each pair
is flipped (before encoding) to a standard orientation: whichever sign makes `u`'s largest-magnitude
entry positive. All directions land in the same "hemisphere", no catalog entries get wasted on
mirror images of each other — effectively doubling the catalog's resolution for free.

### 5e. The 352-bit card, counted exactly (the first PQ champion — Levels 6–9 shrink it further)

| piece | how | bits |
|---|---|---|
| 4 WKV directions (u, v × 2 heads) | PQ: 2 indices × 8 bits each | 4 × 2 × 8 = **64** |
| 4 direction norms | one int8 scalar each | 4 × 8 = **32** |
| token-shift codes (2 vectors × 32 values) | Level 1, int4 | 2 × 32 × 4 = **256** |
| **card total** | | **64 + 32 + 256 = 352** |

A note stores 3 layers = 3 × 352 = **1056 bits**. The catalog itself: 2 roles (u, v — both heads share
it) × 2 chunk positions × 256 centroids × 8 dims = 8,192 floats = 32 KB as f32, shipped once in the
app — it costs no per-card bits.

Full disclosure on the scalars, since this table claims to be exact: the two **token-shift scales**
(one per vector, from Level 1) are the only per-card numbers *not* in the table. The project's
accounting convention amortizes quantization scales (they are O(1) per object and every scheme in
every comparison table pays them identically); if you insist on charging literally every per-card
scalar at int8, the strict total is 352 + 2×8 = **368 bits**. The 4 direction norms *are* charged
(at 8 bits each) because they are the magnitude payload of the PQ scheme itself. One more honesty
note: the evaluation engine holds these 6 scalars as floats in RAM at run time — the bit charges
above are *storage* precision.

---

## Level 6 — Coarser catalogs are (nearly) free under QAT

The 352-bit card uses 256-entry catalogs — 8-bit indices — chosen conservatively. The obvious
question: how few swatches can the palette have before quality collapses? The answer turned out to be
the single most surprising fact of the second descent:

> Under QAT, shrinking the WKV catalogs from **256 entries to 8** — from 8-bit indices to **3-bit**
> indices — costs almost nothing.

Measured along the way (all with the shifts and training recipe held fixed): 256-entry catalogs at
352 b gave +0.0010/−0.0003; **64-entry** catalogs (6-bit indices, 272 b) gave +0.0011/−0.0003 — a
statistical tie with *the best robustness profile of any scheme tested*; **16-entry** catalogs (4-bit
indices) carried the 256-, 192- and 144-bit rungs; **8-entry** catalogs (3-bit indices) hold the
current 80-bit champion. Each halving of the index width moved the needle by roughly +0.0000 to
+0.0004.

Why doesn't an 8-swatch palette hurt? Because QAT changes what the palette has to cover. With a
*fixed* network, the catalog must approximate whatever directions the states happen to visit — more
entries, better coverage. But under QAT the network is fine-tuned *through* the snap (STE), so it
learns to **park its states near the centroids it knows exist**. The palette stops chasing the data;
the data walks over to the palette.

**A one-number illustration of the mechanism.** Suppose (toy) some state coordinate the frozen
network likes to produce is `0.43`, and the nearest catalog value is `0.50`. Frozen network: every
single review eats a `0.07` snap error, forever. QAT: the network discovers during fine-tuning that
producing `0.49` costs it almost nothing elsewhere but shrinks the ever-repeating snap error to
`0.01` — so it simply *moves*. Multiply that adjustment across every coordinate and every card, and
an 8-swatch palette the network has rehearsed with beats a 256-swatch palette it hasn't. A side
effect seen repeatedly in the numbers: the coarse snap acts as a mild regularizer (several
coarse-catalog runs beat their own uncompressed base — negative compression cost).

This lever is what pays for most of the 352 → 80 descent on the WKV side: 4 directions × 16 bits =
64 b of indices at Level 5 become 4 × 6 = **24 b** at the 80-bit card.

---

## Level 7 — The shifts join the catalog: PQ beats int-N there too

Level 6 leaves the token shifts as the dominant cost (256 of the bits — 4× the WKV payload). Recall
the 352-era wall: integer grids on the shifts stall at int4. The hypothesis (Andrew's, stated in
advance of the measurement): *PQ won over integer grids for the WKV directions, so the same should
happen for the shifts.* Confirmed, decisively:

- **PQ shifts at 80 bits beat int2 shifts at 128 bits** — fewer bits *and* better log-loss.
- The deployed recipe: normalize each 32-value shift vector, chop into **m = 4 chunks of 8**, one
  catalog per chunk position per role (the time-mix shift and the channel-mix shift get separate
  catalogs — their distributions differ), store 4 indices + the norm. With 64-entry catalogs
  ("m4b6"), a shift vector costs 4 × 6 = 24 bits of indices; both vectors together **48 bits** —
  down from 256.

The shifts are not rank-1 factors like the WKV case, but the same recipe applies: split off the
magnitude (norm), catalog the direction. The k-means corpus here is ~450k real shift vectors per
role, dumped from deployment runs.

**Worked count for one shift vector.** Take a time-mix shift vector of 32 numbers with length 5.8.
Divide by 5.8 → a unit vector. Chop into 4 chunks of 8. Chunk 1 gets compared against the 64 entries
of catalog (time-mix, position 1) — say entry #23 is nearest → store `23` (6 bits, since 2⁶ = 64).
Same for chunks 2–4 against their own catalogs → three more 6-bit indices. Total: `4 × 6 = 24 bits`
of indices + the norm 5.8 (whose own fate is Level 9's story). Decoding: look up the four entries,
concatenate into a 32-number unit vector, multiply by 5.8. Both shift vectors together: **48 bits**
of indices — where the 352-bit era spent 256.

**Where the new wall is — and the trick that finally bent it.** Halving the shift catalogs once
more (32 entries, "m4b5", 40 bits) fails the gate: +0.0027/+0.0017 — and, crucially, it fails
*identically* (+0.0027/+0.0018) even with every learnable lever of Level 8 engaged. That is the
signature of a **capacity wall**, not an optimization gap: 32 swatches per chunk cannot cover the
shift manifold, and no amount of catalog training moves it.

What finally bent it is the one thing chunked catalogs are structurally blind to. Chopping a vector
into 4 independent chunks assumes the chunks don't coordinate — but real shift vectors have
correlations *across* the chunk boundaries, and no amount of per-chunk catalog quality can encode
those. The fix: **learn a rotation** (an orthogonal remix of the 32 coordinates — think of it as
re-axising the space; lengths and distances are perfectly preserved) that is applied *before* the
chop and undone after decoding. The rotation is trained jointly with everything else (initialized
to "do nothing", it drifts to wherever the loss wants), ships as one global 32×32 matrix per role —
amortized like the catalogs, zero per-card bits — and moves the cross-chunk structure to where the
chunked catalogs can see it. Result: the 32-entry wall that stood at +0.0027 twice fell to
**+0.002492** — under the gate by a whisker — buying the final 8 bits of the 64-bit card. (This is
the "SpinQuant/QuaRot" idea from the LLM-quantization literature, adapted to product quantization.)

---

## Level 8 — Learnable catalogs: the codebook joins the training

The catalogs of Levels 5–7 are built by k-means on states of the *original* network — then frozen.
But QAT fine-tunes the network away from those states; why should the old palette stay optimal?

Two ways to update it were tried; only one works:

1. **Post-hoc refit (dead).** Re-run k-means on the QAT'd network's own states, swap the new catalog
   in. This *hurts* (+0.0027 vs +0.0010): the weights equilibrated to the exact catalog they trained
   with; swapping it after the fact just breaks the agreement.
2. **Gradient co-training (works).** Make the centroids **trainable parameters** of the QAT run
   itself. Each step, the nearest-centroid *selection* is frozen (a hard assignment, like an
   embedding lookup), but the *selected* centroid entries receive real gradients through the
   reconstruction, while the state gradients pass through via STE. Selections stay hard; the swatches
   drift to wherever the loss wants them. Measured on the shift catalog: **−0.0004 imm** vs the same
   run with a frozen catalog, plus a better per-user profile — one of the few levers that improves
   quality at *equal* bits.

**What one such update looks like (toy numbers).** Say swatch #12 currently starts with
`[0.26, 0.53, …]`, and this training step 4,000 different card-states happened to snap to it. The
loss gradient aggregates all 4,000 opinions about how #12's first entry should move — suppose they
net out to "a bit bigger". The optimizer nudges `0.26 → 0.262`. Next step, vectors near the boundary
between #12 and some other swatch may now snap differently — the assignment re-freezes each step,
so the catalog and the states negotiate their way to an arrangement the loss likes. Compare k-means
(step 5d), which moves a center to the *average* of its members: same spirit, but k-means optimizes
"look like the data", while co-training optimizes "predict recall well" — and those differ, which is
the whole reason this lever earns anything.

For the shift catalog this is ~30 lines of PyTorch (the shift snap lives in Python). For the WKV
catalog the selection happens *inside the fused CUDA recurrence kernel*, so the backward kernel had
to be taught to accumulate per-centroid gradients (recording which centroid each step selected during
its checkpoint re-run, then reducing `∂L/∂centroid = norm × (state-gradient ⊗ other-factor)` into a
buffer). One methodological trap worth recording: **you cannot finite-difference-check an STE
gradient.** Perturbing a centroid and re-running the true forward measures the *true* function —
where downstream re-quantization (hard selections, staircase norms) has zero local derivative and
damps everything ~5–10× — so numeric and STE gradients *should* disagree. The correct reference is a
PyTorch autograd port of the same STE semantics; against that, the kernel matched to cosine 1.000000
/ median 1.7e-07.

Honest scorecard for the WKV half of this lever: it verified exactly, trains stably, and improved the
GPU-side readout slightly — but it did **not** rescue the 80-bit shift-route attempt (Level 7's
capacity wall held). Learning tunes a palette; it cannot make 32 swatches paint like 64.

---

## Level 9 — The norms turn out to be nearly redundant

The last remaining per-card payload besides the indices: the **norm scalars** (the magnitude half of
every catalog entry — Level 5a). The 352-bit card charged 8 bits each. Two discoveries collapsed them.

**9a. Half of them were duplicates all along.** Recall Level 3's storage detail: the singular value
gets split as a square root into both factors (`uf = u·√σ`, `vf = v·√σ`). Since `u` and `v` are unit
vectors, scaling each by `√σ` gives both scaled factors length exactly `√σ` — **the same number**.
With numbers: σ = 0.16 → √σ = 0.4 → `uf` has length 0.4 and `vf` has length 0.4, always, for every
card, by construction. The engine had nevertheless been storing both lengths. One norm per matrix
suffices: 4 WKV norms are really 2, and 16 bits vanish by pure accounting — found by staring at the
algebra, not by any experiment.

**9b. The survivors barely need bits.** Quantize each remaining norm log-uniformly — i.e. store
*which power-of-two band it falls in* — over a **fixed, corpus-derived range**: measurements across
the whole corpus show the WKV `√σ` always lands between `2⁻³ = 0.125` and `2⁰ = 1`, and the shift
norms are pinned by the network's internal normalization into an even narrower band.

**Worked example (WKV norm, 2 bits).** The range `[2⁻³, 2⁰]` at 2 bits gets 4 representable values:
`2⁻³, 2⁻², 2⁻¹, 2⁰` — that is `0.125, 0.25, 0.5, 1.0`, one per octave. To store a norm of `0.37`:
its logarithm is `log₂(0.37) = −1.43`, the nearest band is `−1`, so store band index 2 (2 bits) and
reconstruct as `2⁻¹ = 0.5`. Yes — `0.37` comes back as `0.5`, a 35% error on the magnitude, *and
the log-loss doesn't move*. At 1 bit the bands are just `{0.125, 1.0}` and `0.37` comes back as
`1.0` (nearest in log space) — still nothing. Matching still uses the exact norm (so the *selection*
of catalog entries is unaffected); only the reconstruction magnitude is snapped. Measured, deployed
on the real engine, 400 users:

```
norm bits:   5        4        3        2
imm:      +0.0022  +0.0022  +0.0023  +0.0023     ← identical within noise
```

And the endgame of the lever, measured the same way: **1-bit norms** (each norm = one of just *two*
values) are *still* free — +0.0023/+0.0005 — while **0 bits** (a fixed constant norm) finally breaks
the pattern, failing decisively at +0.0064/+0.0036. So the norm axis bottoms out at exactly **one
bit per scalar**. The information content of the state is almost entirely in the *directions* (the
catalog indices); the magnitudes are nearly — but not quite — determined (layernorm pins the shifts;
the recurrence's normalization concentrates `√σ`; the last bit of "big or small?" is the part that
genuinely matters).

One training-side detail made 9b deployable at the frontier: the norm snap is **modeled inside QAT**
too (same theme as everything above — train ≈ deploy, exactly). As pure post-training quantization
the int4 norm snap cost +0.0005 imm — enough to fail the 88-bit rung by a hair; with the snap in the
training forward, the cost fell to ~+0.0002 and the rung passed.

### The 72-bit card, counted exactly

One last twist first. All the rungs above chop each 16-entry WKV direction into 2 chunks with
separate catalogs (the "product" trick of Level 5b). That trick exists because a whole-vector
catalog would need to be enormous — for a *big* vector. But 16 dimensions with 32 entries is not
enormous, and a single **joint** catalog over the whole direction can capture patterns *between*
the halves that two independent chunk catalogs cannot (the halves of real directions are
correlated; the product form is blind to that). Measured: one 32-entry joint catalog (5 bits per
direction) beats the two 8-entry chunk catalogs (6 bits per direction) on BOTH quality and size —
+0.0018 vs +0.0023 imm. Chunking is a compromise for high dimensions, and at 16 dimensions it
turns out we didn't need the compromise.

| piece | how | bits |
|---|---|---|
| 4 WKV direction indices | one 32-entry JOINT catalog per role | 4 × 5 = **20** |
| 2 WKV norms (deduped, 9a) | Level 9: 1-bit log₂ band | 2 × 1 = **2** |
| 2 token-shift vectors | Level 7: 64-entry catalogs, 4 chunks × 6 b | 2 × 24 = **48** |
| 2 shift norms | Level 9: 1-bit log₂ band | 2 × 1 = **2** |
| **card total** | | **20 + 2 + 48 + 2 = 72** |

**Nine bytes per card**; a note (3 layers) = 216 bits = 27 bytes. Against the 18,432-bit raw
state: **256×**. Log-loss degradation +0.0018/+0.0016 — measured on the deployed Rust engine over
400 held-out users, with the best per-user robustness profile since the 144-bit rung.

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

("Frobenius error" is just the Interlude's length recipe applied to the *difference* between the
reconstructed and original matrices: subtract entry by entry, square, add, root — a single number
for "how different the two matrices look".)

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

Think of it as practicing in the gear you'll compete in: the network trains while wearing the exact
compression it will be deployed with, so nothing about deployment surprises it.

There's one technical obstacle. Training works by gradients — "if this number were a hair bigger,
the loss would change by this much" — and a snap-to-grid step breaks that: nudge the input `0.03` a
hair and the output stays parked at `0.06` (Level 1's example), so the measured slope is zero and no
learning signal gets through. The fix is the **straight-through estimator (STE)**: the forward pass
uses the snapped value `0.06` (real deploy behavior), but the backward pass *pretends the snap was
the identity function* — a gradient of "make it bigger by 0.001" arriving at the snapped value is
passed straight through to the unsnapped `0.03` unchanged. Forward = honest; backward = smooth
enough to train. It's a lie, but a small and famously effective one.

```
forward :  state ─▶ [rank-1 + codebook snap] ─▶ rest of net ─▶ loss
backward:  grad  ◀────── (straight through) ──────────────◀ grad
```

Two findings made the 352-bit scheme work:

1. **Train ≈ deploy, exactly.** Fake-quantizing something *similar* to deployment teaches the wrong
   robustness. The compression in the QAT forward is verified bit-compatible with the Rust deploy
   path to ~1e-7.
2. **Train long enough.** With a short fine-tune, PQ passed most of its cost as a drifted base model.
   Training longer let the base *recover under the compressed regime* — the degradation fell
   monotonically with fine-tune length at every point measured (0.05 → 1.5 epochs, never turning
   around), to the point where the PQ-compressed model **beats its own uncompressed weights** (the
   compression acts as a familiar, trained-for representation, not an injury — and that is how the
   352-bit scheme manages a *negative* degradation on the forgetting-curve head: the deployed
   compressed model ends up marginally better there than the original fp32 champion). Numbers at
   352 b: PTQ +0.0046/+0.0040 → 0.1-epoch QAT +0.0043/+0.0037 → 0.75 ep +0.0021/+0.0012 → **1.5 ep
   +0.0010/−0.0003** (confirmed on the held-out dev split, +0.0009/+0.0003).

A third rule was imposed for the second descent (Levels 6–9) and shaped it: **the fine-tune budget is
hard-capped at 2 epochs** — quality beyond that must come from *learnable parameters* (Level 8), not
longer training. And the "train ≈ deploy, exactly" principle kept earning: the two rungs that
initially failed by a hair (88 b, 96 b) were exactly the ones where some deploy detail (the norm
snap, the shift catalog size) was *not yet* modeled in the training forward; modeling the norm snap
turned the 88-bit near-miss (+0.0026) into a pass (+0.0023).

---

## The whole pipeline, end to end

```
                 ┌────────────────────  PER CARD, PER LAYER, PER HEAD, EVERY REVIEW  ────────────────────┐
  16×16 WKV state A
        │
        ▼
  power iteration  →  dominant direction u, then its partner factor v + singular value σ  (Interlude)
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

  …and separately: token-shift vectors → normalize, PQ-encode against    (Level 7, 48 b + norms)
  the learned shift catalogs (4 chunks × 6-bit index per vector)
```

The bit ladder, selected rungs (all measured on 400 held-out users; gate = ≤ +0.0025 on both heads):

| scheme | card bits | log-loss degradation (imm / ahead) |
|---|---|---|
| raw fp32 | 18,432 | 0 (reference) |
| Level 4: rank-1 int4 + QAT | 512 | +0.0024 / +0.0021 |
| Level 5: rank-1 PQ (256-entry catalogs) + QAT | 352 | **+0.0010 / −0.0003** |
| Level 6: 64-entry WKV catalogs + int3 shifts | 272 | +0.0011 / −0.0003 |
| ≤ 256-bit goal (the project's founding target) | 256 | +0.0014 / −0.0001 |
| Level 7: PQ shifts (fixed catalog) | 144 | +0.0017 / +0.0000 |
| Level 8: + learnable shift catalog | 144 | +0.0013 / −0.0001 |
| Levels 6+7+8+9 combined (8-entry WKV catalogs, modeled norms) | 88 | +0.0023 / +0.0006 |
| Level 9 endgame: 1-bit norms | 76 | +0.0023 / +0.0005 |
| **Joint WKV catalog (comfortable champion)** | **72** | **+0.0018 / +0.0016** |
| Learned rotation + 32-entry shift catalogs (boundary) | 64 | +0.002492 / +0.0012 |

Every rung above passes the gate — though the 64-bit rung only just: its margin is +0.000008 on
the immediate head and its per-user profile is the tightest of the ladder (power users average
slightly over the gate), which is why 72 bits is called the *comfortable* champion. Three instructive failures bracket the frontier: 32-entry shift
catalogs (80 b via the shift route) fail at +0.0027 *even with everything learnable* — a capacity
wall; the 96-bit attempt via 16-entry shift catalogs failed the same way; and **0-bit (fixed) norms
fail decisively at +0.0064** — the norm axis bottoms out at one bit. The descent below 88 b happened
entirely on the norm axis, and every rung of it was pure post-training quantization: no retraining,
just deploying the 88-bit QAT weights with coarser and coarser norms.

---

## Cheat-sheet / glossary

- **Quantize (Level 1)** — snap numbers to an integer grid; store integers + a scale; `x ≈ q·s`.
- **qmax / int-N** — grid limit: int4 = ±7, int2 = ±1.
- **Per-column scaling (Level 2)** — one scale per column/sub-vector instead of one global; "more,
  smaller scales = less damage".
- **Vector / matrix** — a list of numbers / a rectangular grid of numbers (Interlude).
- **Rank-1 matrix / outer product** — a matrix whose every entry = (row's factor number) ×
  (column's factor number); fully rebuilt from its two factor vectors (Interlude, Idea 1).
- **Norm** — a vector's length: square entries, add, square-root. `[3,4]` has norm 5 (Idea 2).
- **SVD** — the theorem that any matrix is a sum of rank-1 pieces, sorted biggest-first (Idea 3).
- **Singular value (σ)** — the importance weight of one rank-1 piece in that sum.
- **Eckart–Young** — truncated SVD is the *best possible* rank-r approximation (in least squares).
- **Power iteration** — repeat `u ← normalize(A Aᵀ u)` to get the top direction cheaply.
- **Low-rank (Level 3)** — store `σ·u·vᵀ` (two directions + magnitude) instead of the full matrix.
- **Token shift** — every block blends the current activation with the *previous step's*; the two
  32-value "previous activation" vectors per layer must therefore persist between reviews. In the
  352-bit era: int4 + one scale per vector (256 bits, the then-binding wall). Since Level 7: PQ
  against learned catalogs (48 bits of indices).
- **Level 4** — low-rank factors, int4-quantized per column: 512 b/card.
- **PQ / product quantization (Level 5)** — chop a unit direction into chunks; replace each chunk by
  its nearest catalog entry; store the indices + the norm. Combinations multiply: two 256-entry
  catalogs cover 65,536 direction combinations.
- **Codebook / centroid** — the global catalog of typical chunks, built once by k-means on a corpus
  of real states; ships in the app, costs no per-card bits.
- **Sign canonicalization** — flip `(u, v)` together so u's dominant entry is positive; `u vᵀ`
  unchanged, catalog coverage doubled.
- **PTQ vs QAT** — compress after training vs fine-tune with the exact compression in the forward pass.
- **STE** — forward uses the snapped value, backward passes gradients through unchanged.
- **Log-loss, not Frobenius** — judge only by recall-prediction loss; matrix distance anti-correlates.
- **Coarse catalogs under QAT (Level 6)** — shrinking WKV catalogs 256 → 8 entries (8 → 3-bit
  indices) is ~free once the net trains through the snap: the data walks over to the palette.
- **Shift-PQ (Level 7)** — the shifts get the catalog treatment too; PQ at 80 b beat int2 at 128 b.
  Separate catalogs per role (time-mix / channel-mix) and chunk position; 64 entries is the floor.
- **Capacity wall vs optimization gap** — a scheme that fails *identically* with and without every
  learnable lever is out of representational capacity (32-entry shift catalogs); no training fixes it.
- **Learnable catalog (Level 8)** — centroids as trainable parameters: frozen hard selection,
  embedding-style gradients into the selected entries. Worth −0.0004 at equal bits (shift catalog).
  Post-hoc k-means refit is the dead version of this idea — swap-after-training hurts.
- **Norm dedup (Level 9a)** — split-√σ factors have equal norms *by construction*; store one per
  head, save 16 bits, cost exactly zero.
- **Log₂ norm quant (Level 9b)** — norms stored as coarse power-of-two bands over fixed
  corpus-derived ranges; match by true norm, reconstruct by the snapped one. 1-bit norms measure
  identical to 5-bit; 0 bits (a constant) fails decisively — the floor is exactly one bit. The
  information is in the directions, not the magnitudes.
- **Finite differences can't check STE** — the true function's local derivative through downstream
  re-quantization is ~zero (staircases); validate STE gradients against an autograd port of the same
  semantics, not against numeric differentiation.

---

*Grounded in the deployed code: `engine/src/model.rs` (`compress_wkv_state`, `PqCodebook`
incl. the norm quant, `RWKV_PQ_NORM_BITS`), `scratchpad/pq_train.py` + `pq_train_shift.py`
(catalog k-means), the QAT kernel in `gpu_train/rwkv/model/csrc/cuda/rwkv7_cuda.cu`
(`qat_lr_rank1`, learnable-catalog gradients), and `gpu_train/rwkv/model/rwkv_model.py`
(`fake_pq_shift`, learnable shift catalog). Real numbers computed by
`scratchpad/pq_explainer_numbers.py` on a real WKV state; every log-loss above is a 400-user
held-out measurement recorded in `research_log_h2k16.md`.*
