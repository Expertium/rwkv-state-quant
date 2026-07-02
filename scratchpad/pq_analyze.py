#!/usr/bin/env python
"""PQ GO/NO-GO clustering analysis.

Reads a corpus of WKV states (dumped via `rwkv-infer --dump-corpus <user> <card|note> [stride]`, each line
`STATE <h*k*k floats>`), replicates the ENGINE's rank-2 factorization (model.rs::compress_wkv_state), extracts
the factor DIRECTION sub-vectors, and asks the necessary-condition question for product quantization:

    Do the factor directions CLUSTER enough that a learned codebook could beat uniform int2?

We compare, on HELD-OUT directions, the quantization MSE of a k-means codebook vs uniform int2 at matched (or
fewer) bits. NOTE: MSE is anti-correlated with log-loss here (project lesson), so a codebook WIN in MSE is only
a NECESSARY condition (green-light to build + test in real log-loss); a codebook that can't even beat uniform
in MSE is a hard NO-GO (directions don't cluster -> PQ cannot help). Usage:

    python scratchpad/pq_analyze.py <corpus_file...>   [--k 16 --h 2 --m 2 --bits 8]
"""
import sys, numpy as np

def parse_args(argv):
    files, opts = [], {"k": 16, "h": 2, "m": 2, "bits": 8}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            opts[a[2:]] = int(argv[i + 1]); i += 2
        else:
            files.append(a); i += 1
    return files, opts

def load_states(files, h, k):
    hkk = h * k * k
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.startswith("STATE"):
                    continue
                vals = np.fromstring(line[5:], sep=" ", dtype=np.float32)
                if vals.size == hkk and np.all(np.isfinite(vals)):
                    rows.append(vals)
    return np.array(rows, dtype=np.float32).reshape(-1, h, k, k)

def rank2_factors(a, k):
    """Match model.rs::compress_wkv_state rank-2 path. Returns 4 direction columns (u1,v1,u2,v2), each
    UNIT-normalized + sign-canonicalized (u's max-abs entry positive; v flipped with u to keep u v^T sign)."""
    amax = np.abs(a).max()
    scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = a / scale
    gram = an @ an.T
    w, V = np.linalg.eigh(gram)  # ascending
    order = np.argsort(w)[::-1]
    dirs = []
    for j in range(2):
        col = order[j]; ev = w[col]
        if not np.isfinite(ev) or ev <= 0:
            dirs += [np.zeros(k, np.float32), np.zeros(k, np.float32)]
            continue
        sigma = np.sqrt(ev) * scale
        u = V[:, col].astype(np.float32)
        v_un = a.T @ u                      # = sigma * v
        v = (v_un / sigma).astype(np.float32)
        # sign canon: flip u so its dominant entry is +, flip v with it (u v^T invariant)
        s = 1.0 if u[np.argmax(np.abs(u))] >= 0 else -1.0
        u, v = u * s, v * s
        for d in (u, v):
            n = np.linalg.norm(d)
            dirs.append(d / n if n > 1e-20 else d)
    return dirs  # [u1, v1, u2, v2]

def collect_directions(states, k):
    """Return dict role -> (Ndir, k) array of unit directions, role in {u1,v1,u2,v2}."""
    roles = {r: [] for r in ("u1", "v1", "u2", "v2")}
    for st in states:
        for hh in range(st.shape[0]):
            u1, v1, u2, v2 = rank2_factors(st[hh], k)
            for r, d in zip(("u1", "v1", "u2", "v2"), (u1, v1, u2, v2)):
                if np.linalg.norm(d) > 1e-6:
                    roles[r].append(d)
    return {r: np.array(v, np.float32) for r, v in roles.items() if v}

def kmeans(X, ncent, seed=0):
    from sklearn.cluster import KMeans
    n = min(ncent, len(X))
    km = KMeans(n_clusters=n, n_init=3, max_iter=50, random_state=seed).fit(X)
    return km.cluster_centers_.astype(np.float32)

def quant_mse_codebook(Xtr, Xte, m, bits):
    """PQ: split k-dim into m sub-vecs, k-means codebook (2^bits) per position, MSE on held-out."""
    k = Xtr.shape[1]; sub = k // m
    mse = 0.0
    for p in range(m):
        s = slice(p * sub, (p + 1) * sub)
        C = kmeans(Xtr[:, s], 2 ** bits)
        d = ((Xte[:, s][:, None, :] - C[None, :, :]) ** 2).sum(-1)
        mse += d.min(1).sum()
    return mse / (len(Xte) * k)

def quant_mse_uniform_int2(Xte):
    """Per-vector amax-scaled ternary {-1,0,1} (matches percol int2), MSE per entry."""
    amax = np.abs(Xte).max(1, keepdims=True); scale = np.maximum(amax, 1e-12)
    q = np.round(Xte / scale).clip(-1, 1) * scale
    return ((Xte - q) ** 2).mean()

def main():
    files, o = parse_args(sys.argv[1:])
    if not files:
        print(__doc__); return
    k, h, m, bits = o["k"], o["h"], o["m"], o["bits"]
    states = load_states(files, h, k)
    print(f"loaded {len(states)} states ({h}x{k}x{k})  ->  {len(states)*h} head-matrices")
    dirs = collect_directions(states, k)
    print(f"PQ config: split k={k} into m={m} sub-vecs of {k//m}, codebook 2^{bits}={2**bits} "
          f"-> {m*bits} bits/direction ({m*bits/k:.2f} bits/value)")
    print(f"{'role':>5} {'Ndir':>7} {'uni_int2_MSE':>13} {'codebook_MSE':>13} {'ratio(cb/uni)':>13}  verdict")
    for r, X in dirs.items():
        # subsample for speed (GO/NO-GO needs ~30k, not all)
        rng = np.random.default_rng(1)
        if len(X) > 40000:
            X = X[rng.choice(len(X), 40000, replace=False)]
        n = len(X); ntr = int(n * 0.8)
        perm = rng.permutation(n)
        Xtr, Xte = X[perm[:ntr]], X[perm[ntr:]]
        uni = quant_mse_uniform_int2(Xte)
        cb = quant_mse_codebook(Xtr, Xte, m, bits)
        ratio = cb / uni if uni > 0 else float("inf")
        verdict = "CLUSTERS (cb<uni)" if ratio < 1.0 else "no cluster win"
        print(f"{r:>5} {n:>7} {uni:>13.6f} {cb:>13.6f} {ratio:>13.3f}  {verdict}")
    print("\nGO if codebook MSE < uniform int2 MSE at FEWER bits/value (necessary, not sufficient -> confirm in log-loss).")

if __name__ == "__main__":
    main()
