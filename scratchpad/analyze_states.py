"""Offline characterization of real card/note WKV states.

Reads `STATE <1024 floats>` lines (32x32 per head; note: dump is one HEAD's KxK? -- the dump emits
the full (H,K,K) flattened, so length = H*K*K). We reshape to (H,K,K) using K from the line length /
H if H known; here the dump is a single stream's t_state[0] = (H,K,K). We infer H*K*K = len, K=32.

This is for UNDERSTANDING ONLY (rank energy, factor distributions). The decision is log-loss in Rust.
"""
import sys
import numpy as np

K = 32


def load(path):
    mats = []
    with open(path) as f:
        for line in f:
            if not line.startswith("STATE"):
                continue
            vals = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
            n = vals.size
            if n % (K * K) != 0:
                continue
            h = n // (K * K)
            mats.append(vals.reshape(h, K, K))
    return mats  # list of (H,K,K)


def main():
    path = sys.argv[1]
    mats = load(path)
    if not mats:
        print("no states loaded")
        return
    H = mats[0].shape[0]
    print(f"loaded {len(mats)} states, H={H}, K={K}  (each = {H} heads of {K}x{K})")

    # ---- singular value spectrum per head matrix ----
    # energy fraction in top-1, top-2 components (this bounds what rank-r truncation can capture).
    e1, e2, e_tail = [], [], []
    sig_ratio = []  # sigma2/sigma1
    allmats = []
    for m in mats:
        for h in range(H):
            A = m[h]
            if not np.all(np.isfinite(A)) or np.abs(A).max() < 1e-12:
                continue
            s = np.linalg.svd(A, compute_uv=False)
            tot = (s ** 2).sum()
            if tot <= 0:
                continue
            e1.append(s[0] ** 2 / tot)
            e2.append((s[0] ** 2 + s[1] ** 2) / tot)
            e_tail.append(1.0 - (s[0] ** 2 + s[1] ** 2) / tot)
            sig_ratio.append(s[1] / s[0] if s[0] > 0 else 0)
            allmats.append(A)
    e1 = np.array(e1); e2 = np.array(e2); sig_ratio = np.array(sig_ratio)
    print(f"\n=== SV energy ({len(e1)} head-matrices) ===")
    print(f"  top-1 energy frac:  mean {e1.mean():.4f}  median {np.median(e1):.4f}  p10 {np.percentile(e1,10):.4f}  p90 {np.percentile(e1,90):.4f}")
    print(f"  top-2 energy frac:  mean {e2.mean():.4f}  median {np.median(e2):.4f}  p10 {np.percentile(e2,10):.4f}")
    print(f"  sigma2/sigma1:      mean {sig_ratio.mean():.4f}  median {np.median(sig_ratio):.4f}  p90 {np.percentile(sig_ratio,90):.4f}")
    print(f"  -> rank-1 loses {(1-e1).mean()*100:.2f}% of energy on avg; rank-2 loses {(1-e2).mean()*100:.2f}%")

    # ---- factor entry distribution (rank-2 SVD split sqrt(sigma) into both, per-col scaled) ----
    # mirror lowrank_roundtrip: uf[:,j]=u_j*sqrt(s_j), vf[:,j]=v_j*sqrt(s_j); then per-col amax scale.
    all_u_norm = []  # entries normalized by their column amax (the quantization grid is [-1,1])
    near_zero_frac = []  # fraction of |entry/amax| < 0.25 (would round to 0 in ternary)
    for A in allmats:
        U, s, Vt = np.linalg.svd(A)
        for j in range(2):
            uf = U[:, j] * np.sqrt(s[j])
            vf = Vt[j, :] * np.sqrt(s[j])
            for fac in (uf, vf):
                amax = np.abs(fac).max()
                if amax < 1e-20:
                    continue
                norm = fac / amax
                all_u_norm.append(norm)
                near_zero_frac.append(np.mean(np.abs(norm) < 0.25))
    alln = np.concatenate(all_u_norm)
    nz = np.array(near_zero_frac)
    print(f"\n=== factor entries normalized to per-col grid [-1,1] ({alln.size} entries) ===")
    # ternary rounds x/amax to {-1,0,1}: |.|<0.5 -> 0, else +-1. histogram of |normalized|:
    bins = [0, 0.1, 0.25, 0.5, 0.75, 1.01]
    hist, _ = np.histogram(np.abs(alln), bins=bins)
    for i in range(len(bins) - 1):
        print(f"  |x/amax| in [{bins[i]:.2f},{bins[i+1]:.2f}): {hist[i]/alln.size*100:5.1f}%")
    print(f"  fraction rounding to ZERO under ternary (|x/amax|<0.5): {np.mean(np.abs(alln)<0.5)*100:.1f}%")
    print(f"  per-column near-zero(<0.25) frac: mean {nz.mean():.3f}")

    # ---- Lloyd-Max 4-level (keep zero) vs ternary: what levels would k-means pick? ----
    # symmetric: fit on |x|, mirror. We want to see if a non-uniform grid w/ zero beats uniform ternary.
    absx = np.abs(alln)
    # candidate symmetric 4-level grids including 0: {-b,-a,0,a,b}? that's 5. 2-bit=4 codes.
    # report the magnitude quantiles to inform level placement.
    print(f"\n  |normalized| quantiles: ", {q: round(float(np.percentile(absx, q)), 3) for q in (25, 50, 75, 90, 95)})


if __name__ == "__main__":
    main()
