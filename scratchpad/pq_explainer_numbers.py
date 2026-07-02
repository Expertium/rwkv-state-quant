"""Pull concrete numbers from a REAL RWKV WKV state for the PQ explainer doc."""
import numpy as np, glob, os
np.set_printoptions(precision=3, suppress=True, linewidth=120)

# grab the first finite STATE from a card corpus dump (each line = h*k*k = 2*16*16 = 512 floats)
f = sorted(glob.glob("scratchpad/corpus/card_*.txt"))[0]
K = 16
state = None
with open(f) as fh:
    for line in fh:
        if line.startswith("STATE"):
            v = np.fromstring(line[5:], sep=" ", dtype=np.float64)
            if v.size == 2 * K * K and np.all(np.isfinite(v)) and np.abs(v).max() > 1e-6:
                state = v.reshape(2, K, K)
                break
print(f"source: {os.path.basename(f)}   shape={state.shape}")
A = state[0]  # head 0, 16x16
print(f"\nHead-0 matrix stats: max|A|={np.abs(A).max():.4f}  mean|A|={np.abs(A).mean():.4f}")

# SVD -> singular values (motivates low-rank)
U, S, Vt = np.linalg.svd(A)
print("\nSingular values (top 8):", S[:8])
print("energy captured by rank-1: {:.1%}   rank-2: {:.1%}".format(
    S[0]**2 / (S**2).sum(), (S[:2]**2).sum() / (S**2).sum()))
r1 = S[0] * np.outer(U[:, 0], Vt[0])
print(f"rank-1 Frobenius error ||A - A1||/||A|| = {np.linalg.norm(A-r1)/np.linalg.norm(A):.3f}")

# rank-1 factors, split-sqrt (as the engine does): uf = u*sqrt(sigma), vf = v*sqrt(sigma)
sigma = S[0]; u = U[:, 0].copy(); v = Vt[0].copy()
# sign-canon: dominant-abs entry of u positive
ai = np.argmax(np.abs(u))
if u[ai] < 0:
    u, v = -u, -v
print(f"\ntop singular value sigma = {sigma:.4f}")
print("u (left dir, unit):", np.round(u, 3))
print("v (right dir, unit):", np.round(v, 3))
print("u split into m=2 sub-vectors of dim 8:")
print("  sub0:", np.round(u[:8], 3))
print("  sub1:", np.round(u[8:], 3))
print(f"u norm check: {np.linalg.norm(u):.4f}  (unit)")

# tiny illustration of nearest-centroid on a 4-dim toy (for the doc's worked example)
print("\n--- toy 4-dim PQ example ---")
d = np.array([0.30, 0.62, 0.10, -0.72]); d = d / np.linalg.norm(d)
cents = np.array([[0.5,0.5,0.5,0.5],[0.26,0.53,0.09,-0.80],[-0.7,0.1,0.7,0.0],[0.0,0.71,0.0,-0.71]])
cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)
dists = ((d[None] - cents)**2).sum(1)
print("unit direction d:", np.round(d, 3))
print("centroid dists:", np.round(dists, 4), "-> nearest =", int(np.argmin(dists)))
print("chosen centroid:", np.round(cents[np.argmin(dists)], 3))
