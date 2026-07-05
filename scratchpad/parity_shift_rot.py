"""Parity: python fake_pq_shift WITH a random learned rotation vs a numpy mirror of the engine path
(rot_apply forward -> PqCodebook::encode_decode -> rot_apply transpose). Catches R/R^T orientation
mismatches between the two implementations. Norm quant modeled at int1 (deploy frontier config).
Usage: RWKV_QAT_NORM_BITS=1 python scratchpad/parity_shift_rot.py <shift_cb> [n]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
os.environ.setdefault("RWKV_QAT_NORM_BITS", "1")
os.environ["RWKV_QAT_SHIFT_ROT"] = "1"
import numpy as np
import torch
from rwkv.model import rwkv_model as RM

cb_path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/pq_cb_shift_m4b6.txt"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300

with open(cb_path) as fh:
    lines = [ln for ln in fh if ln.strip()]
m, bits, sub, C, ncent = (int(x) for x in lines[0].split()[:5])
rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]], np.float64).reshape(2, m, ncent, sub)

RM._SHIFT_PQ_PATH = cb_path
RM._SHIFT_PQ_CB = None
RM.shift_rot_init("cpu", C)
with torch.no_grad():
    RM._SHIFT_ROT_P.copy_(0.1 * torch.randn(2, C, C, generator=torch.Generator().manual_seed(11)))

def nq1(norm):  # engine int1 norm quant, shift range
    lo, hi = 2.2, 2.9
    t = (np.log2(max(norm, 1e-30)) - lo) / (hi - lo)
    q = min(max(np.floor(t * 1.0 + 0.5), 0.0), 1.0)
    return float(np.exp2(lo + q * (hi - lo)))

def engine_mirror(role, x, R):
    y = R @ x                       # rot_apply forward: y_i = sum_j R[i,j] x_j
    norm = float(np.linalg.norm(y))
    if np.isfinite(norm) and norm >= 1e-20:
        inv = 1.0 / norm
        normq = nq1(norm)
        out = y.copy()
        for p in range(m):
            s = p * sub
            d = ((y[s:s+sub] * inv)[None, :] - rows[role, p]) ** 2
            best = int(np.argmin(d.sum(1)))
            out[s:s+sub] = rows[role, p][best] * normq
        y = out
    return R.T @ y                  # rot_apply transpose

rng = np.random.default_rng(2)
X = rng.standard_normal((n, C)).astype(np.float32)
X *= (np.exp2(rng.uniform(2.2, 2.9, size=(n, 1))).astype(np.float32) / np.linalg.norm(X, axis=1, keepdims=True))
agree, flips = [], 0
for role in range(2):
    with torch.no_grad():
        R64 = RM._shift_rot_matrix(role).double().numpy()
    with torch.no_grad():
        qt = RM.fake_pq_shift(torch.from_numpy(X).unsqueeze(0), role).squeeze(0).numpy()
    for i in range(n):
        ref = engine_mirror(role, X[i].astype(np.float64), R64)
        rel = np.abs(qt[i] - ref).max() / max(np.abs(ref).max(), 1e-12)
        if rel < 1e-3:
            agree.append(rel)
        else:
            flips += 1
print(f"rot parity: n={2*n}  agree<1e-3: {len(agree)}  near-tie flips: {flips}  maxrel(agree): {max(agree):.3e}")
assert len(agree) >= 2 * n * 0.98, "ROTATION PARITY MISMATCH (orientation bug?)"
print("SHIFT_ROT_PARITY_PASS")
