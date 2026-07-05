"""Parity for the QAT-modeled NORM QUANT (RWKV_QAT_NORM_BITS) vs the engine RWKV_PQ_NORM_BITS semantics.
Two halves:
  A) CUDA WKV path: qat_lr_rank1 PQ branch + rwkv7_set_norm_quant(4,-3,0) vs a numpy port of deploy
     compress_wkv_state r==1 + PqCodebook::encode_decode WITH the norm quantized (match by TRUE norm,
     reconstruct by quantized norm) — the parity_lr_pq.py harness + the norm-quant twist.
  B) Python shift path: fake_pq_shift with RWKV_QAT_NORM_BITS=4 vs the same numpy mirror on the shift
     codebook (range [2.2,2.9] octaves), on synthetic C=32 vectors with realistic norms (2^2.2..2^2.9).
Near-tie centroid flips (f32 vs f64) are the same accepted class as parity_lr_pq/parity_shift_pq.
Usage: RWKV_QAT_NORM_BITS=4 python scratchpad/parity_norm_quant.py [wkv_cb] [shift_cb]"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
os.environ.setdefault("RWKV_QAT_NORM_BITS", "4")
BITS = int(os.environ["RWKV_QAT_NORM_BITS"])
import rwkv, rwkv.model  # noqa: registers torch.ops.rwkv

def load_cb(path, roles):
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, k, ncent = (int(x) for x in lines[0].split()[:5])
    rows = [[float(x) for x in ln.split()] for ln in lines[1:]]
    cb, idx = {}, 0
    for role in range(roles):
        cb[role] = []
        for _ in range(m):
            cb[role].append(np.array(rows[idx:idx + ncent], dtype=np.float64))
            idx += ncent
    return m, sub, ncent, cb

def nq(norm, lo, hi):  # engine norm quant: log2-uniform, HALF-AWAY round, clamp, exp2
    levels = float((1 << BITS) - 1)
    t = (np.log2(norm) - lo) / (hi - lo)
    q = min(max(np.floor(t * levels + 0.5), 0.0), levels)
    return float(np.exp2(lo + q / levels * (hi - lo)))

def encode_decode(role, col, cb, lo, hi):  # engine mirror WITH norm quant
    norm = float(np.linalg.norm(col))
    if not np.isfinite(norm) or norm < 1e-20:
        return col
    inv = 1.0 / norm                        # match by TRUE norm
    normq = nq(norm, lo, hi)                # reconstruct by quantized norm
    out = col.copy()
    for p, cents in enumerate(cb[role]):
        sub = cents.shape[1]; s = p * sub
        diff = (col[s:s + sub] * inv)[None, :] - cents
        best = int(np.argmin((diff * diff).sum(1)))
        out[s:s + sub] = cents[best] * normq
    return out

def deploy_rank1_pq(A, cb, iters=64):
    A = A.astype(np.float64); K = A.shape[0]
    amax = float(np.max(np.abs(A))); scale = amax if (np.isfinite(amax) and amax > 1e-30) else 1.0
    an = A / scale; u = np.full(K, 1.0 / np.sqrt(K))
    for _ in range(iters):
        nu = an @ (an.T @ u); nrm = np.linalg.norm(nu)
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        nu = nu / nrm; dot = abs(u @ nu); u = nu
        if 1.0 - dot < 1e-7:
            break
    v_un = A.T @ u; sigma = float(np.linalg.norm(v_un))
    uf = np.zeros(K); vf = np.zeros(K)
    if sigma > 1e-20 and np.isfinite(sigma) and np.all(np.isfinite(u)):
        sj = np.sqrt(sigma); uf = u * sj; vf = (v_un / sigma) * sj
    ai = int(np.argmax(np.abs(uf)))
    if uf[ai] < 0:
        uf, vf = -uf, -vf
    uf = encode_decode(0, uf, cb, -3.0, 0.0); vf = encode_decode(1, vf, cb, -3.0, 0.0)
    return np.outer(uf, vf)

# ---- A) CUDA WKV path --------------------------------------------------------------------------------
wkv_cb_path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/pq_cb_m2b3.txt"
m, sub, ncent, cb = load_cb(wkv_cb_path, 4)
flat = []
for role in range(2):
    for p in range(m):
        flat.extend(cb[role][p].reshape(-1).tolist())
cbt = torch.tensor(flat, dtype=torch.float32, device="cuda")
torch.ops.rwkv.rwkv7_set_pq_codebook(cbt, m, sub, ncent)
torch.ops.rwkv.rwkv7_set_norm_quant(cbt, BITS, -3.0, 0.0)
print(f"A) uploaded {wkv_cb_path} (m={m} sub={sub} ncent={ncent}) + norm quant int{BITS} [-3,0]")

torch.manual_seed(0)
B, H, K = 16, 2, 16
s = torch.zeros(B, H, K, K)
for b in range(B):
    for h in range(H):
        for _ in range(3):
            s[b, h] += torch.outer(torch.randn(K), torch.randn(K))
        s[b, h] += 0.05 * torch.randn(K, K)
s[0, 0] *= 1e4; s[1, 0] *= 1e-6; s[2, 0] = 0.0
cuda_out = torch.ops.rwkv.rwkv7_lr_trunc_test_float.default(s.to("cuda").contiguous(), 7.0).cpu()
a = s.numpy(); ref = np.empty((B, H, K, K))
for b in range(B):
    for h in range(H):
        ref[b, h] = deploy_rank1_pq(a[b, h], cb)
ref = torch.from_numpy(ref).float()
rel = (cuda_out - ref).abs() / ref.abs().amax(dim=[2, 3], keepdim=True).clamp_min(1e-9)
permat = rel.reshape(B * H, K, K).amax(dim=[1, 2])
ok_a = (permat < 1e-3).sum().item()
print(f"A) CUDA-vs-ref: maxREL {rel.max():.3e}  #matrices<1e-3: {ok_a}/{B*H}  finite: {torch.isfinite(cuda_out).all().item()}")
torch.ops.rwkv.rwkv7_set_norm_quant(cbt, 0, 0.0, 0.0)  # reset global state for anything after us

# ---- B) Python shift path ----------------------------------------------------------------------------
from rwkv.model import rwkv_model as RM
shift_cb_path = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/pq_cb_shift_m4b6.txt"
ms, subs, ncents, cbs = load_cb(shift_cb_path, 2)
RM._SHIFT_PQ_PATH = shift_cb_path
RM._SHIFT_PQ_CB = None  # force (re)load from our path on first fake_pq_shift call
assert RM._NORM_BITS == BITS, f"rwkv_model._NORM_BITS={RM._NORM_BITS} != {BITS} (env not picked up)"
rng = np.random.default_rng(1)
C = ms * subs
X = rng.standard_normal((400, C)).astype(np.float32)
X *= (np.exp2(rng.uniform(2.2, 2.9, size=(400, 1))).astype(np.float32) / np.linalg.norm(X, axis=1, keepdims=True))
agree, flips = [], 0
for role in range(2):
    qt = RM.fake_pq_shift(torch.from_numpy(X).unsqueeze(0), role).squeeze(0).numpy()
    for i in range(X.shape[0]):
        r = encode_decode(role, X[i].astype(np.float64), cbs, 2.2, 2.9)
        relm = np.abs(qt[i] - r).max() / max(np.abs(r).max(), 1e-12)
        if relm < 1e-3:
            agree.append(relm)
        else:
            flips += 1
print(f"B) shift fake_pq_shift-vs-ref: n=800  agree<1e-3: {len(agree)}  flips: {flips}  maxrel(agree): {max(agree):.3e}")
assert ok_a >= B * H - 4 and len(agree) >= 800 * 0.98, "PARITY MISMATCH - do not train"
print("NORM_QUANT_PARITY_PASS")
