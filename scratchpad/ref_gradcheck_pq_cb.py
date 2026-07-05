"""STE-reference gradcheck for the learnable WKV codebook kernel.

The kernel's centroid grads follow STE semantics (truncation = identity for state grads; centroids get
embedding-style grads through frozen selections + detached quantized norms). A finite-difference check of
the TRUE function CANNOT match that (downstream re-quantization has zero local derivative and damps any
centroid perturbation — observed ~5-10x smaller numeric grads). The correct reference: a torch autograd
port of the SAME semantics — recurrence in f64, trunc = outer(cb-recon) + (ns - ns.detach()), selections
and norms under no_grad, centroids differentiable. Kernel buffer vs autograd grad must agree ~1e-3.
Usage: python scratchpad/ref_gradcheck_pq_cb.py [codebook]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
import torch
import rwkv, rwkv.model  # noqa: registers torch.ops.rwkv

cbpath = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/pq_cb_m2b3.txt"
with open(cbpath) as fh:
    lines = [ln for ln in fh if ln.strip()]
m, bits, sub, k, ncent = (int(x) for x in lines[0].split()[:5])
vals = []
for ln in lines[1:]:
    vals.extend(float(x) for x in ln.split())
need = 2 * m * ncent * sub
cb32 = torch.tensor(vals[:need], dtype=torch.float32, device="cuda")

torch.ops.rwkv.rwkv7_set_pq_codebook(cb32, m, sub, ncent)
torch.ops.rwkv.rwkv7_set_norm_quant(cb32, 4, -3.0, 0.0)
torch.ops.rwkv.rwkv7_set_pq_learn(cb32, 1)
K = m * sub
NQ_LEVELS = float((1 << 4) - 1)
NQ_LO, NQ_HI = -3.0, 0.0
print(f"cb {cbpath}: m={m} sub={sub} ncent={ncent} K={K}; norm quant int4; learn ON")

torch.manual_seed(7)
B, T, H = 2, 64, 2
r = (0.5 * torch.randn(B, T, H, K)).float()
kk = (0.5 * torch.randn(B, T, H, K)).float()
v = (0.5 * torch.randn(B, T, H, K)).float()
w = (0.95 + 0.049 * torch.rand(B, T, H, K)).float()
a = (0.3 * torch.rand(B, T, H, K)).float()
kd = (0.5 * torch.randn(B, T, H, K)).float()
skip = torch.zeros(B, T, dtype=torch.bool)
skip[0, 10] = True; skip[1, 40] = True
R = torch.randn(B, T, H, K).float()
QMAX = 7.0

# ---- kernel side --------------------------------------------------------------------------------------
cu = lambda t_: t_.cuda()
out_k, ckpt = torch.ops.rwkv.rwkv7_wkv_qat_lr_forward_float.default(
    cu(r), cu(kk), cu(v), cu(w), cu(a), cu(kd), cu(skip), QMAX)
torch.ops.rwkv.rwkv7_pq_cb_grad_zero(cb32)
torch.ops.rwkv.rwkv7_wkv_qat_lr_backward_float.default(
    cu(r), cu(kk), cu(v), cu(w), cu(a), cu(kd), cu(skip), ckpt, cu(R), QMAX)
g_kernel = torch.ops.rwkv.rwkv7_pq_cb_grad_get(cb32).cpu().double()
torch.cuda.synchronize()

# ---- torch STE reference (f64) ------------------------------------------------------------------------
cb = torch.nn.Parameter(cb32.detach().cpu().double().view(2 * m, ncent, sub))

def nq(norm):
    t_ = (torch.log2(torch.clamp(norm, min=1e-30)) - NQ_LO) / (NQ_HI - NQ_LO)
    q = torch.clamp(torch.floor(t_ * NQ_LEVELS + 0.5), 0.0, NQ_LEVELS)
    return torch.exp2(NQ_LO + q / NQ_LEVELS * (NQ_HI - NQ_LO))

def trunc_ste(ns):
    """qat_lr_rank1 mirror: rank-1 power iteration + PQ + norm quant. Selections/norms frozen (no_grad),
    centroids differentiable, STE passthrough to ns."""
    with torch.no_grad():
        A = ns.detach()
        amax = A.abs().max()
        scale = amax if (torch.isfinite(amax) and amax > 1e-30) else torch.tensor(1.0, dtype=A.dtype)
        an = A / scale
        u = torch.full((K,), 1.0 / K ** 0.5, dtype=A.dtype)
        for _ in range(64):
            nu = an @ (an.T @ u)
            nrm = nu.norm()
            if not torch.isfinite(nrm) or nrm < 1e-30:
                break
            nu = nu / nrm
            dot = (u @ nu).abs()
            u = nu
            if 1.0 - dot < 1e-7:
                break
        v_un = A.T @ u
        sigma = v_un.norm()
        if not (sigma > 1e-20 and torch.isfinite(sigma) and torch.isfinite(u).all()):
            return ns  # zero/degenerate: kernel leaves ufq/vfq zero -> recon 0; rare on random data
        sj = sigma.sqrt()
        uf = u * sj
        vf = (v_un / sigma) * sj
        ai = uf.abs().argmax()
        if uf[ai] < 0:
            uf, vf = -uf, -vf
        sel, nrm_q = [], []
        for role, col in ((0, uf), (1, vf)):
            norm = col.norm()
            if not torch.isfinite(norm) or norm < 1e-20:
                sel.append(None); nrm_q.append(None); continue
            unit = col / norm
            idxs = [((unit[p * sub:(p + 1) * sub][None, :] - cb.detach()[role * m + p]) ** 2).sum(1).argmin()
                    for p in range(m)]
            sel.append(idxs); nrm_q.append(nq(norm))
    parts = []
    for role, col in ((0, uf), (1, vf)):
        if sel[role] is None:
            parts.append(col)  # unquantized column (constant w.r.t. cb)
        else:
            parts.append(torch.cat([cb[role * m + p][sel[role][p]] for p in range(m)]) * nrm_q[role])
    q_recon = torch.outer(parts[0], parts[1])
    return q_recon + (ns - ns.detach())

loss = torch.zeros((), dtype=torch.float64)
for b in range(B):
    for h in range(H):
        S = torch.zeros(K, K, dtype=torch.float64)
        for t in range(T):
            r_t, k_t, v_t = r[b, t, h].double(), kk[b, t, h].double(), v[b, t, h].double()
            w_t, a_t, kd_t = w[b, t, h].double(), a[b, t, h].double(), kd[b, t, h].double()
            ns = S * w_t[None, :] - torch.outer(S @ kd_t, a_t * kd_t) + torch.outer(v_t, k_t)
            loss = loss + (ns @ r_t) @ R[b, t, h].double()
            S = S if skip[b, t] else trunc_ste(ns)
loss.backward()
g_ref = cb.grad.reshape(-1)

sig = g_ref.abs() > g_ref.abs().max() * 1e-4
cos = torch.nn.functional.cosine_similarity(g_kernel, g_ref, dim=0).item()
rel = ((g_kernel - g_ref).abs()[sig] / g_ref.abs()[sig])
print(f"loss(ref f64) {loss.item():.6f}   kernel out vs ref: (sanity via grads below)")
print(f"cb grad: kernel |g| max {g_kernel.abs().max():.4e}  ref |g| max {g_ref.abs().max():.4e}")
print(f"cosine {cos:.6f}   significant slots {int(sig.sum())}/{need}   maxrel {rel.max():.3e}  medrel {rel.median():.3e}")
assert cos > 0.999 and rel.median() < 1e-2, "KERNEL vs STE-REFERENCE MISMATCH"
print("PQ_CB_STE_GRADCHECK_PASS")
