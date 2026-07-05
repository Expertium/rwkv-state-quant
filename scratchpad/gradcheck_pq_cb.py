"""Finite-difference gradcheck for the LEARNABLE WKV codebook (kernel centroid-grad accumulation).
Runs the fused rank-1 PQ QAT forward/backward ops directly on small random inputs with the full q80-era
config (m2b3 codebook + int4 norm quant + learn flag), compares the fetched analytic centroid grads
against central differences. PQ selection is DISCRETE: a perturbed centroid can flip an assignment, so a
minority of slots may disagree (jump in the true function) — the bulk must match.
Usage: python scratchpad/gradcheck_pq_cb.py [codebook]"""
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
cb0 = torch.tensor(vals[:need], dtype=torch.float32, device="cuda")

def upload(cb):
    torch.ops.rwkv.rwkv7_set_pq_codebook(cb, m, sub, ncent)

upload(cb0)
torch.ops.rwkv.rwkv7_set_norm_quant(cb0, 4, -3.0, 0.0)
torch.ops.rwkv.rwkv7_set_pq_learn(cb0, 1)
print(f"cb {cbpath}: m={m} sub={sub} ncent={ncent}; norm quant int4; learn ON")

torch.manual_seed(7)
B, T, H, K = 2, 64, 2, 16
dev = "cuda"
r = (0.5 * torch.randn(B, T, H, K, device=dev)).float()
kk = (0.5 * torch.randn(B, T, H, K, device=dev)).float()
v = (0.5 * torch.randn(B, T, H, K, device=dev)).float()
w = (0.95 + 0.049 * torch.rand(B, T, H, K, device=dev)).float()
a = (0.3 * torch.rand(B, T, H, K, device=dev)).float()
kd = (0.5 * torch.randn(B, T, H, K, device=dev)).float()
skip = torch.zeros(B, T, dtype=torch.bool, device=dev)
skip[0, 10] = True; skip[1, 40] = True
R = torch.randn(B, T, H, K, device=dev).float()
QMAX = 7.0

def fwd_loss():
    out, ckpt = torch.ops.rwkv.rwkv7_wkv_qat_lr_forward_float.default(r, kk, v, w, a, kd, skip, QMAX)
    return (out.double() * R.double()).sum().item(), ckpt

loss0, ckpt = fwd_loss()
torch.ops.rwkv.rwkv7_pq_cb_grad_zero(cb0)
torch.ops.rwkv.rwkv7_wkv_qat_lr_backward_float.default(r, kk, v, w, a, kd, skip, ckpt, R, QMAX)
g = torch.ops.rwkv.rwkv7_pq_cb_grad_get(cb0).cpu()
torch.cuda.synchronize()
nz = int((g != 0).sum().item())
print(f"analytic cb grad: {nz}/{need} nonzero, |g| max {g.abs().max():.3e} mean {g.abs().mean():.3e}")
assert nz > need // 4, "grad buffer mostly zero - accumulation not firing"

gen = torch.Generator().manual_seed(3)
slots = torch.multinomial((g.abs() > 1e-8).float() + 1e-6, 10, replacement=False, generator=gen).tolist()
EPS = 2e-3
ok, results = 0, []
for s in slots:
    cp = cb0.clone(); cp[s] += EPS; upload(cp); lp, _ = fwd_loss()
    cm = cb0.clone(); cm[s] -= EPS; upload(cm); lm, _ = fwd_loss()
    upload(cb0)
    num = (lp - lm) / (2 * EPS)
    ana = float(g[s])
    rel = abs(num - ana) / max(abs(num), abs(ana), 1e-9)
    results.append((s, ana, num, rel))
    if rel < 0.10:
        ok += 1
for s, ana, num, rel in results:
    print(f"  slot {s:5d}: analytic {ana:+.5e}  numeric {num:+.5e}  rel {rel:.3f}")
print(f"{ok}/10 slots within 10% (selection flips explain outliers)")
assert ok >= 7, "GRADCHECK FAIL - too many disagreements for flip noise"
print("PQ_CB_GRADCHECK_PASS")
