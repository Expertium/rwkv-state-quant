"""task24 warm-start joint-search A/B: run the fused LR QAT op (fwd+bwd, joint cb, learnable path)
on a fixed random trajectory and dump outputs/grads. Launched TWICE in separate processes (env
RWKV_QAT_NO_WARM=1 vs unset -- the flag is read by the C++ host at codebook upload, so it must be
set before process start on Windows CRT). Picks identical => out + input grads BITWISE identical
(the cb grad uses atomicAdd, order-nondeterministic ~1e-7, compared with tolerance instead).
Usage: python warm_ab_test.py <tag>   -> scratchpad/warm_ab_<tag>.pt
"""
import os
import sys

ROOT = r"C:\Users\Andrew\rwkv-state-quant"
GT = os.path.join(ROOT, "gpu_train")
sys.path.insert(0, GT)
os.environ.setdefault("RWKV_NO_JIT", "1")

import torch  # noqa: E402

tag = sys.argv[1]
out_file = os.path.join(ROOT, "scratchpad", f"warm_ab_{tag}.pt")

from rwkv.model.rwkv_ops import RWKV7_WKV_QAT_LR  # noqa: E402  (loads the extension)

dev = "cuda"
path = os.path.join(ROOT, "scratchpad", "pq_cb_juv_b10.txt")
lines = [ln for ln in open(path) if ln.strip()]
m, bits, sub, k, ncent = (int(x) for x in lines[0].split()[:5])
vals = [float(x) for x in " ".join(lines[1:]).split()]
cb = torch.tensor(vals[: ncent * sub], dtype=torch.float32, device=dev)
torch.ops.rwkv.rwkv7_set_pq_codebook(cb, m, sub, ncent, 1)
torch.ops.rwkv.rwkv7_set_norm_quant(cb, 1, -3.0, 0.0)   # 1-bit norms, champion recipe
torch.ops.rwkv.rwkv7_set_pq_learn(cb, 1)                # exercise rec_idx/rec_norm recording too

torch.manual_seed(0)
B, T, H, K = 16, 1024, 2, 16
r = (torch.randn(B, T, H, K, device=dev) * 0.5).requires_grad_(True)
kk = (torch.randn(B, T, H, K, device=dev) * 0.5).requires_grad_(True)
v = (torch.randn(B, T, H, K, device=dev) * 0.5).requires_grad_(True)
w = (torch.rand(B, T, H, K, device=dev) * 0.10 + 0.85).requires_grad_(True)
a = (torch.rand(B, T, H, K, device=dev) * 0.5).requires_grad_(True)
kd = (torch.randn(B, T, H, K, device=dev) * 0.5).requires_grad_(True)
skip = torch.rand(B, T, device=dev) < 0.5
qmax = 7.0

torch.ops.rwkv.rwkv7_pq_cb_grad_zero(cb)
out = RWKV7_WKV_QAT_LR.apply(r, kk, v, w, a, kd, skip, qmax)
g = torch.empty_like(out)
torch.manual_seed(1)
g.normal_()
out.backward(g)
cbg = torch.ops.rwkv.rwkv7_pq_cb_grad_get(cb)
torch.save(
    dict(out=out.detach().cpu(), rg=r.grad.cpu(), kg=kk.grad.cpu(), vg=v.grad.cpu(),
         wg=w.grad.cpu(), ag=a.grad.cpu(), kdg=kd.grad.cpu(), cbg=cbg.cpu()),
    out_file,
)
print(f"[warm_ab] saved {out_file}  (nonskip searches ~ {(~skip).sum().item() * H})")
