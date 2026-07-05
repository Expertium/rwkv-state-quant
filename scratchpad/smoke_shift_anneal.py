"""Smoke test: soft-to-hard selection annealing (RWKV_QAT_SHIFT_ANNEAL) in fake_pq_shift.
Checks: (1) tau=0 is the untouched hard path; (2) soft output converges to the hard output as
tau -> 0+; (3) gradients flow to x, the codebook Parameter AND the rotation P in the soft phase;
(4) a no_grad call ignores tau (validation passes see the deploy-exact hard quantizer);
(5) the linear schedule hits exactly 0 at the END fraction and stays there.
Usage: python scratchpad/smoke_shift_anneal.py [shift_cb]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
os.environ["RWKV_QAT_NORM_BITS"] = "1"
os.environ["RWKV_QAT_SHIFT_ROT"] = "1"
os.environ["RWKV_QAT_SHIFT_PQ_LEARN"] = "1"
os.environ["RWKV_QAT_SHIFT_ANNEAL"] = "0.05"
import torch
from rwkv.model import rwkv_model as RM

cb_path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/pq_cb_shift_m4b5.txt"
with open(cb_path) as fh:
    C = int([ln for ln in fh if ln.strip()][0].split()[3])
RM._SHIFT_PQ_PATH = cb_path
RM._SHIFT_PQ_CB = None
RM.shift_rot_init("cpu", C)
with torch.no_grad():
    RM._SHIFT_ROT_P.copy_(0.1 * torch.randn(2, C, C, generator=torch.Generator().manual_seed(11)))

g = torch.Generator().manual_seed(7)
x = (torch.randn(2, 5, C, generator=g) * 1.05).requires_grad_(True)  # norms ~ the shift range

fails = 0
def check(name, cond, detail=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    fails += 0 if cond else 1

# (1)+(2): tau=0 vs tau=1e-6 vs tau=0.05
RM._SHIFT_ANNEAL_TAU = 0.0
q_hard = RM.fake_pq_shift(x, 0).detach()
RM._SHIFT_ANNEAL_TAU = 1e-6
q_tiny = RM.fake_pq_shift(x, 0).detach()
RM._SHIFT_ANNEAL_TAU = 0.05
q_soft = RM.fake_pq_shift(x, 0)
d_tiny = (q_tiny - q_hard).abs().max().item()
d_soft = (q_soft.detach() - q_hard).abs().max().item()
check("tau=1e-6 == hard", d_tiny < 1e-5, f"max|diff|={d_tiny:.2e}")
check("tau=0.05 differs (soft is live)", d_soft > 1e-4, f"max|diff|={d_soft:.2e}")
check("soft output finite", bool(torch.isfinite(q_soft).all()))

# (3): gradient flow in the soft phase
for p in (RM._SHIFT_PQ_CB, RM._SHIFT_ROT_P):
    p.grad = None
x.grad = None
q_soft.square().sum().backward()
check("grad -> x", x.grad is not None and float(x.grad.abs().sum()) > 0)
check("grad -> codebook", RM._SHIFT_PQ_CB.grad is not None and float(RM._SHIFT_PQ_CB.grad.abs().sum()) > 0)
check("grad -> rotation P", RM._SHIFT_ROT_P.grad is not None and float(RM._SHIFT_ROT_P.grad.abs().sum()) > 0)

# (4): no_grad ignores tau (hard path)
with torch.no_grad():
    q_ng = RM.fake_pq_shift(x, 0)
d_ng = (q_ng - q_hard).abs().max().item()
check("no_grad call == hard", d_ng == 0.0, f"max|diff|={d_ng:.2e}")

# (5): schedule — linear to 0 at END=0.5, 0 after
taus = [RM.set_shift_anneal_progress(f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
check("schedule", abs(taus[0] - 0.05) < 1e-12 and abs(taus[1] - 0.025) < 1e-12
      and taus[2] == 0.0 and taus[3] == 0.0 and taus[4] == 0.0,
      f"taus={['%.4f' % t for t in taus]}")

print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
