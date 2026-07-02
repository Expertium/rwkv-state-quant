import os, torch
from rwkv.model.rwkv_ops import quant_aware_rwkv7
dev="cuda"; torch.manual_seed(0)
def nrm(x): return torch.nn.functional.normalize(x, dim=-1, p=2.0)
def mkinputs(B,T,H,K, seed=0, nskip=0):
    g=torch.Generator(device=dev).manual_seed(seed)
    def rn(*s): return torch.randn(*s, device=dev, generator=g)
    r = 0.5*rn(B,T,H,K)
    kd = nrm(rn(B,T,H,K))
    a = torch.sigmoid(rn(B,T,H,K))
    k = kd*a
    v = nrm(rn(B,T,H,K))*torch.sigmoid(rn(B,T,H,K))
    _d=-0.5-torch.nn.functional.softplus(-rn(B,T,H,K))
    w = torch.exp(-torch.exp(_d.float()))
    skip = torch.zeros(B,T,device=dev,dtype=torch.bool)
    if nskip>0:
        idx = torch.randperm(B*T, generator=g, device=dev)[:nskip]
        skip.view(-1)[idx]=True
    return (r,k,v,w,a,kd), skip

def run(path_fused, tensors, skip, qmax):
    os.environ["RWKV_QAT_FUSED"] = "1" if path_fused else "0"
    ins=[t.detach().clone().requires_grad_(True) for t in tensors]
    out = quant_aware_rwkv7(*ins, skip, qmax, 0, float("inf"))
    g = torch.ones_like(out)  # deterministic upstream grad
    grads = torch.autograd.grad(out, ins, g)
    return out.detach(), [gr.detach() for gr in grads]

names=["r","k","v","w","a","kd"]
def compare(tag, o1,g1,o2,g2):
    od=(o1-o2).abs()
    frac = (od > 1e-3).float().mean().item()
    print(f"  [{tag}] out: max {od.max().item():.3e}  mean {od.mean().item():.3e}  frac>1e-3 {frac*100:.3f}%")
    for n,a,b in zip(names,g1,g2):
        gd=(a-b).abs()
        print(f"       grad {n}: max {gd.max().item():.3e}  mean {gd.mean().item():.3e}")

QMAX={"int8":127.0,"int4":7.0,"int2":1.0}

print("=== TEST 1: T=1 single step (no compounding -> tight even at int2) ===")
tensors,skip = mkinputs(4,1,2,16, seed=1)
for lvl,q in QMAX.items():
    of,gf = run(True, tensors, skip, q)
    op,gp = run(False, tensors, skip, q)
    compare(f"T=1 {lvl}", of,gf, op,gp)

print("\n=== TEST 2: multi-step int8 (tight; validates recurrence+backward+skip) ===")
tensors,skip = mkinputs(8,60,2,16, seed=2, nskip=20)
of,gf = run(True, tensors, skip, 127.0)
op,gp = run(False, tensors, skip, 127.0)
compare("T=60 int8", of,gf, op,gp)

print("\n=== TEST 3: multi-step int4/int2 (expect rare boundary flips) ===")
for lvl,q in [("int4",7.0),("int2",1.0)]:
    of,gf = run(True, tensors, skip, q)
    op,gp = run(False, tensors, skip, q)
    compare(f"T=60 {lvl}", of,gf, op,gp)

print("\n=== TEST 4: NaN safeguard (inject inf at t=5 -> that step's out is inf in BOTH,")
print("            but the CARRIED state must be capped so t>=6 recovers & matches) ===")
tensors,skip = mkinputs(4,30,2,16, seed=3)
tl=list(tensors); tl[2]=tl[2].clone(); tl[2][0,5,0,:]= float('inf')  # blow up v at one step, batch 0
tensors=tuple(tl)
of,gf = run(True, tensors, skip, 1.0)
op,gp = run(False, tensors, skip, 1.0)
post_f = of[:,6:]; post_p = op[:,6:]
print("  post-injection (t>=6) finite:  fused", torch.isfinite(post_f).all().item(),
      " python", torch.isfinite(post_p).all().item())
pd=(post_f-post_p).abs()
print(f"  post-injection out parity: max {pd.max().item():.3e}  mean {pd.mean().item():.3e}")
# only the affected step (batch0,t=5) should be non-finite; everything else finite in both
badf = (~torch.isfinite(of)).sum().item(); badp=(~torch.isfinite(op)).sum().item()
print(f"  non-finite out entries: fused {badf}  python {badp}  (should be equal & confined to t=5)")
print("\nDONE")
