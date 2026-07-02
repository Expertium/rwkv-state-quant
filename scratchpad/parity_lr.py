import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_train"))
sys.path.insert(0, os.path.dirname(__file__))
from rwkv.model.rwkv_ops import quant_aware_rwkv7, single_timestep
from lr_ref import deploy_rank1_batched
dev = "cuda"; torch.manual_seed(0)

def ste_deploy_rank1(s_BHKK, qmax):
    recon = deploy_rank1_batched(s_BHKK, qmax).to(s_BHKK.dtype).to(s_BHKK.device)
    return s_BHKK + (recon - s_BHKK).detach()

def ref_qat_deploy_rank1(r,k,v,w,a,kd,skip,qmax):
    # mirrors quant_aware_rwkv7 but truncation = DEPLOY rank-1 (power-iter) STE
    r,k,v,w,a,kd = [t.float() for t in (r,k,v,w,a,kd)]
    B,T,H,K = r.shape
    out = torch.empty(B,T,H,K,device=r.device)
    st = torch.zeros(B,H,K,K,device=r.device)
    sk = skip.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    for t in range(T):
        out[:,t], ns = single_timestep(r[:,t],k[:,t],v[:,t],w[:,t],a[:,t],kd[:,t], st)
        trunc = ste_deploy_rank1(ns, qmax)
        st = torch.where(sk[:,t], st, trunc)
    return out

def nrm(x): return torch.nn.functional.normalize(x, dim=-1, p=2.0)
B,T,H,K = 4, 24, 2, 16
r = 0.5*torch.randn(B,T,H,K,device=dev)
kd = nrm(torch.randn(B,T,H,K,device=dev))
a = torch.sigmoid(torch.randn(B,T,H,K,device=dev))
k = kd*a
v = nrm(torch.randn(B,T,H,K,device=dev))*torch.sigmoid(torch.randn(B,T,H,K,device=dev))
_d=-0.5-torch.nn.functional.softplus(-torch.randn(B,T,H,K,device=dev)); w=torch.exp(-torch.exp(_d.float()))
skip = torch.zeros(B,T,device=dev,dtype=torch.bool); skip[0,7]=True; skip[2,15]=True
base=(r,k,v,w,a,kd)
names=["r","k","v","w","a","kd"]

os.environ["RWKV_QAT_FUSED"]="1"
ins=[t.detach().clone().requires_grad_(True) for t in base]
of = quant_aware_rwkv7(*ins, skip, float("inf"), 1, 7.0)   # lowrank rank=1 int4 -> fused
gf = torch.autograd.grad(of, ins, torch.ones_like(of))

ins2=[t.detach().clone().requires_grad_(True) for t in base]
oref = ref_qat_deploy_rank1(*ins2, skip, 7.0)
gref = torch.autograd.grad(oref, ins2, torch.ones_like(oref))

od=(of-oref).abs()
den=oref.abs().amax().clamp_min(1e-6)
print(f"FWD fused vs deploy-rank1 ref: max {od.max():.3e} mean {od.mean():.3e}  (max/scale {od.max()/den:.2e})")
for n,af,ar in zip(names,gf,gref):
    gd=(af-ar).abs(); gden=ar.abs().amax().clamp_min(1e-6)
    print(f"  grad {n}: max {gd.max():.3e} mean {gd.mean():.3e}  (max/scale {gd.max()/gden:.2e})")
print("finite:", torch.isfinite(of).all().item(), all(torch.isfinite(g).all().item() for g in gf))
