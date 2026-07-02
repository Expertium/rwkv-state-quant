import os, time, torch
from rwkv.model.rwkv_ops import RWKV7_WKV, quant_aware_rwkv7
dev="cuda"; torch.manual_seed(0)
def nrm(x): return torch.nn.functional.normalize(x, dim=-1, p=2.0)
def mk(B,T,H,K):
    r=0.5*torch.randn(B,T,H,K,device=dev); kd=nrm(torch.randn(B,T,H,K,device=dev))
    a=torch.sigmoid(torch.randn(B,T,H,K,device=dev)); k=kd*a
    v=nrm(torch.randn(B,T,H,K,device=dev))*torch.sigmoid(torch.randn(B,T,H,K,device=dev))
    _d=-0.5-torch.nn.functional.softplus(-torch.randn(B,T,H,K,device=dev)); w=torch.exp(-torch.exp(_d.float()))
    skip=torch.zeros(B,T,device=dev,dtype=torch.bool)
    return [x.contiguous().requires_grad_(True) for x in (r,k,v,w,a,kd)], skip
def bench(fn,I,s,label,iters=6):
    for _ in range(2):
        o=fn(I,s); torch.autograd.grad(o,I,torch.ones_like(o))
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(iters):
        o=fn(I,s); torch.autograd.grad(o,I,torch.ones_like(o))
    torch.cuda.synchronize(); dt=(time.time()-t0)/iters
    print(f"{label:38s} {dt*1000:9.2f} ms"); return dt
for (B,T) in [(32,100),(128,200)]:
    I,s=mk(B,T,2,16); print(f"\n=== B={B} T={T} ===")
    base=bench(lambda I,s: RWKV7_WKV.apply(*I,s), I,s,"plain WKV kernel")
    os.environ["RWKV_QAT_FUSED"]="1"
    fus=bench(lambda I,s: quant_aware_rwkv7(*I,s,float('inf'),1,7.0), I,s,"rank-1 int4 QAT (FUSED)")
    print(f"  --> fused low-rank vs plain: {fus/base:.1f}x")
# python-loop ratio at tiny size
I,s=mk(8,40,2,16); print("\n=== python-loop ratio (B=8 T=40) ===")
os.environ["RWKV_QAT_FUSED"]="1"; fus=bench(lambda I,s: quant_aware_rwkv7(*I,s,float('inf'),1,7.0), I,s,"fused",iters=6)
os.environ["RWKV_QAT_FUSED"]="0"; py =bench(lambda I,s: quant_aware_rwkv7(*I,s,float('inf'),1,7.0), I,s,"python SVD loop",iters=2)
print(f"  --> FUSED speedup vs python low-rank loop: {py/fus:.0f}x")
