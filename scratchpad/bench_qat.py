import os, time, torch
from rwkv.model.rwkv_ops import RWKV7_WKV, quant_aware_rwkv7
dev="cuda"; torch.manual_seed(0)
def nrm(x): return torch.nn.functional.normalize(x, dim=-1, p=2.0)
def mkinputs(B,T,H,K):
    r = 0.5*torch.randn(B,T,H,K,device=dev)
    kd = nrm(torch.randn(B,T,H,K,device=dev))
    a = torch.sigmoid(torch.randn(B,T,H,K,device=dev))
    k = kd*a
    v = nrm(torch.randn(B,T,H,K,device=dev))*torch.sigmoid(torch.randn(B,T,H,K,device=dev))
    _d=-0.5-torch.nn.functional.softplus(-torch.randn(B,T,H,K,device=dev))
    w = torch.exp(-torch.exp(_d.float()))
    skip = torch.zeros(B,T,device=dev,dtype=torch.bool)
    return [x.contiguous().requires_grad_(True) for x in (r,k,v,w,a,kd)], skip

def bench(fn, inputs, skip, label, iters=10):
    # warmup
    for _ in range(2):
        out=fn(inputs,skip); g=torch.randn_like(out); torch.autograd.grad(out,inputs,g)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(iters):
        out=fn(inputs,skip); g=torch.randn_like(out); torch.autograd.grad(out,inputs,g)
    torch.cuda.synchronize(); dt=(time.time()-t0)/iters
    print(f"{label:34s} {dt*1000:9.2f} ms/fwd+bwd")
    return dt

for (B,T) in [(32,100),(128,200),(256,300)]:
    inputs,skip = mkinputs(B,T,2,16)
    print(f"\n=== B={B} T={T} H=2 K=16 ===")
    base = bench(lambda I,s: RWKV7_WKV.apply(*I,s), inputs, skip, "plain kernel (no QAT)")
    os.environ["RWKV_QAT_FUSED"]="0"
    qat  = bench(lambda I,s: quant_aware_rwkv7(*I,s,1.0,0,float('inf')), inputs, skip, "quant_aware int2 (python loop)")
    os.environ["RWKV_QAT_FUSED"]="1"
    fus  = bench(lambda I,s: quant_aware_rwkv7(*I,s,1.0,0,float('inf')), inputs, skip, "quant_aware int2 (FUSED kernel)")
    print(f"  --> python-loop slowdown vs plain: {qat/base:.1f}x |  FUSED slowdown vs plain: {fus/base:.1f}x |  FUSED speedup vs python-loop: {qat/fus:.1f}x")
