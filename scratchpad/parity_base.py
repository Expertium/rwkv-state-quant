import torch
from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7, RWKV7_WKV_Stateful, reference_rwkv7_stateful
dev="cuda"; torch.manual_seed(0)
B,T,H,K = 3, 40, 2, 16
def nrm(x): return torch.nn.functional.normalize(x, dim=-1, p=2.0)
# mirror the model's tensor statistics: k,v L2-normalized; a,k_scale via sigmoid; w=exp(-exp(.)) near 1
r = 0.5*torch.randn(B,T,H,K,device=dev)
kd = nrm(torch.randn(B,T,H,K,device=dev))
a = torch.sigmoid(torch.randn(B,T,H,K,device=dev))
k = kd * a                                   # k_BTHK = k_deformed * a  (as in architecture.py)
v = nrm(torch.randn(B,T,H,K,device=dev)) * torch.sigmoid(torch.randn(B,T,H,K,device=dev))
_d = -0.5 - torch.nn.functional.softplus(-torch.randn(B,T,H,K,device=dev))
w = torch.exp(-torch.exp(_d.float()))         # decay near 1
skip = torch.zeros(B,T,device=dev,dtype=torch.bool)
inputs=[x.contiguous().requires_grad_(True) for x in (r,k,v,w,a,kd)]
out_k = RWKV7_WKV.apply(*inputs, skip)
out_r = reference_rwkv7(*[x.detach() for x in inputs], skip)
print("fwd max abs diff (kernel vs ref):", (out_k-out_r).abs().max().item())
# grads
g = torch.randn_like(out_k)
gk = torch.autograd.grad(out_k, inputs, g, retain_graph=True)
inputs2=[x.detach().clone().requires_grad_(True) for x in inputs]
out_r2 = reference_rwkv7(*inputs2, skip)
gr = torch.autograd.grad(out_r2, inputs2, g)
names=["r","k","v","w","a","kd"]
for n,a1,b1 in zip(names,gk,gr):
    print(f"  grad {n}: max abs diff {(a1-b1).abs().max().item():.3e}")
print("OK base parity")
