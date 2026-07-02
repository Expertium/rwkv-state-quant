import torch, time
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
dev = "cuda"
# a small fixed-shape body like the QAT per-step (elementwise + tiny matmul on (B,H,K,K))
def body(state, k, v):
    s = state + torch.einsum('bhk,bhl->bhkl', k, v)          # rank-1 update
    amax = s.abs().amax(dim=[1,2,3], keepdim=True).clamp_min(1e-12)
    q = torch.round(s / (amax/1.0)).clamp(-1,1) * (amax/1.0)  # fake int2
    return s + (q - s).detach()
B,H,K = 4,2,16
state = torch.randn(B,H,K,K,device=dev); k=torch.randn(B,H,K,device=dev); v=torch.randn(B,H,K,device=dev)
for mode in [None, "reduce-overhead"]:
    try:
        f = torch.compile(body, mode=mode) if mode else torch.compile(body)
        y = f(state,k,v); torch.cuda.synchronize()          # triggers compile
        y = f(state,k,v); torch.cuda.synchronize()
        print(f"OK mode={mode}: compiled+ran, out sum={float(y.sum()):.3f}")
    except Exception as e:
        print(f"FAIL mode={mode}: {type(e).__name__}: {str(e)[:300]}")
