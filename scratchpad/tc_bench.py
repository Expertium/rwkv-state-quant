import os, time, torch
from rwkv.model.rwkv_ops import single_timestep, fake_quant_state
dev="cuda"; B,H,K,T = 4,2,16,2000
def mk(): return [torch.randn(B,H,K,device=dev) for _ in range(6)] + [torch.zeros(B,H,K,K,device=dev)]
def body(r,k,v,w,a,kd,state):
    out,ns = single_timestep(r,k,v,w,a,kd,state)
    ns = fake_quant_state(ns, 1.0)   # full-matrix int2, as in the running QAT
    return out, ns
def run_loop(bodyfn, ins):
    r,k,v,w,a,kd,state = ins
    for _ in range(T):
        out, state = bodyfn(r,k,v,w,a,kd,state)
    return state
def timeit(bodyfn, label, iters=3):
    ins = mk()
    run_loop(bodyfn, ins); torch.cuda.synchronize()   # warmup / compile
    t0=time.time()
    for _ in range(iters): run_loop(bodyfn, ins)
    torch.cuda.synchronize()
    dt=(time.time()-t0)/iters
    print(f"{label:28s} {dt*1000:8.1f} ms / {T}-step loop  ({T/dt:,.0f} steps/s)")
    return dt
e = timeit(body, "eager")
c = timeit(torch.compile(body), "torch.compile (default)")
try:
    r = timeit(torch.compile(body, mode='reduce-overhead'), "torch.compile (reduce-ovh)")
except Exception as ex:
    print("reduce-overhead failed:", type(ex).__name__, str(ex)[:200]); r=None
print(f"\nspeedup default={e/c:.2f}x" + (f"  reduce-overhead={e/r:.2f}x" if r else ""))
