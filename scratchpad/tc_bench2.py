import time, torch
from rwkv.model.rwkv_ops import single_timestep, fake_quant_state
dev="cuda"; B,H,K,T = 4,2,16,2000
torch.manual_seed(0)
# per-step-CHANGING inputs (realistic: sliced from a sequence, like the real QAT loop)
seq = [torch.randn(T,B,H,K,device=dev) for _ in range(6)]
state0 = torch.zeros(B,H,K,K,device=dev)
def body(r,k,v,w,a,kd,state):
    out, ns = single_timestep(r,k,v,w,a,kd,state)
    return out, fake_quant_state(ns, 1.0)      # full-matrix int2, as in the QAT run

def loop_plain(fn, clone=False):
    state = state0
    for t in range(T):
        if clone: torch.compiler.cudagraph_mark_step_begin()
        out, state = fn(seq[0][t],seq[1][t],seq[2][t],seq[3][t],seq[4][t],seq[5][t], state)
        if clone: state = state.clone()          # break CUDA-graph output aliasing on the fed-back state
    return state

def timeit(fn, label, clone=False, iters=5):
    loop_plain(fn, clone); torch.cuda.synchronize()      # warmup / compile / graph-capture
    t0=time.time()
    for _ in range(iters): loop_plain(fn, clone)
    torch.cuda.synchronize()
    dt=(time.time()-t0)/iters
    print(f"{label:30s} {dt*1000:8.1f} ms/loop  ({T/dt:,.0f} steps/s)")
    return dt

e = timeit(body, "eager")
c = timeit(torch.compile(body), "compile default")
try:
    r = timeit(torch.compile(body, mode='reduce-overhead'), "compile reduce-overhead", clone=True)
except Exception as ex:
    import traceback; traceback.print_exc(); print("reduce-overhead FAILED:", str(ex)[:200]); r=None
print(f"\nspeedup: default={e/c:.2f}x" + (f"  reduce-overhead={e/r:.2f}x" if r else "  reduce-overhead=FAILED"))
