import torch

a = torch.load(r"C:\Users\Andrew\rwkv-state-quant\scratchpad\warm_ab_nowarm.pt")
b = torch.load(r"C:\Users\Andrew\rwkv-state-quant\scratchpad\warm_ab_warm.pt")
for kname in ("out", "rg", "kg", "vg", "wg", "ag", "kdg"):
    same = torch.equal(a[kname], b[kname])
    print(f"{kname:4s} bitwise_equal={same}")
    assert same, kname
ca, cb2 = a["cbg"], b["cbg"]
rel = ((ca - cb2).abs().max() / ca.abs().max().clamp_min(1e-30)).item()
print(f"cbg  maxREL={rel:.3e} (atomicAdd order tolerance)")
assert rel < 1e-5
print("WARM A/B: PASS")
