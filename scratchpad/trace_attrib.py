"""Attribute the QAT step's kernel-launch storm from the chrome trace (prof_qat_soft.json.gz).
For every cudaLaunchKernel runtime event, find the innermost enclosing CPU op (aten::*/autograd
node) on the same thread and count launches per op — then group tiny aten ops by their TOP-LEVEL
enclosing region (module forward names aren't recorded, but the python_function/user annotations
or the autograd parents give the region). Output: launches-per-op table + total, to decide where
torch.compile / hand-fusion / CUDA-graphs would pay.
"""
import gzip
import json
import sys
from bisect import bisect_right
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Andrew\rwkv-state-quant\scratchpad\prof_qat_soft.json.gz"
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
    data = json.load(fh)
events = data["traceEvents"] if isinstance(data, dict) else data

# CPU op intervals per thread (complete events, cat cpu_op), and cudaLaunchKernel instants
ops_by_tid = {}
launches_by_tid = {}
n_launch = 0
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    tid = e.get("tid")
    if cat == "cpu_op":
        ops_by_tid.setdefault(tid, []).append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))
    elif cat in ("cuda_runtime", "runtime") and e.get("name", "").startswith("cudaLaunchKernel"):
        launches_by_tid.setdefault(tid, []).append(e["ts"])
        n_launch += 1

print(f"total cudaLaunchKernel events: {n_launch}")

per_op = Counter()
for tid, launches in launches_by_tid.items():
    ops = sorted(ops_by_tid.get(tid, []))
    starts = [o[0] for o in ops]
    for ts in launches:
        i = bisect_right(starts, ts) - 1
        # walk left to the INNERMOST interval containing ts (interval list is sorted by start;
        # the innermost enclosing op is the one with the latest start whose end covers ts)
        name = "<no enclosing op>"
        j = i
        while j >= 0 and j > i - 200:
            s, t, n = ops[j]
            if s <= ts <= t:
                name = n
                break
            j -= 1
        per_op[name] += 1

print("\nlaunches per innermost enclosing CPU op (top 35):")
for name, c in per_op.most_common(35):
    print(f"  {c:8d}  {name[:100]}")
print(f"\n  {sum(per_op.values()):8d}  TOTAL attributed")
