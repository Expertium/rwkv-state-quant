"""Who calls the fill_/mul/add_ storm? For each target aten op event, find its PARENT CPU op
(innermost strictly-enclosing interval) and count parents. Run on prof_qat_soft.json.gz."""
import gzip
import json
import sys
from bisect import bisect_right
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Andrew\rwkv-state-quant\scratchpad\prof_qat_soft.json.gz"
targets = set((sys.argv[2] if len(sys.argv) > 2 else "aten::fill_,aten::mul,aten::add_,aten::mm,aten::copy_,aten::arange").split(","))
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
    data = json.load(fh)
events = data["traceEvents"] if isinstance(data, dict) else data

ops_by_tid = defaultdict(list)
for e in events:
    if e.get("ph") == "X" and e.get("cat") == "cpu_op":
        ops_by_tid[e.get("tid")].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))

parents = {t: Counter() for t in targets}
for tid, ops in ops_by_tid.items():
    ops.sort()
    starts = [o[0] for o in ops]
    for k, (s, t, name) in enumerate(ops):
        if name not in targets:
            continue
        # parent = latest op starting at/before s (excluding self) that still covers t
        j = bisect_right(starts, s) - 1
        parent = "<top-level>"
        cand = j
        while cand >= 0 and cand > j - 400:
            ps, pt, pn = ops[cand]
            if (ps, pt) != (s, t) and ps <= s and pt >= t:
                parent = pn
                break
            cand -= 1
        parents[name][parent] += 1

for tname, ctr in parents.items():
    total = sum(ctr.values())
    print(f"\n== {tname}  ({total} events) — parents:")
    for pn, c in ctr.most_common(12):
        print(f"   {c:8d}  {pn[:95]}")
