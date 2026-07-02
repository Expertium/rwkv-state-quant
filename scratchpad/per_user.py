"""Per-user penalty breakdown vs fp32 for the H=2/K=16 phase — robustness + size-dependence check.

Usage:  python scratchpad/per_user.py <users_file> <tag1> [tag2 ...]   (tag1 should be fp32 = the base)
Reads traces from RWKV_TRACE_DIR (default reference_big), preds from RWKV_PRED_DIR (default preds).
Prints, per tag: overall mean penalty, worst users, # users over the +0.0025 gate, and mean penalty
bucketed by user trace-file size (a proxy for review-history length) to expose whether the penalty is
concentrated in the big/power users.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss

TRACE = Path(os.environ.get("RWKV_TRACE_DIR", "reference_big"))
PRED = Path(os.environ.get("RWKV_PRED_DIR", "preds"))


def load_preds(path):
    d = json.load(open(path))
    rth = d["review_th"]
    imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
    ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
    return imm, ahead


def ll(label_bin, predmap, keys):
    return log_loss([label_bin[rt] for rt in keys], [predmap[rt] for rt in keys], labels=[0, 1])


def main():
    users_file = sys.argv[1]
    tags = sys.argv[2:]
    base = tags[0]
    users = [int(x) for x in Path(users_file).read_text().split()]

    pen = {t: {"imm": {}, "ahead": {}} for t in tags[1:]}
    size = {}
    for u in users:
        meta_p = TRACE / f"trace_user_{u}.json"
        if not meta_p.exists() or not all((PRED / f"rust_pred_{t}_{u}.json").exists() for t in tags):
            continue
        meta = json.load(open(meta_p))
        eq = meta["equalize_review_ths"]
        label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
        label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
        preds = {t: load_preds(PRED / f"rust_pred_{t}_{u}.json") for t in tags}
        for mode, idx in (("imm", 0), ("ahead", 1)):
            keys = [rt for rt in eq if all(rt in preds[t][idx] for t in tags)]
            if not keys:
                continue
            b = ll(label_bin, preds[base][idx], keys)
            for t in tags[1:]:
                pen[t][mode][u] = ll(label_bin, preds[t][idx], keys) - b
        stf = TRACE / f"trace_user_{u}.safetensors"
        size[u] = stf.stat().st_size if stf.exists() else 0

    scored = [u for u in users if u in size and u in pen[tags[1]]["imm"]]
    print(f"scored {len(scored)} users from {users_file}   base={base}   tags={tags[1:]}")
    for t in tags[1:]:
        for mode in ("imm", "ahead"):
            vals = {u: pen[t][mode][u] for u in scored if u in pen[t][mode]}
            arr = np.array(list(vals.values()))
            worst = sorted(vals.items(), key=lambda kv: -kv[1])[:5]
            nbad = int((arr > 0.0025).sum())
            print(f"\n### {t}  {mode}:  mean {arr.mean():+.4f}   median {np.median(arr):+.4f}   "
                  f"nbad(>+0.0025) {nbad}/{len(arr)}")
            print("   worst5: " + ", ".join(f"{u}({size[u]//(1024*1024)}MB {d:+.4f})" for u, d in worst))
            # bucket by size quartile
            by_size = sorted(scored, key=lambda u: size[u])
            q = len(by_size) // 4
            for qi, lab in enumerate(["Q1 smallest", "Q2", "Q3", "Q4 largest"]):
                grp = by_size[qi*q:(qi+1)*q] if qi < 3 else by_size[3*q:]
                gv = np.array([pen[t][mode][u] for u in grp if u in pen[t][mode]])
                mb = np.mean([size[u] for u in grp]) / (1024*1024)
                print(f"     {lab:<12} (~{mb:4.0f}MB avg): mean {gv.mean():+.4f}   nbad {int((gv>0.0025).sum())}/{len(gv)}")


if __name__ == "__main__":
    main()
