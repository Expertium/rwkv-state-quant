"""Log-loss scorer for the H=2/K=16 phase (reference_big, 400+400 dev/val).

Usage:  python score.py <users_file> <tag1> [tag2 ...]
  <users_file> = file with one user id per line (dev_users.txt / val_users.txt / scratchpad/active_users.txt)
  Inputs (traces) and outputs (preds) live in SEPARATE folders:
    RWKV_TRACE_DIR (default reference_big): {TRACE}/trace_user_{u}.json  -- labels (rating, equalize ths)
    RWKV_PRED_DIR  (default preds):         {PRED}/rust_pred_{tag}_{u}.json -- Rust preds per tag
Prints by-user-mean imm/ahead LogLoss per tag + the penalty (tag - fp32) when fp32 is among the tags.
A user is included only if its trace + ALL requested tag preds exist (others skipped, counted)."""
import json
import os
import sys
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
    users = [int(x) for x in Path(users_file).read_text().split()]

    agg = {t: {"imm": [], "ahead": []} for t in tags}
    used, missing = [], 0
    for u in users:
        meta_p = TRACE / f"trace_user_{u}.json"
        if not meta_p.exists() or not all((PRED / f"rust_pred_{t}_{u}.json").exists() for t in tags):
            missing += 1
            continue
        meta = json.load(open(meta_p))
        eq = meta["equalize_review_ths"]
        label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
        label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
        preds = {t: load_preds(PRED / f"rust_pred_{t}_{u}.json") for t in tags}
        imm_keys = [rt for rt in eq if all(rt in preds[t][0] for t in tags)]
        ah_keys = [rt for rt in eq if all(rt in preds[t][1] for t in tags)]
        if not imm_keys or not ah_keys:
            missing += 1
            continue
        for t in tags:
            agg[t]["imm"].append(ll(label_bin, preds[t][0], imm_keys))
            agg[t]["ahead"].append(ll(label_bin, preds[t][1], ah_keys))
        used.append(u)

    if not used:
        print(f"NO USERS scored from {users_file} (missing {missing}). Run the Rust passes first.")
        return
    m = {t: {k: float(np.mean(v)) for k, v in agg[t].items()} for t in tags}
    print(f"traces: {TRACE}  preds: {PRED}   users scored: {len(used)}  (skipped {missing} for missing trace/preds)")
    print(f"\n{'set':<16} {'imm':>10} {'ahead':>10}")
    for t in tags:
        print(f"{('rust_'+t):<16} {m[t]['imm']:>10.6f} {m[t]['ahead']:>10.6f}")
    if "fp32" in tags:
        print("\nQUANT PENALTY (rust_<tag> - rust_fp32):  [WIN = <= +0.0025 in BOTH imm AND ahead at <=256 bits]")
        for t in tags:
            if t == "fp32":
                continue
            print(f"  {t:<10}: imm {m[t]['imm']-m['fp32']['imm']:+.6f}  ahead {m[t]['ahead']-m['fp32']['ahead']:+.6f}")


if __name__ == "__main__":
    main()
