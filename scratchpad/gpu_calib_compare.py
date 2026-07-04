#!/usr/bin/env python
"""Calibrate the GPU eval against the CPU (Rust deploy) eval, per user.

GPU side: result jsonls from rwkv.get_result (FILE_IMM = imm head, FILE_AHEAD = ahead head; 'LogLoss').
CPU side: recomputed from reference_big traces + preds/rust_pred_<tag>_<u>.json exactly like per_user.py.

Usage: python scratchpad/gpu_calib_compare.py <gpu_imm.jsonl> <gpu_ahead.jsonl> <cpu_tag> <users_file>"""
import json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

TRACE = Path("reference_big")
PRED = Path("preds")


def cpu_user_losses(tag, users):
    out = {}
    for u in users:
        meta_p = TRACE / f"trace_user_{u}.json"
        pred_p = PRED / f"rust_pred_{tag}_{u}.json"
        if not meta_p.exists() or not pred_p.exists():
            continue
        meta = json.load(open(meta_p))
        eq = meta["equalize_review_ths"]
        lr = {int(k): int(v) for k, v in meta["label_rating"].items()}
        lb = {rt: int(np.clip(lr[rt], 0, 1)) for rt in eq}
        d = json.load(open(pred_p))
        rth = d["review_th"]
        imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
        ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
        r = {}
        for mode, pm in (("imm", imm), ("ahead", ahead)):
            keys = [rt for rt in eq if rt in pm]
            if keys:
                r[mode] = log_loss([lb[rt] for rt in keys], [pm[rt] for rt in keys], labels=[0, 1])
        out[u] = r
    return out


def gpu_user_losses(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        out[d["user"]] = d["metrics"]["LogLoss"]
    return out


def main():
    gpu_imm_f, gpu_ahead_f, cpu_tag, users_file = sys.argv[1:5]
    users = [int(x) for x in Path(users_file).read_text().split()]
    gi, ga = gpu_user_losses(gpu_imm_f), gpu_user_losses(gpu_ahead_f)
    cpu = cpu_user_losses(cpu_tag, users)
    for mode, g in (("imm", gi), ("ahead", ga)):
        both = [u for u in users if u in g and u in cpu and mode in cpu[u]]
        gv = np.array([g[u] for u in both])
        cv = np.array([cpu[u][mode] for u in both])
        d = gv - cv
        print(f"{mode}: n={len(both)}  GPU mean {gv.mean():.6f}  CPU mean {cv.mean():.6f}  "
              f"mean diff {d.mean():+.6f}  mean|diff| {np.abs(d).mean():.6f}  max|diff| {np.abs(d).max():.6f}  "
              f"corr {np.corrcoef(gv, cv)[0,1]:.6f}")
        worst = sorted(zip(both, np.abs(d)), key=lambda kv: -kv[1])[:3]
        print("   worst |diff|: " + ", ".join(f"{u}({x:.5f})" for u, x in worst))


if __name__ == "__main__":
    main()
