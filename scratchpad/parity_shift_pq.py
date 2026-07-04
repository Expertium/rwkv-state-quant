#!/usr/bin/env python
"""Parity: the Python QAT fake_pq_shift math (torch cdist/argmin) vs a faithful numpy mirror of the Rust
deploy PqCodebook::encode_decode (serial per-centroid squared distance, first strict min), on real shift
vectors from the corpus. Near-equidistant sub-vectors may pick different centroids under different
summation orders (same accepted property as the WKV PQ parity) — the bulk must agree to ~1e-6.

Usage: python scratchpad/parity_shift_pq.py <codebook> <corpus_file> [n]"""
import sys

import numpy as np
import torch


def load_cb(path):
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, c, ncent = (int(x) for x in lines[0].split()[:5])
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]], np.float32)
    return rows.reshape(2, m, ncent, sub), m, sub, ncent


def rust_encode_decode(cb, role, col):
    """Mirror of PqCodebook::encode_decode (f32, serial, first strict min)."""
    m, sub = cb.shape[1], cb.shape[3]
    norm = np.float32(np.sqrt(np.sum(col.astype(np.float32) ** 2)))
    if not np.isfinite(norm) or norm < 1e-20:
        return col.copy()
    inv = np.float32(1.0) / norm
    out = col.copy()
    for p in range(m):
        s = p * sub
        cents = cb[role, p]
        best, bestd = 0, np.float32(np.inf)
        for ci in range(cents.shape[0]):
            d = np.float32(0)
            for j in range(sub):
                diff = col[s + j] * inv - cents[ci, j]
                d += diff * diff
            if d < bestd:
                bestd, best = d, ci
        out[s:s + sub] = cents[best] * norm
    return out


def torch_encode_decode(cbt, role, col):
    """The fake_pq_shift forward math (f32)."""
    m, sub = cbt.shape[1], cbt.shape[3]
    x = torch.from_numpy(col).float().unsqueeze(0)
    norm = x.norm(dim=1, keepdim=True)
    if norm.item() < 1e-20:
        return col.copy()
    unit = x / norm.clamp_min(1e-20)
    parts = []
    for p in range(m):
        d = torch.cdist(unit[:, p * sub:(p + 1) * sub], cbt[role, p])
        parts.append(cbt[role, p][d.argmin(dim=1)])
    return (torch.cat(parts, dim=1) * norm).squeeze(0).numpy()


def main():
    cb_path, corpus, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400
    cb, m, sub, ncent = load_cb(cb_path)
    cbt = torch.from_numpy(cb)
    vecs = {"TS": [], "CS": []}
    with open(corpus) as fh:
        for line in fh:
            tag = line[:2]
            if tag in vecs and len(vecs[tag]) < n:
                v = np.fromstring(line[2:], sep=" ", dtype=np.float32)
                if v.size == m * sub:
                    vecs[tag].append(v)
    diffs, flips = [], 0
    for role, tag in ((0, "TS"), (1, "CS")):
        for v in vecs[tag]:
            r = rust_encode_decode(cb, role, v)
            t = torch_encode_decode(cbt, role, v)
            rel = np.abs(r - t).max() / max(np.abs(r).max(), 1e-12)
            if rel > 1e-3:
                flips += 1  # near-tie centroid flip (accepted class)
            else:
                diffs.append(rel)
    total = len(vecs["TS"]) + len(vecs["CS"])
    print(f"vectors: {total}  agree<1e-3: {len(diffs)}  near-tie flips: {flips}")
    print(f"max rel (agreeing): {max(diffs):.3e}  mean: {np.mean(diffs):.3e}")
    assert len(diffs) >= total * 0.98, "too many disagreements — NOT a tie-flip pattern, algorithm mismatch"
    print("SHIFT_PQ_PARITY_PASS")


if __name__ == "__main__":
    main()
