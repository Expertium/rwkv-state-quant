"""Define the 400+400 dev/val split for the H=2/K=16 phase.
- Read scratchpad/trace_sizes.txt (id size_bytes for all 836 reference_big users).
- Remove the 36 LARGEST (by trace .safetensors size = ~review count = slowest) -> 800 remain.
- Sort remaining by id: dev = lower 400 (prototyping), val = upper 400 (held-out eval).
- Write dev_users.txt / val_users.txt (project root) + excluded36.txt; MOVE the 36 to excluded36/.
"""
import shutil
from pathlib import Path

ROOT = Path(".")
BIG = ROOT / "reference_big"
EXC = ROOT / "excluded36"

rows = []
for line in open(ROOT / "scratchpad" / "trace_sizes.txt"):
    uid, sz = line.split()
    rows.append((int(uid), int(sz)))
assert len(rows) == 836, f"expected 836, got {len(rows)}"

by_size = sorted(rows, key=lambda r: r[1], reverse=True)
remove = [uid for uid, _ in by_size[:36]]
remove_set = set(remove)
keep = sorted(uid for uid, _ in rows if uid not in remove_set)
assert len(keep) == 800

dev = keep[:400]
val = keep[400:800]

# newline="\n" forces LF (Windows default would write \r\n, which corrupts ids read in bash).
(ROOT / "dev_users.txt").write_text("\n".join(str(u) for u in dev) + "\n", newline="\n")
(ROOT / "val_users.txt").write_text("\n".join(str(u) for u in val) + "\n", newline="\n")
(ROOT / "excluded36.txt").write_text("\n".join(str(u) for u in sorted(remove)) + "\n", newline="\n")

# move the 36 largest aside (reversible "delete")
EXC.mkdir(exist_ok=True)
moved = 0
for uid in remove:
    for ext in (".safetensors", ".json"):
        src = BIG / f"trace_user_{uid}{ext}"
        if src.exists():
            shutil.move(str(src), str(EXC / src.name))
            moved += 1

print("=== 36 largest removed (id  size_MB) ===")
for uid, sz in by_size[:36]:
    print(f"  {uid}  {sz/1e6:8.1f} MB")
print(f"\nmoved {moved} files to {EXC}/  (reversible; rm -rf excluded36 to reclaim disk)")
print(f"dev: {len(dev)} users, id {dev[0]}..{dev[-1]}")
print(f"val: {len(val)} users, id {val[0]}..{val[-1]}")
print(f"removed-from-dev-range: {sum(1 for u in remove if u < dev[-1])}, total removed {len(remove)}")
