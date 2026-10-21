"""Re-run Part A of r3_analyses.py locally, byte-for-byte the same logic.

Source script: <internal host>  <EDSURV_ROOT>
Input:         ensemble_strict3h_probs.npz (per-visit ensemble predictions; not redistributable)
"""
import io, json, sys
import numpy as np
from lifelines.utils import concordance_index

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROBS = r"pull/ensemble_strict3h_probs.npz"


def boot_ci(fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    N = fn.N
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


d = np.load(PROBS)
print("npz keys:", list(d.keys()))
probs, event, t = d["probs"], d["event"].astype(int), d["t_hours"].astype(float)
print("probs shape:", probs.shape, " events:", int(event.sum()), " N:", len(t))

risk = probs[:, :-1].sum(1)


def ci_on(mask, risk_vec, tt, ev):
    r, e, tm = risk_vec[mask], ev[mask], tt[mask]

    def f(idx):
        return concordance_index(tm[idx], -r[idx], e[idx])
    f.N = int(mask.sum())
    pt = concordance_index(tm, -r, e)
    m, lo, hi = boot_ci(f)
    return pt, lo, hi, int(mask.sum()), int(e.sum())


out = {}

full = ci_on(np.ones(len(t), bool), risk, t, event)
print(f"[A] Full test: C={full[0]:.4f} [{full[1]:.4f},{full[2]:.4f}] N={full[3]} events={full[4]}")
out["full"] = full

within3 = (event == 1) & (t <= 3)
keep = ~within3
excl = ci_on(keep, risk, t, event)
print(f"[A] Excl within-3h events: C={excl[0]:.4f} [{excl[1]:.4f},{excl[2]:.4f}] "
      f"N={excl[3]} events={excl[4]} (removed {int(within3.sum())} within-3h events)")
out["excl_within3h"] = excl
out["n_within3h_removed"] = int(within3.sum())

elig = t > 4
risk_lm = probs[:, 2:5].sum(1)
lm = ci_on(elig, risk_lm, t, event)
print(f"[A] Landmark@4h (at-risk>4h, predict 4-48h): C={lm[0]:.4f} [{lm[1]:.4f},{lm[2]:.4f}] "
      f"N={lm[3]} events={lm[4]}")
out["landmark_4h"] = lm

with open("r3_partA_results.json", "w", encoding="utf-8") as fh:
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}, fh, indent=2)
print("\nsaved r3_partA_results.json")
print("R3_PARTA_DONE")
