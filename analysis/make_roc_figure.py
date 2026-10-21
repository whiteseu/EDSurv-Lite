"""Regenerate strict-3h Figure 2 (ROC + calibration) on the reproducible
MDS-ED intersection with 1,000-replicate bootstrap.

Identical plotting code to the original regen_strict3h_figs.py; the only
changes are n_boot 200 -> 1000 and local paths.
"""
from pathlib import Path
import io, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROBS = Path("pull/ensemble_strict3h_probs.npz")
MDSED = Path("pull/mds_ed_slim_labels.csv")
FIGDIR = Path("cr/figures")
N_BOOT = 1000

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "pdf.fonttype": 42,
})

COL_MDSED = "#0072B2"
COL_STRICT = "#D55E00"
COL_CHANCE = "#888888"
TIME_BINS_HOURS = [1, 4, 12, 24, 48]


def cum_to_horizon(probs, horizon=24, edges=TIME_BINS_HOURS):
    cum = np.zeros(len(probs))
    for k, e in enumerate(edges):
        if e <= horizon + 1e-9:
            cum += probs[:, k]
    return cum


def bootstrap_roc(y, p, n_boot=N_BOOT, n_grid=101, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, n_grid)
    tprs = np.zeros((n_boot, n_grid))
    aucs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            tprs[b] = grid
            aucs[b] = 0.5
            continue
        fpr, tpr, _ = roc_curve(yb, pb)
        tprs[b] = np.interp(grid, fpr, tpr)
        aucs[b] = roc_auc_score(yb, pb)
    return grid, tprs, aucs


def plot_roc(y, p, out_path):
    grid, tprs, aucs = bootstrap_roc(y, p)
    mean_tpr = tprs.mean(axis=0)
    lo = np.percentile(tprs, 2.5, axis=0)
    hi = np.percentile(tprs, 97.5, axis=0)
    auc_pt = roc_auc_score(y, p)
    auc_lo, auc_hi = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=COL_CHANCE, lw=1.0, linestyle=(0, (3, 3)), label="chance")
    ax.fill_between(grid, lo, hi, color=COL_MDSED, alpha=0.18, lw=0)
    ax.plot(grid, mean_tpr, color=COL_MDSED, lw=2.2,
            label=f"EDSurv-Lite  AUROC {auc_pt:.3f}\n[{auc_lo:.3f},{auc_hi:.3f}]")
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  AUROC {auc_pt:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    return auc_pt, auc_lo, auc_hi


def plot_calibration(y, p, out_path, n_bins=10):
    bin_edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_edges[0] -= 1e-6; bin_edges[-1] += 1e-6
    bins = np.clip(np.digitize(p, bin_edges) - 1, 0, len(bin_edges) - 2)
    pred = []; obs = []; lo = []; hi = []
    for b in range(len(bin_edges) - 1):
        m = bins == b
        if m.sum() < 5:
            continue
        k = y[m].sum(); n = m.sum()
        pred.append(p[m].mean())
        obs.append(k / n)
        z = 1.96
        ph = k / n
        denom = 1 + z**2 / n
        center = (ph + z**2 / (2 * n)) / denom
        half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n * n)) / denom
        lo.append(max(0, center - half))
        hi.append(min(1, center + half))
    pred = np.array(pred); obs = np.array(obs); lo = np.array(lo); hi = np.array(hi)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 0.6], [0, 0.6], color=COL_CHANCE, lw=1.0, linestyle=(0, (3, 3)), label="perfect")
    ax.fill_between(pred, lo, hi, color=COL_STRICT, alpha=0.18, lw=0)
    ax.plot(pred, obs, color=COL_STRICT, lw=2.2, marker="o", ms=6,
            markerfacecolor="white", markeredgecolor=COL_STRICT, markeredgewidth=1.5,
            label="EDSurv-Lite\n(10 quantile bins)")
    xmax = max(pred.max(), obs.max()) * 1.05
    ax.set_xlim(0, xmax); ax.set_ylim(0, xmax)
    ax.set_xlabel("Predicted 24-h ICU probability", fontweight="bold")
    ax.set_ylabel("Observed 24-h ICU rate", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    d = np.load(PROBS, allow_pickle=True)
    risk = cum_to_horizon(d["probs"], 24)
    df = pd.DataFrame({"stay_id": d["stay_ids"].astype(int), "risk": risk})

    mds = pd.read_csv(MDSED, usecols=["general_ed_stay_id", "deterioration_icu_24h"])
    mds = mds.dropna(subset=["general_ed_stay_id"])
    mds["general_ed_stay_id"] = mds["general_ed_stay_id"].astype(int)
    by = mds.groupby("general_ed_stay_id").agg(icu_24h=("deterioration_icu_24h", "max")).reset_index()

    merged = df.merge(by, left_on="stay_id", right_on="general_ed_stay_id", how="inner")
    y = (merged["icu_24h"] > 0).astype(int).values
    p = merged["risk"].values
    print(f"Intersection N = {len(merged)}, positives = {int(y.sum())}, prevalence = {y.mean():.4f}")

    plot_roc(y, p, FIGDIR / "fig2_roc_strict3h.pdf")
    plot_calibration(y, p, FIGDIR / "fig2_calibration_strict3h.pdf")

    rng = np.random.default_rng(0)
    aps = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        aps.append(average_precision_score(y[i], p[i]))
    print(f"AUPRC {average_precision_score(y, p):.4f} "
          f"[{np.percentile(aps, 2.5):.4f}, {np.percentile(aps, 97.5):.4f}]")


if __name__ == "__main__":
    main()
