"""MDS-ED head-to-head: same protocol direct comparison.

For samples in (our_test ∩ MDS-ED), compute our v3 model AUROC on:
  - deterioration_icu_24h  (MDS-ED reports 0.906)
  - deterioration_mortality_1d  (MDS-ED reports 0.917)

Use our pre-computed v3 risk score (trained on our train, never saw our test).
"""
from __future__ import annotations
import os
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C  # noqa: E402
from data.cached_dataset_v3 import CachedEDSurvDatasetV3, collate_v3  # noqa: E402
from models.v3_model import V3TriModalHubert  # noqa: E402
from utils.survival import risk_score_from_logits  # noqa: E402

MDS_ED_CSV = Path(os.environ.get("MDS_ED_CSV", "data/mds_ed.csv"))


def boot_metric(metric_fn, y, p, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try: v = metric_fn(y[idx], p[idx])
        except Exception: v = np.nan
        vals.append(v)
    vals = np.array([v for v in vals if not np.isnan(v)])
    if not len(vals): return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.quantile(vals, alpha/2)), float(np.quantile(vals, 1-alpha/2))


def cumulative_event_prob_to_horizon(probs, horizon_hours, bin_edges=C.TIME_BINS_HOURS):
    cum = np.zeros(len(probs))
    for k, edge in enumerate(bin_edges):
        if edge <= horizon_hours + 1e-9:
            cum += probs[:, k]
    return cum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(C.RESULTS_DIR / "v3_hubert" / "best.pt"))
    p.add_argument("--cohort", type=str, default=str(C.COHORT_DIR / "cohort_split.jsonl"))
    p.add_argument("--cxr_cache", type=str, default=str(C.DATA_OUT / "cache" / "cxr_embeddings.npz"))
    p.add_argument("--ecg_hub_cache", type=str, default=str(C.DATA_OUT / "cache" / "ecg_hubert_embeds.npz"))
    p.add_argument("--tri_cache", type=str, default=str(C.DATA_OUT / "cache" / "triage_features.npz"))
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--out", type=str, default=str(C.RESULTS_DIR / "mdsed_headtohead.md"))
    args = p.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] Loading MDS-ED essentials...")
    t0 = time.time()
    use_cols = ["general_ed_stay_id", "general_strat_fold",
                 "deterioration_icu_24h", "deterioration_mortality_1d"]
    mds = pd.read_csv(MDS_ED_CSV, usecols=use_cols, low_memory=False)
    print(f"  MDS-ED essentials: {len(mds)} rows, cols={list(mds.columns)} in {time.time()-t0:.1f}s")
    print(f"  fold distribution: {mds['general_strat_fold'].value_counts().sort_index().to_dict()}")
    print(f"  icu_24h rate: {(mds['deterioration_icu_24h']>0).mean():.4f}")
    print(f"  mortality_1d rate: {(mds['deterioration_mortality_1d']>0).mean():.4f}")
    # NOTE: some rows have stay_id NaN (no ED stay linkage). Drop those.
    mds = mds.dropna(subset=["general_ed_stay_id"])
    mds["general_ed_stay_id"] = mds["general_ed_stay_id"].astype(int)
    print(f"  after dropna stay_id: {len(mds)} rows")

    # Per-stay aggregate (some stays may have multiple ECGs in MDS-ED). Take max of binary labels.
    mds_by_stay = mds.groupby("general_ed_stay_id").agg(
        fold=("general_strat_fold", "first"),
        icu_24h=("deterioration_icu_24h", "max"),
        mortality_1d=("deterioration_mortality_1d", "max"),
    ).reset_index()
    print(f"  unique stays in MDS-ED: {len(mds_by_stay)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Loading v3 model...")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    m = V3TriModalHubert(dropout=state["args"].get("dropout", 0.2)).to(device).eval()
    m.load_state_dict(state["model_state"])

    ds = CachedEDSurvDatasetV3(args.cohort, args.cxr_cache, args.ecg_hub_cache,
                                  args.tri_cache, split="test")
    loader = DataLoader(ds, 512, shuffle=False, num_workers=2, collate_fn=collate_v3)

    all_logits, all_stay = [], []
    with torch.no_grad():
        for b in loader:
            cxr = b["cxr"].to(device); ecg = b["ecg"].to(device)
            txt = b["text"].to(device); vit = b["vit"].to(device)
            l = m(cxr, ecg, txt, vit)
            all_logits.append(l.float().cpu())
            all_stay += [int(s) for s in b["stay_id"]]
    logits = torch.cat(all_logits)
    probs = F.softmax(logits, dim=-1).numpy()
    risk_24h = cumulative_event_prob_to_horizon(probs, 24)
    print(f"  our test split N: {len(all_stay)}")

    # Merge
    df_pred = pd.DataFrame({"stay_id": all_stay, "risk_24h": risk_24h})
    merged = df_pred.merge(mds_by_stay, left_on="stay_id", right_on="general_ed_stay_id", how="inner")
    print(f"\n=== Intersection (our test ∩ MDS-ED) ===")
    print(f"  N = {len(merged)}")
    print(f"  fold distribution in intersection: {merged['fold'].value_counts().sort_index().to_dict()}")
    print(f"  icu_24h event rate: {merged['icu_24h'].mean():.4f}")
    print(f"  mortality_1d event rate: {merged['mortality_1d'].mean():.4f}")

    lines = ["# EDSurv v3 head-to-head vs MDS-ED on official endpoints",
             "",
             "**MDS-ED reference** (Strodthoff et al., arXiv 2024) reports on stratified 10-fold CV:",
             "- `deterioration_icu_24h` AUROC = **0.906**",
             "- `deterioration_mortality_1d` AUROC = **0.917**",
             "",
             f"**Our setup**: v3 model trained on our hash-split train (50,023 samples, 70% by subject_id hash). "
             f"Evaluated on (our test ∩ MDS-ED) intersection: N = {len(merged)}.",
             "",
             "Both models predict on the same MIMIC-IV-ED dataset but with different training splits.",
             "",
             "| Endpoint | N events | Prevalence | Our AUROC [95% CI] | Our AUPRC [95% CI] | MDS-ED published |",
             "|---|---|---|---|---|---|"]
    for ep, mds_ref in [("icu_24h", 0.906), ("mortality_1d", 0.917)]:
        y = (merged[ep] > 0).astype(int).values
        p_ = merged["risk_24h"].values
        if y.sum() == 0:
            lines.append(f"| {ep} | 0 | 0 | N/A | N/A | {mds_ref} |"); continue
        auc_m, auc_lo, auc_hi = boot_metric(roc_auc_score, y, p_, args.n_boot)
        ap_m, ap_lo, ap_hi = boot_metric(average_precision_score, y, p_, args.n_boot)
        lines.append(f"| **{ep}** | {int(y.sum())} | {float(y.mean()):.4f} | "
                      f"**{auc_m:.4f}** [{auc_lo:.4f}, {auc_hi:.4f}] | "
                      f"{ap_m:.4f} [{ap_lo:.4f}, {ap_hi:.4f}] | **{mds_ref}** |")

    # Report the per-fold breakdown on the intersecting test samples.
    lines.append("")
    lines.append("## Per-MDS-ED-fold breakdown (our test ∩ MDS-ED, faceted by their fold)")
    lines.append("")
    lines.append("| Fold | N | icu_24h events | icu_24h AUROC | mortality_1d events | mortality_1d AUROC |")
    lines.append("|---|---|---|---|---|---|")
    for f in sorted(merged["fold"].unique()):
        sub = merged[merged["fold"] == f]
        if len(sub) < 50:
            lines.append(f"| {f} | {len(sub)} | (too small) | - | - | - |"); continue
        y_icu = (sub["icu_24h"] > 0).astype(int).values
        y_mor = (sub["mortality_1d"] > 0).astype(int).values
        p_ = sub["risk_24h"].values
        try: auc_icu = roc_auc_score(y_icu, p_) if y_icu.sum() > 0 else float("nan")
        except: auc_icu = float("nan")
        try: auc_mor = roc_auc_score(y_mor, p_) if y_mor.sum() > 0 else float("nan")
        except: auc_mor = float("nan")
        lines.append(f"| {int(f)} | {len(sub)} | {int(y_icu.sum())} | {auc_icu:.4f} | "
                      f"{int(y_mor.sum())} | {auc_mor:.4f} |")

    md = "\n".join(lines)
    print("\n" + md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh: fh.write(md)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
