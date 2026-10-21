"""5-seed v6-lite ensemble eval + MDS-ED head-to-head."""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C  # noqa: E402
from data.cached_dataset_v6lite import CachedEDSurvDatasetV6Lite, collate_v6lite  # noqa: E402
from models.v6lite_model import V6LiteSixModal  # noqa: E402
from utils.survival import integrated_brier_score  # noqa: E402
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, average_precision_score


def boot_cindex(risk, t, event, n=1000, seed=42):
    rng = np.random.default_rng(seed); cis = []
    for _ in range(n):
        idx = rng.integers(0, len(risk), len(risk))
        try: cis.append(concordance_index(t[idx], -risk[idx], event[idx]))
        except: pass
    cis = np.array(cis)
    return float(cis.mean()), float(np.quantile(cis, 0.025)), float(np.quantile(cis, 0.975))


def boot_auc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except: pass
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def cum_to_horizon(probs, horizon, edges=C.TIME_BINS_HOURS):
    cum = np.zeros(len(probs))
    for k, e in enumerate(edges):
        if e <= horizon + 1e-9: cum += probs[:, k]
    return cum


def main():
    device = torch.device("cuda")
    SEEDS = {s: C.RESULTS_DIR / f"v6lite_seed{s}" / "best.pt" for s in [1, 2, 3, 7, 42]}
    print(f"v6-lite ensemble across {len(SEEDS)} seeds")

    ds = CachedEDSurvDatasetV6Lite(
        str(C.COHORT_DIR / "cohort_split.jsonl"),
        str(C.DATA_OUT / "cache" / "cxr_raddino_embeds.npz"),
        str(C.DATA_OUT / "cache" / "ecg_features.npz"),
        str(C.DATA_OUT / "cache" / "triage_features.npz"),
        str(C.DATA_OUT / "cache" / "labs_features.npz"),
        str(C.DATA_OUT / "cache" / "meds_features.npz"),
        split="test",
    )
    loader = DataLoader(ds, 512, shuffle=False, num_workers=0, collate_fn=collate_v6lite)

    all_probs = []
    event_arr = None; t_arr = None; stays_arr = None
    for seed, ckpt in SEEDS.items():
        if not ckpt.exists():
            print(f"  seed {seed}: ckpt missing, skip"); continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        m = V6LiteSixModal(cxr_dim=768, dropout=state["args"].get("dropout", 0.2)).to(device).eval()
        m.load_state_dict(state["model_state"])
        all_logits = []; ev_l=[]; t_l=[]; stays=[]
        with torch.no_grad():
            for b in loader:
                l = m(b["cxr"].to(device), b["ecg"].to(device),
                      b["text"].to(device), b["vit"].to(device),
                      b["labs"].to(device), b["meds"].to(device))
                all_logits.append(l.float().cpu())
                ev_l.append(b["event"].cpu()); t_l.append(b["t_hours"].cpu())
                stays += [int(s) for s in b["stay_id"]]
        logits = torch.cat(all_logits).numpy()
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        all_probs.append(probs)
        if event_arr is None:
            event_arr = torch.cat(ev_l).numpy(); t_arr = torch.cat(t_l).numpy(); stays_arr = stays
        risk_ind = probs[:, :-1].sum(axis=1)
        ci, lo, hi = boot_cindex(risk_ind, t_arr, event_arr)
        ibs = integrated_brier_score(probs, t_arr, event_arr)
        print(f"  seed {seed:3d}: C-index {ci:.4f} [{lo:.4f}, {hi:.4f}]  IBS {ibs:.4f}")

    if not all_probs:
        print("No checkpoints found"); return
    ensemble_probs = np.mean(all_probs, axis=0)
    risk_ens = ensemble_probs[:, :-1].sum(axis=1)
    ci, lo, hi = boot_cindex(risk_ens, t_arr, event_arr)
    ibs = integrated_brier_score(ensemble_probs, t_arr, event_arr)
    print(f"\n=== v6-lite ENSEMBLE ({len(all_probs)} seeds) ===")
    print(f"  C-index: {ci:.4f} [{lo:.4f}, {hi:.4f}]  IBS: {ibs:.4f}")
    print(f"  N test: {len(stays_arr)}")

    # MDS-ED head-to-head
    print(f"\n=== MDS-ED head-to-head ===")
    mds = pd.read_csv(
        os.environ.get("MDS_ED_CSV", "data/mds_ed.csv"),
        usecols=["general_ed_stay_id", "deterioration_icu_24h", "deterioration_mortality_1d"],
    )
    mds = mds.dropna(subset=["general_ed_stay_id"])
    mds["general_ed_stay_id"] = mds["general_ed_stay_id"].astype(int)
    mds_by_stay = mds.groupby("general_ed_stay_id").agg(
        icu_24h=("deterioration_icu_24h", "max"),
        mortality_1d=("deterioration_mortality_1d", "max"),
    ).reset_index()
    risk_24h_ens = cum_to_horizon(ensemble_probs, 24)
    df_pred = pd.DataFrame({"stay_id": stays_arr, "risk_24h": risk_24h_ens})
    merged = df_pred.merge(mds_by_stay, left_on="stay_id", right_on="general_ed_stay_id", how="inner")
    print(f"  intersection N = {len(merged)}")
    for ep, ref in [("icu_24h", 0.9147), ("mortality_1d", 0.9600)]:
        y = (merged[ep] > 0).astype(int).values
        p_ = merged["risk_24h"].values
        if y.sum() == 0: continue
        auc_m, auc_lo, auc_hi = boot_auc(y, p_)
        ap = average_precision_score(y, p_)
        beats = auc_lo > ref
        print(f"  {ep}: AUROC {auc_m:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]  AUPRC {ap:.4f}  "
              f"(MDS-ED {ref})  beats={beats}  N_pos={int(y.sum())}")


if __name__ == "__main__":
    main()
