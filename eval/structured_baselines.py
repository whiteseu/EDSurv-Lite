from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis


ALTERED_KW = re.compile(
    r"\b(altered\s+mental|confused|confusion|disorient|unconscious|unresponsive|"
    r"comatose|coma\b|stupor|stuporous|obtunded|lethargic|lethargy|lethagic|drowsy|"
    r"drowsiness|somnolen|encephalopath|delirium|delirious|"
    r"change\s+in\s+mental\s+status|ams\b)",
    re.IGNORECASE,
)


def parse_num(v):
    try:
        if v is None:
            return np.nan
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return np.nan
            return float(v)
        return float(v)
    except Exception:
        return np.nan


def news2_rr(rr):
    if np.isnan(rr):
        return 0
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def news2_spo2(s):
    if np.isnan(s):
        return 0
    if s <= 91:
        return 3
    if s <= 93:
        return 2
    if s <= 95:
        return 1
    return 0


def news2_temp_f(t):
    if np.isnan(t):
        return 0
    c = (t - 32) * 5.0 / 9.0
    if c <= 35.0:
        return 3
    if c <= 36.0:
        return 1
    if c <= 38.0:
        return 0
    if c <= 39.0:
        return 1
    return 2


def news2_sbp(s):
    if np.isnan(s):
        return 0
    if s <= 90:
        return 3
    if s <= 100:
        return 2
    if s <= 110:
        return 1
    if s <= 219:
        return 0
    return 3


def news2_hr(h):
    if np.isnan(h):
        return 0
    if h <= 40:
        return 3
    if h <= 50:
        return 1
    if h <= 90:
        return 0
    if h <= 110:
        return 1
    if h <= 130:
        return 2
    return 3


def avpu_proxy(chief_complaint):
    if not isinstance(chief_complaint, str) or not chief_complaint.strip():
        return 0
    return 3 if ALTERED_KW.search(chief_complaint) else 0


def partial_news2(row):
    return (
        news2_hr(row["heartrate"])
        + news2_rr(row["resprate"])
        + news2_spo2(row["o2sat"])
        + news2_sbp(row["sbp"])
        + news2_temp_f(row["temperature"])
    )


def full_news2(row):
    return partial_news2(row) + avpu_proxy(row["chief_complaint"])


def qsofa(row):
    score = 0
    if not np.isnan(row["resprate"]) and row["resprate"] >= 22:
        score += 1
    if not np.isnan(row["sbp"]) and row["sbp"] <= 100:
        score += 1
    if avpu_proxy(row["chief_complaint"]) > 0:
        score += 1
    return score


def bootstrap_metric(scores, times, events, metric="cindex", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    scores = np.asarray(scores)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    y24 = ((events == 1) & (times <= 24.0)).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        try:
            if metric == "cindex":
                val = concordance_index(times[idx], -scores[idx], events[idx])
            elif metric == "auroc24":
                if len(np.unique(y24[idx])) < 2:
                    continue
                val = roc_auc_score(y24[idx], scores[idx])
            elif metric == "auprc24":
                if len(np.unique(y24[idx])) < 2:
                    continue
                val = average_precision_score(y24[idx], scores[idx])
            else:
                raise ValueError(metric)
        except Exception:
            if metric == "cindex":
                val = 0.5
            else:
                continue
        vals.append(val)
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def point_metrics(scores, times, events):
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    y24 = ((events == 1) & (times <= 24.0)).astype(int)
    return {
        "c_index": float(concordance_index(times, -scores, events)),
        "auroc_24h_composite": float(roc_auc_score(y24, scores)),
        "auprc_24h_composite": float(average_precision_score(y24, scores)),
        "n_24h_positive": int(y24.sum()),
    }


def to_sksurv_y(times, events):
    times = np.maximum(np.asarray(times, dtype=float), 0.001)
    events = np.asarray(events, dtype=bool)
    return np.array(list(zip(events, times)), dtype=[("event", bool), ("time", float)])


def load_npz_index(path):
    d = np.load(path, allow_pickle=True)
    return {int(s): i for i, s in enumerate(d["stay_ids"])}, d


def build_table(args):
    cxr_idx, cxr = load_npz_index(args.cxr_cache)
    ecg_idx, ecg = load_npz_index(args.ecg_cache)
    tri_idx, tri = load_npz_index(args.triage_cache)
    labs_idx, labs = load_npz_index(args.labs_cache)
    meds_idx, meds = load_npz_index(args.meds_cache)

    tri_vit = tri["vitals"]
    ecg_feats = ecg["features_z"]
    labs_feats = np.concatenate([labs["features"], labs["missingness"]], axis=1)
    meds_feats = meds["features"]

    rows = []
    feature_rows = []
    with open(args.cohort, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sid = int(r["stay_id"])
            ci = cxr_idx.get(sid)
            ei = ecg_idx.get(sid)
            ti = tri_idx.get(sid)
            li = labs_idx.get(sid)
            mi = meds_idx.get(sid)
            model_valid = (
                ci is not None
                and bool(cxr["valid"][ci])
                and ei is not None
                and bool(ecg["valid"][ei])
                and ti is not None
                and bool(tri["valid"][ti])
                and li is not None
                and mi is not None
            )
            if not model_valid:
                continue

            triage = r.get("triage") or {}
            dem = r.get("demographics") or {}
            base = {
                "stay_id": sid,
                "subject_id": int(r["subject_id"]),
                "split": r.get("split", "train"),
                "t": float(r["t_event_h"]),
                "event": int(r["event"]),
                "acuity": parse_num(triage.get("acuity")),
                "heartrate": parse_num(triage.get("heartrate")),
                "resprate": parse_num(triage.get("resprate")),
                "o2sat": parse_num(triage.get("o2sat")),
                "sbp": parse_num(triage.get("sbp")),
                "dbp": parse_num(triage.get("dbp")),
                "temperature": parse_num(triage.get("temperature")),
                "pain": parse_num(triage.get("pain")),
                "chief_complaint": triage.get("chief_complaint", "") or "",
                "gender_m": 1.0 if dem.get("gender") == "M" else 0.0,
                "anchor_age": float(dem.get("anchor_age") or 0.0),
            }
            feats = np.concatenate(
                [
                    tri_vit[ti].astype(float),
                    ecg_feats[ei].astype(float),
                    labs_feats[li].astype(float),
                    meds_feats[mi].astype(float),
                    np.asarray([base["gender_m"], base["anchor_age"]], dtype=float),
                ]
            )
            rows.append(base)
            feature_rows.append(feats)

    df = pd.DataFrame(rows)
    X = np.vstack(feature_rows)
    return df, X


def add_result(results, name, scores, test_df, n_boot):
    pm = point_metrics(scores, test_df["t"].values, test_df["event"].values)
    ci = bootstrap_metric(scores, test_df["t"].values, test_df["event"].values, "cindex", n_boot=n_boot)
    auroc = bootstrap_metric(scores, test_df["t"].values, test_df["event"].values, "auroc24", n_boot=n_boot)
    auprc = bootstrap_metric(scores, test_df["t"].values, test_df["event"].values, "auprc24", n_boot=n_boot)
    results[name] = {
        **pm,
        "c_index_ci": ci,
        "auroc_24h_composite_ci": auroc,
        "auprc_24h_composite_ci": auprc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=r"experiments\strict_3h_early\cohort_split_strict3h.jsonl")
    parser.add_argument("--cxr_cache", default=r"data\cache\cxr_embeddings.npz")
    parser.add_argument("--ecg_cache", default=r"data\cache\ecg_features.npz")
    parser.add_argument("--triage_cache", default=r"data\cache\triage_features.npz")
    parser.add_argument("--labs_cache", default=r"data\cache\labs_features.npz")
    parser.add_argument("--meds_cache", default=r"data\cache\meds_features.npz")
    parser.add_argument("--out_dir", default=r"experiments\strict_3h_early\baselines")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--rsf_estimators", type=int, default=200)
    parser.add_argument("--rsf_depth", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building strict 3h model-aligned table")
    df, X = build_table(args)
    split_counts = df["split"].value_counts().to_dict()
    print(f"  N={len(df)} split_counts={split_counts} X={X.shape}")
    train_mask = df["split"].values == "train"
    val_mask = df["split"].values == "val"
    test_mask = df["split"].values == "test"
    test_df = df.loc[test_mask].reset_index(drop=True)
    print(f"  test N={len(test_df)} events={int(test_df['event'].sum())}")

    results = {}
    rng = np.random.default_rng(42)
    add_result(results, "Random", rng.random(len(test_df)), test_df, args.n_boot)

    acuity = test_df["acuity"].fillna(3).values
    esi = 5.0 - acuity
    add_result(results, "ESI", esi, test_df, args.n_boot)

    q = test_df.apply(qsofa, axis=1).values.astype(float)
    add_result(results, "qSOFA", q, test_df, args.n_boot)

    pn = test_df.apply(partial_news2, axis=1).values.astype(float)
    add_result(results, "Partial NEWS2", pn, test_df, args.n_boot)

    fn = test_df.apply(full_news2, axis=1).values.astype(float)
    add_result(results, "NEWS2 + AVPU proxy", fn, test_df, args.n_boot)

    add_result(results, "NEWS2 + ESI", fn + esi, test_df, args.n_boot)

    print(f"[{time.strftime('%H:%M:%S')}] Fitting Cox PH")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    y_train = to_sksurv_y(df.loc[train_mask, "t"].values, df.loc[train_mask, "event"].values)
    try:
        cox = CoxPHSurvivalAnalysis(alpha=1.0)
        cox.fit(X_train, y_train)
    except Exception as exc:
        print(f"  Cox alpha=1.0 failed: {exc}")
        cox = CoxPHSurvivalAnalysis(alpha=10.0)
        cox.fit(X_train, y_train)
    cox_risk = cox.predict(X_test)
    add_result(results, "Cox PH structured", cox_risk, test_df, args.n_boot)

    print(f"[{time.strftime('%H:%M:%S')}] Fitting Random Survival Forest")
    rsf = RandomSurvivalForest(
        n_estimators=args.rsf_estimators,
        max_depth=args.rsf_depth,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X_train, y_train)
    rsf_risk = rsf.predict(X_test)
    add_result(results, "Random Survival Forest structured", rsf_risk, test_df, args.n_boot)

    summary = {
        "cohort": str(args.cohort),
        "filter": "strict 3h cohort plus v6lite model-valid cache alignment",
        "n_total_model_valid": int(len(df)),
        "split_counts_model_valid": {str(k): int(v) for k, v in split_counts.items()},
        "test_n": int(len(test_df)),
        "test_events": int(test_df["event"].sum()),
        "test_24h_composite_positive": int(((test_df["event"].values == 1) & (test_df["t"].values <= 24.0)).sum()),
        "feature_dim_structured": int(X.shape[1]),
        "runtime_seconds": float(time.time() - t0),
        "results": results,
    }

    json_path = out_dir / "strict3h_baseline_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    lines = [
        "# Strict 3h baseline results",
        "",
        "All rows use the strict 3h cohort and the same cache-valid test set as v6lite.",
        "",
        f"Model-valid test N = {summary['test_n']}; events = {summary['test_events']}; "
        f"24h composite positives = {summary['test_24h_composite_positive']}.",
        "",
        "| Baseline | C-index [95% CI] | 24h composite AUROC [95% CI] | 24h composite AUPRC [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            "| {name} | {c:.4f} [{cl:.4f}, {ch:.4f}] | {a:.4f} [{al:.4f}, {ah:.4f}] | "
            "{p:.4f} [{pl:.4f}, {ph:.4f}] |".format(
                name=name,
                c=r["c_index"],
                cl=r["c_index_ci"]["lo"],
                ch=r["c_index_ci"]["hi"],
                a=r["auroc_24h_composite"],
                al=r["auroc_24h_composite_ci"]["lo"],
                ah=r["auroc_24h_composite_ci"]["hi"],
                p=r["auprc_24h_composite"],
                pl=r["auprc_24h_composite_ci"]["lo"],
                ph=r["auprc_24h_composite_ci"]["hi"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Clinical scores use triage-time variables only.",
            "- Structured baselines use cached triage vitals, ECG numeric features, labs, meds, age, and sex.",
            "- The 24h endpoint here is the 24h version of the project composite event, not the MDS-ED ICU-only endpoint.",
        ]
    )
    md_path = out_dir / "strict3h_baselines.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    pred_path = out_dir / "strict3h_baseline_predictions.npz"
    np.savez_compressed(
        pred_path,
        stay_ids=test_df["stay_id"].values.astype(np.int64),
        times=test_df["t"].values.astype(float),
        events=test_df["event"].values.astype(int),
        esi=esi.astype(float),
        qsofa=q.astype(float),
        partial_news2=pn.astype(float),
        news2_avpu=fn.astype(float),
        news2_esi=(fn + esi).astype(float),
        cox_risk=np.asarray(cox_risk, dtype=float),
        rsf_risk=np.asarray(rsf_risk, dtype=float),
    )

    print("\n".join(lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
