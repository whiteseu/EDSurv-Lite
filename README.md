# EDSurv-Lite

Core code for **EDSurv-Lite: A Strict Early-Window Benchmark for Multimodal ED Deterioration Prediction** (NLPCC 2026, Springer LNAI).

The benchmark keeps only modalities observed **within 3 hours of ED arrival and before any observed event**, so that a retrospective linkage cannot admit a record acquired after the outcome. An audit of an unconstrained ±48h linkage of MIMIC-IV-ED with MIMIC-CXR-JPG and MIMIC-IV-ECG finds that **20.4% of event-positive visits carry a chest radiograph recorded after the deterioration event, and 10.1% carry a post-event ECG**.

## No data in this repository

MIMIC-IV, MIMIC-IV-ED, MIMIC-CXR-JPG and MIMIC-IV-ECG are distributed under PhysioNet credentialed access, and MDS-ED under its own terms. **This repository contains code only.** No cohort file, split file, stay identifier, cached embedding or model checkpoint is included, and none will be. Obtain the source datasets yourself and point the scripts at your own copies.

## Layout

```
data/cached_dataset.py                 six-branch cached dataset (CXR, ECG, text, vitals, labs, meds)
eval/structured_baselines.py           rule-based triage scores, Cox PH, random survival forest
eval/ensemble_eval.py                  five-seed ensemble evaluation
eval/mdsed_headtohead.py               scoring the MDS-ED 24h ICU label with the composite risk
analysis/landmark_variants.py          fixed 3h prediction time, 4h landmark, arrival-only inputs
analysis/patient_cluster_bootstrap.py  bootstrap with the patient as the resampling unit
analysis/make_roc_figure.py            ROC figure, 1,000-replicate bootstrap
```

## Task

Time to the first of ICU transfer or in-hospital death within 48 hours of ED arrival, discretised into `(0,1]`, `(1,4]`, `(4,12]`, `(12,24]`, `(24,48]` hours plus a sixth output for no observed event by 48 hours. Visits with neither event are administratively censored at 48 hours. The head is a small per-branch projection, a fusion MLP and a discrete-time survival layer trained with a DeepHit-style NLL plus ranking objective (λ = 0.5); both foundation encoders stay frozen and their inference is performed once offline.

## Paths

`eval/mdsed_headtohead.py` reads the MDS-ED release from `$MDS_ED_CSV` (default `data/mds_ed.csv`). Other scripts take their cache and cohort paths as arguments; internal paths from the original runs have been removed.

## Reproducibility note

The MDS-ED intersection reported in the paper (N = 6,381, 901 positives) is reconstructed by `eval/mdsed_headtohead.py` from the official `mds_ed.csv` together with the per-visit ensemble predictions. Intervals in the paper are 1,000-replicate percentile bootstraps over visits; `analysis/patient_cluster_bootstrap.py` reproduces the patient-clustered interval for the headline estimate.

## Citation

```bibtex
@inproceedings{shen2026edsurvlite,
  author    = {Shen, Peilin and Zheng, Yulong},
  title     = {{EDSurv-Lite}: A Strict Early-Window Benchmark for Multimodal {ED} Deterioration Prediction},
  booktitle = {Natural Language Processing and Chinese Computing (NLPCC)},
  series    = {Lecture Notes in Artificial Intelligence},
  publisher = {Springer},
  year      = {2026}
}
```

## Licence

MIT, see `LICENSE`. The licence covers this code only, not any dataset.
