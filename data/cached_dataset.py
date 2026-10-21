"""v6-lite cached dataset: CXR + ECG-numeric (9d) + Triage + Labs + Meds."""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C  # noqa: E402
from data.dataset import time_to_bin  # noqa: E402


class CachedEDSurvDatasetV6Lite(Dataset):
    def __init__(
        self,
        cohort_jsonl: str | Path,
        cxr_cache: str | Path,
        ecg_num_cache: str | Path,
        triage_cache: str | Path,
        labs_cache: str | Path,
        meds_cache: str | Path,
        split: str = "train",
        max_items: Optional[int] = None,
    ):
        def make_idx(npz):
            d = np.load(npz, allow_pickle=True)
            return {int(s): i for i, s in enumerate(d["stay_ids"])}, d
        cxr_idx, cxr_d = make_idx(cxr_cache)
        ecg_idx, ecg_d = make_idx(ecg_num_cache)
        tri_idx, tri_d = make_idx(triage_cache)
        labs_idx, labs_d = make_idx(labs_cache)
        meds_idx, meds_d = make_idx(meds_cache)

        self.cxr_embeds = cxr_d["embeds"]
        self.ecg_feats = ecg_d["features_z"]  # 9d z-normalized
        self.tri_text = tri_d["text_embed"]
        self.tri_vit = tri_d["vitals"]
        self.labs_feats = np.concatenate([labs_d["features"], labs_d["missingness"]], axis=1)
        self.meds_feats = meds_d["features"]

        cxr_valid = cxr_d["valid"]; ecg_valid = ecg_d["valid"]
        tri_valid = tri_d["valid"]; labs_valid = labs_d["valid"]; meds_valid = meds_d["valid"]

        records = []
        with open(cohort_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split", split) != split: continue
                sid = int(r["stay_id"])
                ci = cxr_idx.get(sid); ei = ecg_idx.get(sid); ti = tri_idx.get(sid)
                li = labs_idx.get(sid); mi = meds_idx.get(sid)
                if ci is None or not bool(cxr_valid[ci]): continue
                if ei is None or not bool(ecg_valid[ei]): continue
                if ti is None or not bool(tri_valid[ti]): continue
                if li is None or mi is None: continue
                r["_ci"] = ci; r["_ei"] = ei; r["_ti"] = ti; r["_li"] = li; r["_mi"] = mi
                records.append(r)
        if max_items: records = records[:max_items]
        self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        cxr = torch.from_numpy(self.cxr_embeds[r["_ci"]]).float()
        ecg = torch.from_numpy(self.ecg_feats[r["_ei"]]).float()
        text = torch.from_numpy(self.tri_text[r["_ti"]]).float()
        vit = torch.from_numpy(self.tri_vit[r["_ti"]]).float()
        labs = torch.from_numpy(self.labs_feats[r["_li"]]).float()
        meds = torch.from_numpy(self.meds_feats[r["_mi"]]).float()
        ev = int(r["event"]); t = float(r["t_event_h"])
        b = time_to_bin(t)
        if ev == 0: b = len(C.TIME_BINS_HOURS)
        return {"cxr": cxr, "ecg": ecg, "text": text, "vit": vit, "labs": labs, "meds": meds,
                "event": ev, "t_hours": t, "bin_index": b,
                "stay_id": r["stay_id"], "subject_id": r["subject_id"]}


def collate_v6lite(batch):
    return {
        "cxr": torch.stack([b["cxr"] for b in batch]),
        "ecg": torch.stack([b["ecg"] for b in batch]),
        "text": torch.stack([b["text"] for b in batch]),
        "vit": torch.stack([b["vit"] for b in batch]),
        "labs": torch.stack([b["labs"] for b in batch]),
        "meds": torch.stack([b["meds"] for b in batch]),
        "event": torch.tensor([b["event"] for b in batch], dtype=torch.long),
        "t_hours": torch.tensor([b["t_hours"] for b in batch], dtype=torch.float32),
        "bin_index": torch.tensor([b["bin_index"] for b in batch], dtype=torch.long),
        "stay_id": [b["stay_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }
