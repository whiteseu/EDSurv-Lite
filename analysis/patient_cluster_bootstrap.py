"""Patient-clustered bootstrap for the strict-3h test C-index.

The paper's intervals resample visits. Because a patient can contribute more
than one ED visit, visit-level resampling treats correlated observations as
independent and can understate uncertainty. This reruns the same statistic
with the patient as the resampling unit.
"""
import io, sys, csv
import numpy as np
from lifelines.utils import concordance_index

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

d = np.load('pull/ensemble_strict3h_probs.npz')
probs, event, t, stay = d['probs'], d['event'].astype(int), d['t_hours'].astype(float), d['stay_ids'].astype(int)
risk = probs[:, :-1].sum(1)

m = {}
with open('pull/stay2subj.csv') as f:
    for row in csv.DictReader(f):
        m[int(row['stay_id'])] = int(row['subject_id'])
subj = np.array([m[s] for s in stay])

uniq = np.unique(subj)
print(f'test visits {len(stay)}, unique patients {len(uniq)}, visits/patient {len(stay)/len(uniq):.3f}')
pt = concordance_index(t, -risk, event)
print(f'point estimate C-index = {pt:.4f}')

# index lists per patient
byp = {}
for i, s in enumerate(subj):
    byp.setdefault(s, []).append(i)
groups = [np.array(byp[s]) for s in uniq]

def boot(unit, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        if unit == 'visit':
            idx = rng.integers(0, len(t), len(t))
        else:
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[k] for k in pick])
        try:
            vals.append(concordance_index(t[idx], -risk[idx], event[idx]))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

lo1, hi1 = boot('visit')
lo2, hi2 = boot('patient')
print(f'  visit-level    95% CI [{lo1:.4f}, {hi1:.4f}]  width {hi1-lo1:.4f}')
print(f'  patient-level  95% CI [{lo2:.4f}, {hi2:.4f}]  width {hi2-lo2:.4f}')
print(f'  width ratio    {(hi2-lo2)/(hi1-lo1):.2f}x')
