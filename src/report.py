from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .metrics import md


def build(out: Path, res: pd.DataFrame, s: dict) -> str:
    rep = out / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy(f, rep / "figures" / f.name)
    for f in (out / "tables").glob("*.csv"):
        shutil.copy(f, rep / f.name)
    c = s["config"]
    auroc = res.pivot(index="outcome", columns="method", values="auroc").reset_index()
    lift = res.dropna(subset=["lift_at_k"]).pivot(index="outcome", columns="method", values="lift_at_k").reset_index()
    demo_in = ", ".join(f"{k}={v}" for k, v in s["demo_patient"].items())
    demo = pd.DataFrame({"": list(s["demo"].keys()), "value": list(s["demo"].values())})
    text = f"""# Results — ICU patient similarity search (MIMIC-IV)

**Database:** {s['stays']:,} adult ICU stays ≥ {c['seq_hours']} h (full MIMIC-IV), split by patient
{'/'.join(str(int(x * 100)) for x in c['split'])}. Each stay becomes a {c['emb_dim']}-d embedding of its first
{c['seq_hours']} h (hourly vitals + first-day labs, SOFA, GCS, urine, weight, Charlson). The training split is the
searchable "past patients" index (FAISS, cosine); test patients are the queries.
**Outcomes (after the first 24 h):** in-hospital death, ICU stay > 7 days, invasive ventilation, vasopressor,
AKI stage ≥ 2 (days 2–7).

## Data
{md(pd.DataFrame(s['split']))}

## Do similar past patients predict what happens next?
Outcome rate among the k = {c['k']} nearest past patients, used as a risk score → AUROC on test queries.
"Encoder heads" = the network's own direct prediction (upper reference).

{md(auroc)}

![kNN AUROC](figures/knn_auroc.png)

**Lift@{c['k_lift']}:** for test patients who had the outcome, how many times more often their {c['k_lift']} nearest
neighbours had it than a random past patient.

{md(lift)}

## Embedding space
![map](figures/embedding_map.png)

## Speed (FAISS)
{md(pd.DataFrame([s['ann']]))}

## Demo query (hypothetical patient, not a real record)
Input: {demo_in}

{md(demo)}

---
Encoder best epoch {s['encoder_best_epoch']} · runtime {s['runtime_min']} min. Aggregate results only; the index and the
encoder stay private (the index holds embeddings of real patients). This is a decision-support prototype, not a
clinical tool.
"""
    (rep / "REPORT.md").write_text(text, encoding="utf-8")
    return text
