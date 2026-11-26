"""Project-specific texts for tools/build_notebook.py."""
ACCELERATOR = "nvidiaTeslaT4"
EXTRA_PIP = "faiss-cpu"
PROBES = ["mimiciv_3_1_derived.icustay_detail", "mimiciv_derived.icustay_detail", "mimiciv_3_1_icu.icustays",
          "mimiciv_icu.icustays"]

INTRO = """# 🔎 ICU Patient Similarity Search — MIMIC-IV (Kaggle)

Turns the **first 24 h of every ICU stay** into an embedding and indexes all past patients with **FAISS**. For a new
patient it finds the most similar past patients and summarises **what happened to them**: death, long ICU stay,
ventilation, vasopressors, AKI.

**Before Run All**
1. Accelerator **GPU T4** · Internet **On** (installs `faiss-cpu`).
2. Add-ons → **Google Cloud Services** → attach the Google account with PhysioNet BigQuery access → tick
   **BigQuery** and **Cloud Storage**.
3. Set `project_id` in the Config cell. The `mimiciv_derived` (mimic-code) dataset must be visible in BigQuery.

**Resume:** query results (Parquet) and encoder checkpoints (model + optimizer + position inside the epoch) are kept in
a private bucket (`gs://<project_id>-mimic-ckpt/icu-patient-similarity-search/`). If the session stops, **Run All**
again.

Pipeline (~15–30 min): BigQuery → features → encoder → FAISS index → retrieval evaluation vs raw-feature / PCA search →
exact vs IVF-PQ speed → demo query with a hypothetical patient → report.
To try your own query, edit `DEMO_PATIENT` in the demo cell (values you leave out are set to typical values)."""

CONFIG = """import sys
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

cfg = Config(
    project_id="mimic-cxr-training",   # <-- your GCP project ID (billing + BigQuery); bucket is created automatically
    work_dir="/kaggle/working/similarity",
    emb_dim=64, epochs=40,
    k=50,                              # neighbours used for the outcome summary
    run_name="main",                   # change to start a fresh set of checkpoints
)
cfg"""
