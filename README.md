# ICU Patient Similarity Search — MIMIC-IV 🔎

![repo health](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/TasnimPuc/bd-rooftop-solar-calculator/main/.github/health/badge.json)

"Patients like this one: what happened to them?" This repo turns the **first 24 hours of every ICU stay** in MIMIC-IV
into a 64-dimensional embedding and indexes all past patients with **FAISS**. For a new patient it retrieves the most
similar past patients and summarises their outcomes. This is retrieval-based decision support, not a black-box score.

## Data (full MIMIC-IV, BigQuery + mimic-code concepts)
- **Stays:** every adult ICU stay ≥ 24 h, split by patient 70/15/15. The training split is the searchable database of
  past patients, and the test patients are the queries.
- **Input (first 24 h only):** hourly heart rate, BP, MAP, respiratory rate, temperature, SpO₂ and glucose (GRU), plus
  first-day labs, SOFA and its components, GCS, urine output, weight, Charlson, ICU unit and admission type.
- **Outcomes (after the first 24 h):** in-hospital death, ICU stay > 7 days, invasive ventilation, vasopressor, and AKI
  stage ≥ 2 on days 2–7 (KDIGO).

## Method
1. **Encoder:** a GRU over the vitals and an MLP over the static data produce the embedding. Five outcome heads train
   it (multi-task), so that "similar" means similar in what matters clinically.
2. **Index:** L2-normalised embeddings in a FAISS `IndexFlatIP` (exact cosine). An IVF-PQ index is benchmarked for
   speed and recall@10.
3. **Evaluation:** the outcome rate among the k = 50 nearest past patients is used as a risk score (AUROC on test),
   plus lift@10. It is compared with search on **raw features** and on **PCA** of the raw features, and with the
   encoder's own heads.
4. **Demo:** a hypothetical patient typed in by hand (`DEMO_PATIENT`) → outcome rates among the 50 most similar past
   patients. The demo shows no individual records.

## Run on Kaggle (~15–30 min)
1. Import `notebooks/kaggle_run.ipynb`. Set Accelerator **GPU T4** and Internet **On** (the notebook installs
   `faiss-cpu`).
2. Add-ons → **Google Cloud Services**: attach the account that has PhysioNet BigQuery access and tick **BigQuery** +
   **Cloud Storage**.
3. Set `project_id` in the Config cell, then **Run All**. At the end, unzip `results_for_github.zip` here as `results/`.

### Fast and resumable
- **BigQuery runs once.** The query results are exported as Parquet into a private, auto-created bucket
  (`gs://<project_id>-mimic-ckpt/icu-patient-similarity-search/`). Later sessions only download them.
- **Training continues where it stopped.** The encoder saves model, optimizer, RNG and the *position inside the epoch*
  every few minutes. Run All again in a new session to continue.

## Outputs
`results/REPORT.md` (retrieval AUROC and lift, embedding map with hexagons ≥ 10 patients, FAISS speed, demo).
`patient_encoder.pth` (with preprocessing), `patients.faiss` and the index embeddings and outcomes are written to
`/kaggle/working/similarity/model/` and your bucket. **They are not published:** the index holds embeddings of real
patients.

## Local tests / CI
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
python -m pytest -q            # real SQL on DuckDB (outcome windows, no data after 24 h), FAISS vs torch search,
                               # encoder + resume, retrieval evaluation, demo, report
python tools/build_notebook.py # regenerate the Kaggle notebook after editing src/
```

## Data, privacy and licence
- **MIMIC-IV:** Johnson A. et al., *Scientific Data* 10, 1 (2023). **mimic-code:** Johnson A. et al., *JAMIA* 25(1),
  2018. This is credentialed PhysioNet data: you need your own credentialing, CITI training and a signed DUA.
- This repository contains **code and aggregate results only**. No data is sent to any online LLM API.
- Code: MIT licence.

**Limitation:** single centre, and neighbours reflect past practice. Retrieval shows what happened to similar patients,
not what would happen under a different treatment.
