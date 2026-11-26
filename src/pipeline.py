"""BigQuery (MIMIC-IV ICU stays, full) -> first-24-h inputs + later outcomes -> resumable encoder -> FAISS index of
past patients -> retrieval evaluation vs raw-feature / PCA search -> speed test -> demo query -> report + .pth (bucket only)."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cohort, features, model as enc_model, search
from .bq import BigQuerySource, materialize, read
from .config import Config
from .metrics import md, suppress
from .trainer import history

log = logging.getLogger("similarity")

DEMO_PATIENT = {"age": 72, "female": 0, "sofa": 9, "cardiovascular": 3, "respiration": 2, "renal": 2,
                "creatinine_max": 2.4, "bun_max": 48, "wbc_max": 18, "platelets_min": 110, "gcs_min": 13,
                "hr": 112, "mbp": 62, "sbp": 92, "rr": 26, "spo2": 93, "temp": 38.4,
                "first_careunit": "Medical Intensive Care Unit (MICU)", "admission_type": "EW EMER."}


def run(cfg: Config, source=None) -> dict:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    for noisy in ("google", "urllib3", "matplotlib", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(cfg.work_dir)
    for sub in ("figures", "model", "report", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    from .store import make_store
    store = make_store(cfg)
    src = source or BigQuerySource(cfg, store, cfg.datasets())
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("device: %s | bucket: %s | FAISS: %s", device, store.uri(), search._faiss() is not None)

    # %% ICU stays from BigQuery | All adult MIMIC-IV ICU stays ≥ 24 h: first-24-h data (vitals hourly, first-day labs/SOFA/GCS/urine, Charlson) and outcomes after 24 h. Cached as Parquet in your bucket.
    # show: features.summary(d)
    coh = read(materialize(src, store, "sim_cohort", cohort.cohort_sql(cfg), cfg.cache_dir))
    hourly = read(materialize(src, store, "sim_vitals", cohort.hourly_sql(cfg), cfg.cache_dir))
    d = features.build(coh, hourly, cfg)
    fp = hashlib.sha1(json.dumps([len(d.Y), float(d.Y.sum()), list(d.static.shape), cfg.seed, cfg.emb_dim]).encode()).hexdigest()[:8]
    key = lambda name: f"ckpt/{cfg.run_name}-{fp}/{name}"

    # %% Train the patient encoder (resumable mid-epoch) | GRU (vitals) + MLP (static) → 64-d embedding with five outcome heads. Model, optimizer and the position inside the epoch are saved to the bucket every few minutes.
    # show: history(st, "val_mean_auroc")
    model, st = enc_model.train(cfg, d, device, store, key("encoder"))
    allidx = np.arange(len(d.Y))
    emb, head_p = enc_model.encode(model, d, allidx, device)

    # %% Build the FAISS index | Database = training patients (the "past patients"); queries = test patients, never seen in training.
    # show: pd.Series({"indexed patients": len(tr), "dimension": emb.shape[1], "backend": "FAISS" if index.faiss else "torch"})
    tr, te = d.idx("train"), d.idx("test")
    index = search.Index(emb[tr], "ip")

    # %% Retrieval evaluation | For each test patient, the outcome rate of its k nearest past patients is used as a risk estimate. Compared with search on raw features and on PCA; `lift_at_k` = how enriched the top neighbours of positive patients are.
    # show: res.pivot(index="outcome", columns="method", values="auroc")
    res = search.evaluate(d, emb, head_p, cfg)
    res.to_csv(out / "tables" / "retrieval.csv", index=False)
    search.plot_knn(res[~res.method.str.startswith("Encoder")], out / "figures" / "knn_auroc.png")
    search.plot_map(emb[te], d.Y[te, 0], out / "figures" / "embedding_map.png")

    # %% Speed: exact vs approximate index | FAISS IVF-PQ compresses each patient to a few bytes; recall@10 against exact search.
    # show: pd.Series(ann)
    ann = search.ann_benchmark(emb[tr], emb[te], cfg)

    # %% Demo query | A HYPOTHETICAL patient typed in by hand (not a real record). The answer only shows outcome rates among the most similar past patients.
    # show: demo_out
    demo_out = search.demo(DEMO_PATIENT, model, d.prep, index, d.Y[tr], cfg, device)

    # %% Export + report | Encoder `.pth` (with preprocessing) and the FAISS index go to /kaggle/working and your private bucket only: the index holds embeddings of real patients.
    # show: pd.Series(summary["models"])
    md_dir = out / "model"
    torch.save({"state_dict": model.state_dict(), "prep": pickle.dumps(d.prep), "d_seq": d.seq.shape[2],
                "d_static": d.static.shape[1], "outcomes": cohort.OUTCOMES, "emb_dim": cfg.emb_dim,
                "gru_hidden": cfg.gru_hidden, "static_hidden": cfg.static_hidden}, md_dir / "patient_encoder.pth")
    np.save(md_dir / "index_embeddings.npy", emb[tr])
    np.save(md_dir / "index_outcomes.npy", d.Y[tr])
    if index.faiss:
        import faiss
        faiss.write_index(index.ix, str(md_dir / "patients.faiss"))
    by_unit = []
    for u in np.unique(d.careunit[te]):
        m = d.careunit[te] == u
        if m.sum() >= 10:
            by_unit.append(dict(careunit=u, n=int(m.sum()), mortality=float(d.Y[te][m, 0].mean())))
    summary = dict(stays=len(d.Y), split=features.summary(d).to_dict("records"), encoder_best_epoch=st.best_epoch,
                   retrieval=res.to_dict("records"), ann=ann, demo=demo_out.to_dict(), demo_patient=DEMO_PATIENT,
                   test_units=suppress(pd.DataFrame(by_unit), ("n",)).to_dict("records"),
                   runtime_min=round((time.time() - t0) / 60, 1), config=cfg.to_dict(),
                   models={f.name: str(f) for f in sorted(md_dir.iterdir())})
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    from .report import build as build_report
    build_report(out, res, summary)
    store.put_dir(out / "report", f"runs/{cfg.run_name}/report")
    store.put_dir(md_dir, f"runs/{cfg.run_name}/model")
    log.info("DONE in %.1f min", summary["runtime_min"])
    return summary
