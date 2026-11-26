"""End-to-end on synthetic MIMIC-IV + mimic-code-shaped tables with a planted severity signal: real SQL on DuckDB,
outcome windows, encoder with resume, FAISS / torch search, evaluation, demo, report."""
import duckdb
import numpy as np
import pandas as pd

from src import cohort, search
from src.bq import DuckDBSource
from src.config import Config
from src.store import LocalStore


def _tables(n=2000, seed=0):
    r = np.random.default_rng(seed)
    T0 = pd.Timestamp("2150-01-01")
    sid, hid, sub = np.arange(n) + 30_000_000, np.arange(n) + 20_000_000, np.arange(n) + 10_000_000
    intime = pd.to_datetime([T0 + pd.Timedelta(days=int(i)) for i in range(n)])
    sev = r.normal(size=n)
    los = np.where(np.arange(n) % 25 == 0, 0.5, r.gamma(2, 2, n) + 1 + np.clip(sev, 0, None) * 3)
    death = (r.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.6 * sev)))).astype(int)
    det = pd.DataFrame({"subject_id": sub, "hadm_id": hid, "stay_id": sid, "icu_intime": intime,
                        "admission_age": r.integers(18, 95, n), "gender": r.choice(["F", "M"], n), "los_icu": los,
                        "hospital_expire_flag": death})
    vent, vaso, kd, vs = [], [], [], []
    for i in range(n):
        if sev[i] > 0.3:
            vent.append((sid[i], intime[i], intime[i] + pd.Timedelta(hours=48), "InvasiveVent"))
        elif i % 7 == 0:
            vent.append((sid[i], intime[i], intime[i] + pd.Timedelta(hours=10), "InvasiveVent"))   # ends < 24 h -> 0
        if sev[i] > 0.6:
            vaso.append((sid[i], intime[i] + pd.Timedelta(hours=20), intime[i] + pd.Timedelta(hours=40)))
        if sev[i] > 0.8:
            kd.append((sid[i], intime[i] + pd.Timedelta(hours=60), 2))
        if i % 11 == 0:
            kd.append((sid[i], intime[i] + pd.Timedelta(hours=10), 3))                             # before 24 h -> 0
        for h in range(0, 24, 2):
            vs.append((sid[i], intime[i] + pd.Timedelta(hours=h + .5), 85 + 12 * sev[i] + r.normal(0, 5),
                       120 - 10 * sev[i], 80 - 8 * sev[i] + r.normal(0, 4), 18 + 4 * sev[i], 37 + .3 * sev[i],
                       97 - sev[i], 130.0))
        vs.append((sid[i], intime[i] + pd.Timedelta(hours=30), 300, 300, 300, 300, 300, 300, 300))          # after 24 h
    D = pd.DataFrame
    derived = dict(
        icustay_detail=det,
        ventilation=D(vent, columns=["stay_id", "starttime", "endtime", "ventilation_status"]),
        vasoactive_agent=D(vaso, columns=["stay_id", "starttime", "endtime"]),
        kdigo_stages=D(kd, columns=["stay_id", "charttime", "aki_stage"]),
        first_day_sofa=D({"stay_id": sid, **{c: np.clip(np.round(4 + 2 * sev + r.normal(0, 1, n)), 0, 24)
                                             for c in cohort.SOFA}}),
        first_day_lab=D({"stay_id": sid, **{c: np.where(r.random(n) < .1, np.nan, r.lognormal(1 + .2 * sev, .3))
                                            for c in cohort.LABS}}),
        first_day_gcs=D({"stay_id": sid, "gcs_min": np.clip(np.round(14 - 2 * sev), 3, 15)}),
        first_day_urine_output=D({"stay_id": sid, "urineoutput": r.gamma(3, 500, n)}),
        first_day_weight=D({"stay_id": sid, "weight": r.normal(80, 15, n)}),
        charlson=D({"hadm_id": hid, "charlson_comorbidity_index": r.integers(0, 10, n)}),
        vitalsign=D(vs, columns=["stay_id", "charttime", "heart_rate", "sbp", "mbp", "resp_rate", "temperature", "spo2",
                                 "glucose"]))
    icu = dict(icustays=D({"stay_id": sid, "first_careunit": r.choice(["Medical Intensive Care Unit (MICU)",
                                                                      "Surgical Intensive Care Unit (SICU)"], n)}))
    hosp = dict(admissions=D({"hadm_id": hid, "admission_type": r.choice(["EW EMER.", "ELECTIVE"], n)}))
    return {"derived": derived, "icu": icu, "hosp": hosp}, sev


def _source(store):
    con = duckdb.connect()
    tabs, _ = _tables()
    for schema, ts in tabs.items():
        con.execute(f"CREATE SCHEMA {schema}")
        for name, df in ts.items():
            con.register("tmp_df", df)
            con.execute(f"CREATE TABLE {schema}.{name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    return DuckDBSource(con, {"derived": "derived", "icu": "icu", "hosp": "hosp"}, store)


def _cfg(tmp, work="o", **kw):
    base = dict(work_dir=str(tmp / work), cache_dir=str(tmp / f"cache_{work}"), local_store=str(tmp / "bucket"),
                epochs=6, patience=3, batch_size=128, k=20, n_boot=10, device="cpu")
    base.update(kw)
    return Config(**base)


def test_sql_outcome_windows(tmp_path):
    cfg = _cfg(tmp_path)
    src = _source(LocalStore(cfg.local_store, cfg.gcs_prefix))
    c = src.df(cohort.cohort_sql(cfg)).set_index("stay_id")
    _, sev = _tables()
    ids = c.index - 30_000_000
    assert not any(i % 25 == 0 for i in ids)                                   # ICU stay < 24 h excluded
    assert (c.vent_after_24h.to_numpy() == (sev[ids] > 0.3)).all()            # vent ending before 24 h ignored
    assert (c.vaso_after_24h.to_numpy() == (sev[ids] > 0.6)).all()
    assert (c.aki2_day2_7.to_numpy() == (sev[ids] > 0.8)).all()               # AKI before 24 h ignored
    v = src.df(cohort.hourly_sql(cfg))
    assert v.h.between(0, 23).all() and v.hr.max() < 300                      # data after 24 h never used


def test_index_backends_agree():
    r = np.random.default_rng(0)
    B = r.normal(size=(500, 8)).astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    Q = B[:20] + 0.01
    ix = search.Index(B, "ip")
    _, I = ix.search(Q, 5)
    assert (I[:, 0] == np.arange(20)).all()
    ix.faiss = False                                                            # torch fallback gives the same answer
    _, I2 = ix.search(Q, 5)
    assert (I[:, 0] == I2[:, 0]).all()


def test_pipeline_resume_and_demo(tmp_path):
    from src.pipeline import run
    cfg = _cfg(tmp_path)
    s = run(cfg, source=_source(LocalStore(cfg.local_store, cfg.gcs_prefix)))
    r = pd.DataFrame(s["retrieval"])
    emb = r[(r.method == "Learned embedding (cosine)") & (r.outcome == "death")].auroc.iloc[0]
    assert emb > 0.7                                                           # neighbours carry the planted signal
    assert all(0 <= v <= 1 for k, v in s["demo"].items() if "among" in k)
    assert "Demo query" in (tmp_path / "o" / "report" / "REPORT.md").read_text()
    s2 = run(_cfg(tmp_path, "o2"), source=_source(LocalStore(cfg.local_store, cfg.gcs_prefix)))
    r2 = pd.DataFrame(s2["retrieval"])
    assert np.allclose(r.auroc, r2.auroc)


def test_notebook_steps(tmp_path):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from nbsteps import exec_steps
    cfg = _cfg(tmp_path, epochs=2)
    ns = exec_steps(root / "src" / "pipeline.py", cfg, source=_source(LocalStore(cfg.local_store, cfg.gcs_prefix)))
    assert ns["result"]["stays"] > 1500
