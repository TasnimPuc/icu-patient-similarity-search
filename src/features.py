"""Static first-day table + hourly vital-sign sequence -> model inputs. `Prep` keeps the training statistics so that a
new (or hypothetical) patient can be encoded the same way (used by the demo query)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cohort import LABS, OUTCOMES, SOFA, VITALS

log = logging.getLogger(__name__)
RANGES = {"hr": (20, 250), "sbp": (40, 260), "mbp": (20, 200), "rr": (2, 70), "temp": (30, 43), "spo2": (50, 100),
          "glucose": (20, 1000)}
NUM = ["age", "female", "gcs_min", "urineoutput", "weight", "charlson"] + SOFA + LABS
SKEW = ["urineoutput", "bun_max", "creatinine_max", "wbc_max", "inr_max", "bilirubin_total_max", "glucose_max"]


class Prep:
    def fit(self, coh: pd.DataFrame, cube: dict):
        n = self._num(coh)
        self.med = n.median()
        self.miss_cols = list(n.columns[n.isna().mean() > 0.05])
        z = n.fillna(self.med)
        self.mu, self.sd = z.mean(), z.std().replace(0, 1).fillna(1)
        self.cats = {c: coh[c].fillna("unknown").value_counts().index[:10].tolist()
                     for c in ("first_careunit", "admission_type")}
        self.seq = {}
        for k in VITALS:
            v = cube[k][~np.isnan(cube[k])]
            self.seq[k] = (float(np.median(v)) if len(v) else 0.0, float(v.mean()) if len(v) else 0.0,
                           float(v.std()) if len(v) > 1 and v.std() > 0 else 1.0)
        return self

    @staticmethod
    def _num(coh):
        n = coh.reindex(columns=NUM).apply(pd.to_numeric, errors="coerce").astype(float)
        for c in SKEW:
            n[c] = np.log1p(n[c].clip(lower=0))
        return n

    def static(self, coh: pd.DataFrame) -> np.ndarray:
        n = self._num(coh)
        flags = n[self.miss_cols].isna().astype(float).add_suffix("_missing")
        z = ((n.fillna(self.med) - self.mu) / self.sd).clip(-6, 6)
        oh = [pd.get_dummies(pd.Categorical(coh[c].fillna("unknown").where(coh[c].isin(v), "other"),
                                            categories=v + ["other"]), prefix=c, dtype=float)
              for c, v in self.cats.items()]
        return pd.concat([z.reset_index(drop=True), flags.reset_index(drop=True)] +
                         [o.reset_index(drop=True) for o in oh], axis=1).to_numpy(np.float32)

    def sequence(self, cube: dict) -> np.ndarray:
        vals, mask = [], []
        for k in VITALS:
            a = cube[k]
            m = ~np.isnan(a)
            f = pd.DataFrame(a).ffill(axis=1).bfill(axis=1).to_numpy()
            med, mu, sd = self.seq[k]
            f = np.where(np.isnan(f), med, f)
            vals.append(np.clip((f - mu) / sd, -6, 6))
            mask.append(m.astype(float))
        return np.stack(vals + mask, -1).astype(np.float32)


def cube_from_long(long: pd.DataFrame, stays: np.ndarray, hours: int) -> dict:
    pos = pd.Series(np.arange(len(stays)), index=stays)
    long = long[long.stay_id.isin(pos.index) & long.h.between(0, hours - 1)]
    i, t = pos[long.stay_id].to_numpy(), long.h.to_numpy().astype(int)
    out = {}
    for k in VITALS:
        a = np.full((len(stays), hours), np.nan)
        v = pd.to_numeric(long[k], errors="coerce").to_numpy(float)
        lo, hi = RANGES[k]
        a[i, t] = np.where((v >= lo) & (v <= hi), v, np.nan)
        out[k] = a
    return out


@dataclass
class Data:
    static: np.ndarray
    seq: np.ndarray
    Y: np.ndarray                   # [n, len(OUTCOMES)]
    part: np.ndarray
    raw: np.ndarray                 # baseline feature space: static + per-vital 24-h mean/min/max (standardised)
    careunit: np.ndarray
    prep: Prep

    def idx(self, p):
        return np.flatnonzero(self.part == p)


def build(coh: pd.DataFrame, hourly: pd.DataFrame, cfg) -> Data:
    coh = coh.drop_duplicates("stay_id").reset_index(drop=True)
    stays = coh.stay_id.to_numpy()
    cube = cube_from_long(hourly, stays, cfg.seq_hours)
    u = (pd.util.hash_pandas_object(coh.subject_id.astype(str) + f"|{cfg.seed}", index=False).to_numpy() % 10_000) / 10_000
    a, b = cfg.split[0], cfg.split[0] + cfg.split[1]
    part = np.where(u < a, "train", np.where(u < b, "val", "test"))
    tr = part == "train"
    prep = Prep().fit(coh[tr], {k: v[tr] for k, v in cube.items()})
    S, Q = prep.static(coh), prep.sequence(cube)
    nv = len(VITALS)
    agg = np.concatenate([Q[:, :, :nv].mean(1), Q[:, :, :nv].min(1), Q[:, :, :nv].max(1)], 1)
    raw = np.concatenate([S, agg], 1)
    raw = (raw - raw[tr].mean(0)) / (raw[tr].std(0) + 1e-6)
    Y = coh[OUTCOMES].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).to_numpy(np.float32)
    log.info("stays %d | static %d cols | seq %s | outcome rates %s", len(coh), S.shape[1], Q.shape,
             dict(zip(OUTCOMES, Y.mean(0).round(3))))
    return Data(static=S, seq=Q, Y=Y, part=part, raw=raw.astype(np.float32),
                careunit=coh.first_careunit.fillna("unknown").to_numpy(), prep=prep)


def summary(d: Data) -> pd.DataFrame:
    rows = []
    for p in ("train", "val", "test"):
        m = d.part == p
        rows.append({"part": p, "stays": int(m.sum()), **{o: float(d.Y[m, j].mean()) for j, o in enumerate(OUTCOMES)}})
    return pd.DataFrame(rows)
