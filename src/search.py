"""Similarity search with FAISS (exact inner product on L2-normalised embeddings = cosine), a torch fallback when FAISS
is missing, retrieval evaluation, an approximate-index speed test and a privacy-safe demo query."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .cohort import OUTCOMES, VITALS


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class Index:
    def __init__(self, base: np.ndarray, metric: str = "ip"):
        self.metric, self.base = metric, np.ascontiguousarray(base, dtype=np.float32)
        f = _faiss()
        self.faiss = f is not None
        if self.faiss:
            self.ix = (f.IndexFlatIP if metric == "ip" else f.IndexFlatL2)(self.base.shape[1])
            self.ix.add(self.base)

    def search(self, Q: np.ndarray, k: int):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if self.faiss:
            return self.ix.search(Q, k)
        b = torch.from_numpy(self.base)
        D, I = [], []
        for s in range(0, len(Q), 4096):
            q = torch.from_numpy(Q[s:s + 4096])
            sim = q @ b.T if self.metric == "ip" else -torch.cdist(q, b)
            v, i = sim.topk(k, dim=1)
            D.append((v if self.metric == "ip" else -v).numpy())
            I.append(i.numpy())
        return np.concatenate(D), np.concatenate(I)


def knn_eval(base, Yb, query, Yq, k, k_lift, metric) -> list[dict]:
    ix = Index(base, metric)
    _, I = ix.search(query, k)
    risk = Yb[I].mean(1)                                       # neighbour outcome rate = predicted risk
    rows = []
    for j, o in enumerate(OUTCOMES):
        y = Yq[:, j]
        if 0 < y.sum() < len(y):
            base_rate = Yb[:, j].mean()
            lift = Yb[I[y == 1, :k_lift], j].mean() / max(base_rate, 1e-9)
            rows.append(dict(outcome=o, auroc=roc_auc_score(y, risk[:, j]), lift_at_k=lift))
    return rows


def evaluate(d, emb, head_p, cfg) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    tr, te = d.idx("train"), d.idx("test")
    pca = PCA(n_components=min(cfg.emb_dim, d.raw.shape[1]), random_state=cfg.seed).fit(d.raw[tr])
    spaces = {"Learned embedding (cosine)": (emb[tr], emb[te], "ip"),
              "Raw features (Euclidean)": (d.raw[tr], d.raw[te], "l2"),
              f"PCA-{pca.n_components_} of raw features": (pca.transform(d.raw[tr]), pca.transform(d.raw[te]), "l2")}
    rows = []
    for name, (b, q, metric) in spaces.items():
        for r in knn_eval(b, d.Y[tr], q, d.Y[te], cfg.k, cfg.k_lift, metric):
            rows.append(dict(method=name, **r))
    for j, o in enumerate(OUTCOMES):                         # reference: the encoder's own prediction heads
        y = d.Y[te, j]
        if 0 < y.sum() < len(y):
            rows.append(dict(method="Encoder heads (direct prediction)", outcome=o, auroc=roc_auc_score(y, head_p[te, j]),
                             lift_at_k=np.nan))
    return pd.DataFrame(rows)


def ann_benchmark(base, query, cfg) -> dict:
    """Exact vs approximate (IVF-PQ) FAISS: recall@10 and milliseconds per query."""
    f = _faiss()
    if f is None or len(base) < cfg.ivf_nlist * 40:
        return {"note": "FAISS not installed or database too small for IVF-PQ; exact search used"}
    base, query = np.ascontiguousarray(base, np.float32), np.ascontiguousarray(query[:5000], np.float32)
    flat = f.IndexFlatIP(base.shape[1])
    flat.add(base)
    t = time.time()
    _, I0 = flat.search(query, 10)
    t_flat = (time.time() - t) / len(query) * 1000
    q = f.IndexFlatIP(base.shape[1])
    ivf = f.IndexIVFPQ(q, base.shape[1], cfg.ivf_nlist, cfg.ivf_pq_m, 8, f.METRIC_INNER_PRODUCT)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = 16
    t = time.time()
    _, I1 = ivf.search(query, 10)
    t_ivf = (time.time() - t) / len(query) * 1000
    rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I0, I1)])
    return {"database_size": len(base), "exact_ms_per_query": round(t_flat, 4), "ivfpq_ms_per_query": round(t_ivf, 4),
            "ivfpq_recall_at_10": round(float(rec), 4), "ivfpq_bytes_per_patient": cfg.ivf_pq_m}


@torch.no_grad()
def demo(profile: dict, model, prep, index: Index, Yb: np.ndarray, cfg, device, k=None) -> pd.Series:
    """Encode a HYPOTHETICAL patient (typed-in values; vitals held constant for 24 h) and summarise the outcomes of
    the k most similar past patients. Only aggregates over >= 10 patients are shown — no individual records."""
    import torch.nn.functional as F
    k = max(10, k or cfg.k)
    row = pd.DataFrame([{c: profile.get(c, np.nan) for c in profile}])
    for c in ("first_careunit", "admission_type"):
        row[c] = profile.get(c, "unknown")
    cube = {v: np.full((1, cfg.seq_hours), profile.get(v, np.nan), float) for v in VITALS}
    S, Q = prep.static(row), prep.sequence(cube)
    model.eval()
    z = F.normalize(model.embed(torch.tensor(Q, device=device), torch.tensor(S, device=device)), dim=-1).cpu().numpy()
    sim, I = index.search(z, k)
    out = {f"{o} among {k} most similar": float(Yb[I[0], j].mean()) for j, o in enumerate(OUTCOMES)}
    out["mean cosine similarity"] = float(sim[0].mean())
    return pd.Series(out)


def plot_knn(res: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = res.pivot(index="outcome", columns="method", values="auroc").reindex(OUTCOMES)
    ax = w.plot.bar(figsize=(9, 4.2), width=.8)
    ax.set(ylabel="AUROC (test queries)", xlabel="", title=f"Outcome of the k nearest past patients as a prediction",
           ylim=(0.5, 1.0))
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=.3, axis="y")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def plot_map(emb_test, y_death, path):
    """2-D PCA of the test embeddings; each hexagon = >= 10 patients, colour = in-hospital mortality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    xy = PCA(2, random_state=0).fit_transform(emb_test)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    hb = ax.hexbin(xy[:, 0], xy[:, 1], C=y_death, reduce_C_function=np.mean, gridsize=30, mincnt=10, cmap="viridis")
    fig.colorbar(hb, label="in-hospital mortality")
    ax.set(xlabel="PC 1", ylabel="PC 2", title="Embedding space (test patients, hexagons ≥ 10)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
