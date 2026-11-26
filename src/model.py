"""Patient encoder: GRU over the first-24-h vitals + MLP over the static table -> 64-d embedding (L2-normalised for
search). Multi-task heads on the embedding force it to keep what matters for the outcomes. Resumable training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .trainer import Checkpointer, TensorBatches, fit


class Encoder(nn.Module):
    def __init__(self, d_seq, d_static, n_out, emb=64, gru=64, static_hidden=128, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(d_seq, gru, batch_first=True)
        self.static = nn.Sequential(nn.Linear(d_static, static_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj = nn.Sequential(nn.Linear(gru + static_hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, emb))
        self.heads = nn.Linear(emb, n_out)

    def embed(self, seq, static):
        return self.proj(torch.cat([self.gru(seq)[1][-1], self.static(static)], -1))

    def forward(self, seq, static):
        return self.heads(self.embed(seq, static))


@torch.no_grad()
def encode(model, d, idx, device, bs=8192):
    """-> (L2-normalised embeddings, head probabilities)"""
    model.eval()
    E, P = [], []
    for s in range(0, len(idx), bs):
        i = idx[s:s + bs]
        z = model.embed(torch.tensor(d.seq[i], device=device), torch.tensor(d.static[i], device=device))
        E.append(F.normalize(z, dim=-1).cpu().numpy())
        P.append(torch.sigmoid(model.heads(z)).cpu().numpy())
    return (np.concatenate(E).astype(np.float32), np.concatenate(P)) if E else (np.zeros((0, 1)), np.zeros((0, 1)))


def train(cfg, d, device, store, key):
    torch.manual_seed(cfg.seed)
    model = Encoder(d.seq.shape[2], d.static.shape[1], d.Y.shape[1], cfg.emb_dim, cfg.gru_hidden, cfg.static_hidden,
                    cfg.dropout).to(device)
    tr, va = d.idx("train"), d.idx("val")
    pos = d.Y[tr].mean(0).clip(1e-3, 1 - 1e-3)
    pw = torch.tensor(np.sqrt((1 - pos) / pos), device=device)          # mild re-weighting of rare outcomes
    batches = TensorBatches({"seq": d.seq[tr], "static": d.static[tr], "y": d.Y[tr]}, cfg.batch_size, cfg.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def loss_fn(m, b):
        return F.binary_cross_entropy_with_logits(m(b["seq"], b["static"]), b["y"], pos_weight=pw)

    def evaluate(m):
        _, p = encode(m, d, va, device)
        aucs = [roc_auc_score(d.Y[va, j], p[:, j]) for j in range(d.Y.shape[1]) if 0 < d.Y[va, j].sum() < len(va)]
        return {"val_mean_auroc": float(np.mean(aucs)) if aucs else float("nan")}
    return fit(model, opt, epochs=cfg.epochs, batches=batches, loss_fn=loss_fn, evaluate=evaluate,
               ckpt=Checkpointer(store, key, f"{cfg.work_dir}/ckpt/encoder"), metric="val_mean_auroc",
               patience=cfg.patience, ckpt_every_min=cfg.ckpt_every_min, log_every=0)
