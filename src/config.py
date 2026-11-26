from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- data: MIMIC-IV ICU stays (>= 24 h), first 24 h only, full cohort ---
    hosp_dataset: str = ""
    icu_dataset: str = ""
    derived_dataset: str = ""
    min_age: int = 18
    seq_hours: int = 24
    split: tuple = (0.7, 0.15, 0.15)       # by patient: train = the searchable "past patients" database
    seed: int = 42

    # --- encoder (GRU + static -> embedding, multi-task heads), resumable ---
    emb_dim: int = 64
    gru_hidden: int = 64
    static_hidden: int = 128
    dropout: float = 0.2
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 40
    patience: int = 6

    # --- retrieval ---
    k: int = 50                            # neighbours used to estimate a query's outcome risk
    k_lift: int = 10
    ivf_nlist: int = 256                   # approximate FAISS index (speed comparison)
    ivf_pq_m: int = 16
    n_boot: int = 200

    # --- outputs + checkpoints (private GCS bucket, auto-created) ---
    work_dir: str = "/kaggle/working/similarity"
    cache_dir: str = "/tmp/similarity_cache"
    project_id: str = ""
    bucket: str = ""
    bucket_location: str = "US"
    gcs_prefix: str = "icu-patient-similarity-search"
    run_name: str = "main"
    local_store: str = ""
    ckpt_every_min: float = 5.0
    device: str = ""

    def datasets(self) -> dict:
        return {"hosp": (self.hosp_dataset, ["mimiciv_3_1_hosp", "mimiciv_hosp", "mimiciv_2_2_hosp"], "admissions"),
                "icu": (self.icu_dataset, ["mimiciv_3_1_icu", "mimiciv_icu", "mimiciv_2_2_icu"], "icustays"),
                "derived": (self.derived_dataset, ["mimiciv_3_1_derived", "mimiciv_derived", "mimiciv_2_2_derived"],
                            "icustay_detail")}

    def to_dict(self):
        return asdict(self)
