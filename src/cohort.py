"""MIMIC-IV ICU stays (adult, >= 24 h): first-24-h data as input, outcomes AFTER the first 24 h as targets.

Outcomes: in-hospital death, ICU stay > 7 days, invasive ventilation after 24 h, vasopressor after 24 h,
AKI stage >= 2 between 24 h and 7 days (KDIGO, mimic-code). All from mimic-code `derived` concepts.
"""
from __future__ import annotations

OUTCOMES = ["death", "long_icu_stay", "vent_after_24h", "vaso_after_24h", "aki2_day2_7"]
LABS = ["hemoglobin_min", "platelets_min", "wbc_max", "bicarbonate_min", "bun_max", "creatinine_max", "sodium_min",
        "potassium_max", "aniongap_max", "glucose_max", "inr_max", "bilirubin_total_max", "albumin_min"]
SOFA = ["sofa", "respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
VITALS = {"hr": "heart_rate", "sbp": "sbp", "mbp": "mbp", "rr": "resp_rate", "temp": "temperature", "spo2": "spo2",
          "glucose": "glucose"}


def _stays(cfg) -> str:
    return f"""
WITH s AS (
  SELECT d.subject_id, d.hadm_id, d.stay_id, d.icu_intime AS intime, d.admission_age, d.gender, i.first_careunit,
    a.admission_type, d.los_icu, d.hospital_expire_flag
  FROM {{derived}}.icustay_detail d
  JOIN {{icu}}.icustays i ON i.stay_id = d.stay_id
  JOIN {{hosp}}.admissions a ON a.hadm_id = d.hadm_id
  WHERE d.admission_age >= {cfg.min_age} AND d.los_icu >= {cfg.seq_hours / 24}
)"""


def cohort_sql(cfg) -> str:
    h = cfg.seq_hours
    return _stays(cfg) + f"""
, vent AS (
  SELECT s.stay_id, 1 AS x FROM s JOIN {{derived}}.ventilation v ON v.stay_id = s.stay_id
   AND v.ventilation_status = 'InvasiveVent' AND v.endtime > DATETIME_ADD(s.intime, INTERVAL {h} HOUR)
  GROUP BY 1
), vaso AS (
  SELECT s.stay_id, 1 AS x FROM s JOIN {{derived}}.vasoactive_agent v ON v.stay_id = s.stay_id
   AND v.endtime > DATETIME_ADD(s.intime, INTERVAL {h} HOUR)
  GROUP BY 1
), aki AS (
  SELECT s.stay_id, 1 AS x FROM s JOIN {{derived}}.kdigo_stages k ON k.stay_id = s.stay_id AND k.aki_stage >= 2
   AND k.charttime > DATETIME_ADD(s.intime, INTERVAL {h} HOUR) AND k.charttime <= DATETIME_ADD(s.intime, INTERVAL 7 DAY)
  GROUP BY 1
)
SELECT s.subject_id, s.stay_id, s.admission_age AS age, IF(s.gender = 'F', 1, 0) AS female, s.first_careunit,
  s.admission_type,
  {", ".join(f"f.{c}" for c in SOFA)},
  {", ".join(f"l.{c}" for c in LABS)},
  g.gcs_min, u.urineoutput, w.weight, ch.charlson_comorbidity_index AS charlson,
  s.hospital_expire_flag AS death, IF(s.los_icu > 7, 1, 0) AS long_icu_stay,
  IFNULL(vent.x, 0) AS vent_after_24h, IFNULL(vaso.x, 0) AS vaso_after_24h, IFNULL(aki.x, 0) AS aki2_day2_7
FROM s
LEFT JOIN {{derived}}.first_day_sofa f ON f.stay_id = s.stay_id
LEFT JOIN {{derived}}.first_day_lab l ON l.stay_id = s.stay_id
LEFT JOIN {{derived}}.first_day_gcs g ON g.stay_id = s.stay_id
LEFT JOIN {{derived}}.first_day_urine_output u ON u.stay_id = s.stay_id
LEFT JOIN {{derived}}.first_day_weight w ON w.stay_id = s.stay_id
LEFT JOIN {{derived}}.charlson ch ON ch.hadm_id = s.hadm_id
LEFT JOIN vent ON vent.stay_id = s.stay_id
LEFT JOIN vaso ON vaso.stay_id = s.stay_id
LEFT JOIN aki ON aki.stay_id = s.stay_id
"""


def hourly_sql(cfg) -> str:
    aggs = ",\n  ".join(f"AVG(v.{c}) AS {k}" for k, c in VITALS.items())
    return _stays(cfg) + f"""
SELECT s.stay_id, CAST(FLOOR(DATETIME_DIFF(v.charttime, s.intime, MINUTE) / 60) AS INT64) AS h,
  {aggs}
FROM s JOIN {{derived}}.vitalsign v ON v.stay_id = s.stay_id
 AND v.charttime >= s.intime AND v.charttime < DATETIME_ADD(s.intime, INTERVAL {cfg.seq_hours} HOUR)
GROUP BY 1, 2
"""
