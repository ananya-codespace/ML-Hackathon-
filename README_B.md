# Feature table: spec for Person B

Built by `src/features/make_tables.py` -> `data/processed/` (git-ignored; regenerate with the command at the bottom).

| File | Rows | Use |
|---|---|---|
| `train_features.parquet` | 1,452,223 (80 trips) | training and grouped CV |
| `student_test_features.parquet` | 149,946 (10 trips) | development evaluation |
| `train_points.parquet`, `student_test_points.parquet` | one row per GPS point | observation-level Top-1 |

One row = one (GPS point, candidate arc). Candidates: arcs within **75 m**, at most **10 nearest** per point.

- **Keys:** `trip_id`, `obs_idx` (GPS point index in the trip), `arc_id`.
- **Target:** `y` (1 if the arc is the point's label; label = nearest arc of the reference route). About 13% positives, so use class weights or similar and document it.
- **Fold:** `fold` (0-4, trip-level, train only; also in `results/cv_folds.csv`). Use it for GroupKFold. Never split by row.
- **Flag:** `trip_flagged` marks trips 33 and 54 (GPS track shifted ~115-137 m from the reference route). Report metrics with and without them.

## Model inputs (14 features, `FEATURES` in `src/features/build_features.py`)
`dist, rank, dist_min, dist_gap, n_cand, t_along, arc_len, cos_head, head_ok, speed, dt_prev, dt_next, dist_prev, dist_next`

- `dist`: metres from the point to the arc. `rank`: 0 = nearest candidate. `dist_gap`: dist minus the best candidate's dist.
- `t_along`: position of the projection along the arc (0-1). `arc_len`: metres.
- `cos_head`: cosine of the angle between movement direction and arc direction (0 when `head_ok` = 0).
- `speed` (m/s), `dt_prev`/`dt_next` (s), `dist_prev`/`dist_next`: distance of the neighbouring GPS points to the same arc.

**Scaling:** `artifacts/scaler.joblib` (fitted on train only). Tree models don't need it.
**Never use as inputs:** `y`, `fold`, `trip_flagged`, `trip_id`, `obs_idx`, `arc_id`.
**Points tables:** `label_arc_id`, `label_in_cands`, `n_cands` for Top-1 (points whose label is not in the candidates count as misses).

Regenerate: `python src/features/make_tables.py` (~3 min; needs `data/train` and `data/student_test`).
