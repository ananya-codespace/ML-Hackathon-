# PathSnap API (Laptop 1)

The API runs the full pipeline for one trip:
1. candidates and features (Person A)
2. tuned classifier (Person B)
3. HMM/Viterbi (Person C)

It then scores the result against the reference route **on the server**. The reference route is never sent to the UI.

## Run (Laptop 1)

```bash
pip install -r requirements.txt          # pinned: scikit-learn must match artifacts/final_model.joblib
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- **Trip data:** the evaluator trips must be in `data/evaluator_test/<trip_id>/` (git-ignored). Each folder holds `<id>.track`, `.nodes`, `.arcs` and `.route`.
- **Rehearsing on other trips:** set `PATHSNAP_TRIPS_PARTITION=student_test` to serve `data/student_test/` instead.
- **Laptop 1's address:** on a Mac, `ipconfig getifaddr en0` (for example `172.20.10.4` on the phone hotspot). The UI on Laptop 2 must call `http://<that address>:8000`, not `127.0.0.1`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "healthy", ...}` |
| GET | `/trips` | `{"partition": "evaluator_test", "total_trips": 10, "trips": ["00000015", ...]}` |
| POST | `/predict/{trip_id}` | the prediction for one trip (below), or 404 if the trip folder doesn't exist |

## `POST /predict/{trip_id}` response

```json
{
  "trip_id": "00000015",
  "n_gps_points": 6854,
  "predicted_route": [1234, 1235, ...],
  "predicted_segments": [[[lat, lon], [lat, lon]], ...],
  "gps_points": [[lat, lon], ...],
  "gps_points_step": 4,
  "metrics": {
    "point_accuracy": 0.8665,
    "route_continuity": 0.9959,
    "route_precision": 0.98,
    "route_recall": 0.97,
    "route_f1": 0.97,
    "classifier_only_point_accuracy": 0.6027,
    "classifier_only_route_continuity": 0.91
  },
  "decoder": {"time_gap_breaks": 0, "unreachable_breaks": 0, "points_without_candidates": 0},
  "processing_seconds": 0.42,
  "is_mock": false
}
```

**Fields:**
- **`predicted_route`:** arc IDs in driving order, with consecutive repeats removed.
- **`predicted_segments`:** one `[[lat, lon], [lat, lon]]` line per entry of `predicted_route`. Draw each as a Folium `PolyLine`.
- **`gps_points`:** the raw GPS observations as `[lat, lon]`, subsampled to at most 2,000 points. `gps_points_step` says how many observations each returned point stands for.
- **`metrics`:** computed on the server against the reference route.
  - `point_accuracy`: share of GPS points matched to the correct arc.
  - `route_continuity`: share of consecutive predictions that are connected.
  - `route_f1`: length-weighted overlap with the reference route.
  - `classifier_only_*`: the same metrics without the HMM, for comparison.
- **Not in the response:** the reference route.

The fields `trip_id`, `predicted_route` and `is_mock` are unchanged from the mock version, so the existing UI keeps working.

## Server log

Each request is logged on Laptop 1 with the caller's IP. These lines are what the evaluator watches:

```
INFO - Incoming request from 172.20.10.2: POST /predict/00000015
INFO - Predicting trip 00000015 for 172.20.10.2
INFO - Trip 00000015 done in 0.42s: 6854 GPS points -> 338 route arcs | point accuracy 0.867 (classifier only 0.603) | continuity 0.996
```

## Tested (Laptop 1, over its network address)

All 10 evaluator trips return 200 in 0.06–0.42 s, and the metrics match `python -m src.evaluate_trips --partition evaluator_test`. Unknown trip IDs return 404.

## Note for the UI

`streamlit run src/ui/app.py` fails with `ModuleNotFoundError: No module named 'src'`, because Streamlit adds only `src/ui` to the import path. Add the repo root before the `src.ui.config` import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```
