# HMM / Viterbi route reconstruction (Role C)

Turns per-point candidate probabilities into one connected route: one arc per GPS point.

## Run

```bash
python -m src.hmm_viterbi.experiment          # tune decoder on train, compare on train + student_test
python -m src.hmm_viterbi.plots               # figures -> results/figures/
python -m src.viz_folium.maps --trip data/student_test/00000052   # map -> results/maps/
```

`data/train` and `data/student_test` are the unzipped trip folders (git-ignored).

When Person B's probability files exist (`results/final_model_probs_train_oof.csv` and `results/final_model_probs_student_test.csv`), decode with the real model instead of the stand-in:

```bash
python -m src.hmm_viterbi.experiment --probs-source b    # results -> results/hmm_bmodel/
```

The decoder is tuned on B's out-of-fold train probabilities only. Labels come from the `label` column of B's table, so Top-1 matches B's numbers. This also rewrites `artifacts/hmm_config.json`, which the API should use.

`route_continuity` also accepts B's call form, `route_continuity(pred_df, "student_test")`.

## Interface for the other roles

```python
from src.hmm_viterbi import load_trip, RoadNetwork, decode, HMMParams, route_continuity

trip = load_trip("data/student_test/00000052", with_route=False)  # prediction path: no labels
net  = RoadNetwork.from_trip(trip)
pred, info = decode(cands, probs, trip, net, HMMParams(**cfg["hmm"]))  # one arc_id per GPS point
route_continuity(pred, net)   # B: use this for the comparison table
```

- `cands`: DataFrame with at least `obs_idx` and `arc_id` (one row per point–candidate pair). `offset` (metres along the arc) is used if present, otherwise computed.
- `probs`: classifier probability per row, in the same row order as `cands`.
- `artifacts/hmm_config.json`: frozen decoder parameters for the API.

Until B's model is ready, `candidates.build_candidates` and `candidates.distance_probs` stand in. The stand-in classifier is a Gaussian on point-to-arc distance, so its top-1 is simply the nearest arc.

## Modules

| File | Purpose |
|---|---|
| `network.py` | Loads a trip, projects each trip to metres (local azimuthal equidistant), builds the directed arc graph, and runs bounded and cached Dijkstra |
| `candidates.py` | k nearest arcs within a radius (STRtree), plus the stand-in classifier |
| `labels.py` | Assigns a reference arc to each GPS point using a monotone DP alignment against `.route` (evaluation only) |
| `viterbi.py` | HMM and Viterbi decoder |
| `metrics.py` | Route continuity, point accuracy, route precision/recall/F1, candidate miss rate |
| `experiment.py` | Decoder tuning on train, then classifier-only vs. classifier + HMM |
| `plots.py` | Report figures |
