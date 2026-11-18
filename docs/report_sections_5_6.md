# Draft: Report Sections 5 and 6 (Role C)

> **Status.** Every number below comes from files in `results/`, produced by `python -m src.hmm_viterbi.experiment`. The emission probabilities currently come from the distance-only stand-in classifier, not B's tuned model. Once B's `predict_proba` is plugged in, rerun the experiment and update the numbers. Rewrite this text in your own words before it goes into the PDF.

## 5. HMM and Viterbi route reconstruction

### 5.1 Hidden states
At GPS point *t*, the hidden states are that point's candidate arcs: up to k = 8 arcs within 50 m, found with a shapely STRtree in a per-trip metric projection. Each state also stores where the point projects onto the arc (its offset in metres from the arc's start). The decoder therefore keeps the candidate-to-arc link: the output for point *t* is always one of its own candidate `arc_id`s.

### 5.2 Emission probabilities
log E_t(j) = w · log p_t(j), where p_t(j) is the classifier probability of the (point *t*, arc *j*) row, floored at 1e-12, and w = 1.

### 5.3 Transition probabilities
log T(i→j) = −| d_route − d_gps | / β

- **d_gps** is the straight-line distance between GPS points *t−1* and *t*.
- **d_route** is the driving distance from point *t−1*'s position on arc *i* to point *t*'s position on arc *j*: the rest of arc *i*, plus a shortest path through the directed network (bounded Dijkstra, cached per source node), plus the offset on arc *j*.
- **Search limit:** 2 · d_gps + 100 m. Anything not reachable within it gets T = 0 (log = −∞).

The idea: a real vehicle's driving distance between two fixes is close to the straight-line distance. Big detours, wrong-direction arcs (one-way arcs) and disconnected roads are penalised.

**Same-arc jitter.** While a vehicle is stopped, GPS noise makes the projected position drift backwards along the arc. Backward movement of up to `same_arc_tol` metres on the same arc is treated as standing still rather than as an impossible U-turn. This turned out to be the most important decoder parameter (see 5.6).

### 5.4 Viterbi recurrence
S_t(j) = max_i [ S_{t−1}(i) + log T(i→j) ] + log E_t(j),  back_t(j) = argmax_i [...]

S_0 = log E_0. All work is in log space, and scores are re-centred every step to avoid underflow. The decoded sequence is recovered by following backpointers from argmax_j S_T(j). Decoding all 80 train trips takes 3.9 s in total (`total_decode_seconds` in `results/hmm_vs_classifier_summary.csv`).

### 5.5 Missing candidates, GPS gaps and disconnections
- **No candidates:** skipped by the HMM, then forward-filled from the previous decoded point (backward-filled at the start). With a nearest-arc fallback this never happened in train or student-test (`n_no_candidate` = 0).
- **Time gap over 60 s:** the chain is broken. Every state at *t* restarts from max S_{t−1} plus its own emission, because the driving distance across a long gap is unknown. This happened 69 times across the train trips.
- **All transitions unreachable** (GPS jump, network hole): same chain break. 2,043 breaks across train; 1,676 of them are in one trip (00000054, see 5.7).

### 5.6 Decoder tuning (train only)
A grid of 40 configurations was searched on the 80 train trips: β ∈ {2, 5, 10, 20, 50} m, stand-in σ ∈ {5, 10} m, same_arc_tol ∈ {5, 15, 30, 60} m. Results are in `results/decoder_tuning_train.csv`, and the selected config is in `artifacts/hmm_config.json`.

- **Best config:** β = 10 m, σ = 5 m, same_arc_tol = 30 m. Mean train point accuracy 0.898.
- **Effect of same_arc_tol:** best accuracy was 0.822 at 5 m and 0.898 at 30 m.
- **Effect of β and σ:** small once same_arc_tol ≥ 15 m.
- **Student-test data:** not used for selection.

### 5.7 Does HMM decoding help?
Source: `results/hmm_vs_classifier_summary.csv`, with means over trips.

| Partition | Method | Point acc. | Route continuity | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|
| train (80) | classifier only | 0.559 | 0.779 | 0.534 | 0.881 | 0.660 |
| train (80) | classifier + HMM | 0.898 | 0.966 | 0.959 | 0.964 | 0.961 |
| student_test (10) | classifier only | 0.551 | 0.804 | 0.534 | 0.906 | 0.667 |
| student_test (10) | classifier + HMM | 0.923 | 0.975 | 0.958 | 0.978 | 0.968 |

**Metric definitions**
- **Route continuity:** share of consecutive GPS points whose predicted arcs are the same arc or directly connected (end node of the first = start node of the second).
- **Route precision/recall/F1:** length-weighted overlap between the predicted arc set and the `.route` arc set.

**Why classifier-only is so weak.** Two-way streets are stored as two arcs, A→B and B→A, with identical geometry. A distance-only score cannot tell them apart, so the nearest arc is wrong about half the time. The HMM settles direction through connectivity: the wrong-direction arc can't be reached from the previous arc at a plausible distance. This also tells Person B that a heading feature is essential.

**Where the HMM did not help.**
- **Trip 00000010:** the only train trip where the HMM is worse (0.537 vs 0.543).
- **Bad data in trips 54 and 33.** These are the lowest-scoring trips:
  - Trip 00000054 (0.086): the whole GPS track is about 84 m (median) from any arc, and 90% of points have no correct candidate (`results/candidate_miss_train.csv`).
  - Trip 00000033 (0.236): GPS points lie about 7 m from roads, but about 122 m from the reference route. The supplied `.route` does not describe this track.

  Both trips show the reference labels are not always trustworthy (see Dataset Exploration §6).
- **Continuity still below 1.0.** At 1 Hz and high speed, a vehicle can cross a short arc between two fixes. The decoded sequence then "skips" that arc, which the continuity metric counts as a break even though the route is drivable.

### 5.8 Limitations
- **Labels:** point-level reference labels come from a monotone DP that aligns GPS points to `.route` (`labels.py`), so they inherit any error in `.route`.
- **Classifier:** the stand-in classifier ignores heading; B's model should raise classifier-only accuracy and narrow the gap.
- **Chain breaks:** a break restarts from the best previous state, so a single bad stretch cannot corrupt the rest of the route, but continuity is lost at the break.

## 6. Geographical visualisation (Folium)

- **Code:** `src/viz_folium/maps.py`
- **Maps:** `results/maps/trip_000000{09,52,65}.html` (student-test)

**Layers:**
- road network near the trip (grey)
- GPS observations (blue dots, subsampled to ≤ 3000)
- reference route (green, thick)
- classifier-only route (orange dashed, off by default)
- HMM route (red)
- HMM error points (black, off by default)
- start and end markers

The map has a legend and a layer control, and opens zoomed to the trip's extent. For the UI, call `trip_map(..., show_reference=False)` so ground truth never reaches the client.

> TODO: add a screenshot of one map here (e.g. trip 00000052) once the final model is plugged in.
