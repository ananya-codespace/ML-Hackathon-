# Report Sections 5 and 6 (Role C)

All numbers come from files in `results/hmm_bmodel/` and can be regenerated with:

```bash
python -m src.models.run_bayes --model hist_gb --n-trials 20 --n-init 6   # tuned classifier + its probabilities
python -m src.hmm_viterbi.experiment --probs-source b                     # decoder tuning + HMM results
python -m src.hmm_viterbi.plots --results results/hmm_bmodel              # figures
```

---

## 5. HMM and Viterbi Route Reconstruction

### Overview

The tuned HistGradientBoosting classifier scores each (GPS point, candidate arc) row on its own. Choosing the highest-scoring arc at every point still produces routes a vehicle could not drive: the prediction flips to the opposite direction of a two-way street for a few seconds, or jumps onto a ramp or a road crossing underneath at an interchange (Figure 6.2).

We add a hidden Markov model over the candidate arcs and decode it with the Viterbi algorithm, so the chosen arcs must also be consistent with the road network and the timing between GPS points.

The HMM is the model: it defines how likely any sequence of arcs is. Viterbi is the algorithm that finds the most likely sequence without enumerating all of them.

Code: `src/hmm_viterbi/` (`viterbi.py` for the decoder, `network.py` for the road graph and shortest paths).

### 5.1 What are the hidden states in your implementation?

At GPS observation *t*, the hidden states are that observation's candidate arcs from Person A's feature table: up to 10 arcs within 75 m.

- **What a state carries:** each state is an `arc_id` plus the observation's projected offset along that arc in metres. The offset is used to compute driving distances in 5.3.
- **Projection:** each trip is projected to metres with its own local projection.
- **Candidate-to-arc link:** the decoder only ever outputs one of an observation's own candidate `arc_id`s, so the link between candidates and road segments is preserved.
- **Candidate miss rate:** the share of points whose reference arc is not among their candidates is 0.77% on train (78 trips) and 0.14% on student-test (`results/hmm_bmodel/candidate_miss_*.csv`). No decoder can recover these points.

### 5.2 How are emission probabilities calculated?

log E_t(j) = w · log p_t(j)

- **p_t(j):** the tuned classifier's probability that arc *j* is the correct arc for observation *t*. This is the `prob` column of `results/final_model_probs_*.csv`.
- **Floor:** probabilities are floored at 10⁻¹² so that a single confident zero can't remove a candidate completely.
- **Weight:** w = 1.
- **No leakage into the decoder settings:** for decoder tuning we use the classifier's out-of-fold probabilities on the training trips. Each trip is scored by a model that never saw it, so tuning the decoder does not leak training labels.

### 5.3 How are transition probabilities calculated?

log T(i → j) = −| d_route − d_gps | / β

- **d_gps** is the straight-line distance between GPS observations *t−1* and *t*.
- **d_route** is the driving distance on the directed road network from observation *t−1*'s position on arc *i* to observation *t*'s position on arc *j*. It is:
  - the remainder of arc *i*,
  - plus the shortest path from the end node of *i* to the start node of *j* (Dijkstra),
  - plus the offset on arc *j*.
- **Search limit:** the search stops at 2 · d_gps + 100 m, and results are cached per source node. Anything not reachable within that limit gets T = 0 (log −∞).

**Why this form:** between two fixes one second apart, a real vehicle drives about as far as the straight-line distance. A wrong-direction arc forces a long detour through the one-way graph, and a ramp or underpass is not reachable from the main road in a few metres. Both therefore get a large penalty. Selected value: β = 2 m (5.6).

**Same-arc jitter rule.** While a vehicle is stopped, GPS noise can make the projected offset move backwards along the arc. A strict model would read this as a U-turn, which is unreachable. Backward movement of up to `same_arc_tol` = 60 m on the same arc is therefore treated as standing still. This was the most influential decoder setting (5.6).

**Viterbi recurrence.** We work in log space, with backpointers:

S₀(j) = log E₀(j)
S_t(j) = max_i [ S_{t−1}(i) + log T(i → j) ] + log E_t(j),  B_t(j) = argmax_i [ … ]

The route is read by backtracking from argmax_j S_T(j). Scores are re-centred at every step to avoid underflow.

**Cost:** with at most 10 candidates, each step compares at most 100 pairs. Decoding all 183,260 training observations took 5.0 s in total; the 19,160 student-test observations took 0.6 s.

### 5.4 How are missing candidates or large GPS gaps handled?

- **No candidates at an observation:** this happens when nothing lies within 75 m, and the feature table then has no rows for that point. The observation is skipped by the HMM. Afterwards it takes the arc decoded for the previous observation (the next one, at the start of a trip), so every point still gets a prediction. There were 715 such points in train (0.39%) and none in student-test.
- **Time gap over 60 s:** the chain is broken. After a long gap the driving distance can't be estimated from d_gps, so every state at *t* restarts from the best previous score plus its own emission: S_t(j) = max_i S_{t−1}(i) + log E_t(j). There were 70 such breaks in train and 1 in student-test. 60 s was chosen because about 97% of steps are 1 s apart, and longer gaps in the EDA were mostly parked vehicles or signal loss.

### 5.5 What happens when consecutive GPS observations suggest disconnected road segments?

If no candidate at *t* can be reached from any candidate at *t−1* within the search limit (every log T is −∞), we use the same chain break as for a gap: restart from the best previous state plus the new emissions.

**Why restart instead of forcing a jump:** a single bad stretch can't then wipe out the probability of the whole route, and a short wrong section can't force a long detour into the decoded route.

**How often it happened:** 91 times in train and 6 times in student-test. Causes include GPS jumps, fragmented road networks (a median of 20 components per trip) and map gaps.

### 5.6 Did HMM decoding improve the final route?

Yes. The decoder parameters were chosen on the 78 training trips only, using out-of-fold probabilities; student-test was not used for any choice.

**Decoder tuning** (`results/hmm_bmodel/decoder_tuning_train.csv`, `results/hmm_bmodel/figures/decoder_tuning_train.png`). We searched a grid of 20 configurations: β ∈ {2, 5, 10, 20, 50} m and same_arc_tol ∈ {5, 15, 30, 60} m. The selection criterion was mean point accuracy over the training trips.

| same_arc_tol | Best mean train point accuracy |
|---|---|
| 5 m | 0.821 |
| 15 m | 0.888 |
| 30 m | 0.899 |
| **60 m (selected)** | **0.899** |

| β | Best mean train point accuracy |
|---|---|
| **2 m (selected)** | **0.899** |
| 5 m | 0.899 |
| 10 m | 0.897 |
| 20 m | 0.897 |
| 50 m | 0.893 |

- **Selected configuration:** β = 2 m, same_arc_tol = 60 m, stored in `artifacts/hmm_config.json`.
- **Jitter tolerance mattered most:** it was worth 7.8 points. β mattered much less.
- **A small β works best:** the classifier's probabilities are already good, so the decoder works best when it strongly penalises any mismatch between driving and straight-line distance.

**Classifier-only vs classifier + HMM** (`results/hmm_bmodel/hmm_vs_classifier_summary.csv`):

| Partition | Method | Point accuracy (trip mean) | Point accuracy (pooled) | Route continuity | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|---|
| Train (78, out-of-fold) | Classifier only | 0.850 | 0.838 | 0.908 | 0.803 | 0.959 | 0.866 |
| Train (78, out-of-fold) | Classifier + HMM | **0.899** | **0.906** | **0.969** | **0.971** | **0.964** | **0.967** |
| Student-test (10) | Classifier only | 0.863 | 0.841 | 0.924 | 0.847 | 0.977 | 0.905 |
| Student-test (10) | Classifier + HMM | **0.916** | **0.889** | **0.977** | **0.980** | **0.980** | **0.980** |

How the metrics are defined (`src/hmm_viterbi/metrics.py`):
- **Point accuracy:** the share of observations whose predicted arc equals the reference arc. The reference arc is Person A's label: the nearest arc of the reference route.
  - *Trip mean* averages the per-trip accuracies.
  - *Pooled* counts every point once, so long trips weigh more.
- **Route continuity:** the share of consecutive observations whose predicted arcs are the same arc, or where the second starts at the first one's end node.
- **Route precision, recall and F1:** length-weighted overlap between the set of predicted arcs and the set of `.route` arcs (the sequence-level measures).

**What the HMM changes most** is precision and continuity, not recall. Recall was already 0.977 for the classifier alone, because the right arcs were almost always among its picks. The classifier also scattered single points onto wrong arcs (precision 0.847), and the HMM removes most of them (precision 0.980).

**Per student-test trip** (`results/hmm_bmodel/hmm_vs_classifier_student_test.csv`). The HMM improved point accuracy on 9 of 10 trips, and continuity and route F1 on all 10:

| Trip | Points | Accuracy: classifier only | Accuracy: + HMM | Continuity: classifier only | Continuity: + HMM | Route F1: classifier only | Route F1: + HMM |
|---|---|---|---|---|---|---|---|
| 00000000 | 2,503 | 0.940 | 0.980 | 0.963 | 0.994 | 0.969 | 0.997 |
| 00000007 | 1,354 | 0.906 | 1.000 | 0.996 | 1.000 | 0.984 | 0.997 |
| 00000009 | 1,049 | 0.867 | 0.905 | 0.880 | 0.943 | 0.873 | 0.933 |
| 00000040 | 2,995 | 0.920 | 0.951 | 0.962 | 0.996 | 0.918 | 0.997 |
| 00000052 | 262 | 0.950 | 0.996 | 0.950 | 0.966 | 0.950 | 0.991 |
| 00000056 | 3,958 | 0.821 | 0.820 | 0.892 | 0.967 | 0.913 | 0.971 |
| 00000065 | 768 | 0.797 | 0.833 | 0.920 | 0.983 | 0.892 | 0.986 |
| 00000076 | 1,486 | 0.859 | 0.952 | 0.876 | 0.966 | 0.836 | 0.969 |
| 00000077 | 3,911 | 0.693 | 0.780 | 0.920 | 0.991 | 0.788 | 0.982 |
| 00000091 | 874 | 0.871 | 0.941 | 0.880 | 0.968 | 0.924 | 0.981 |

Per-trip figures: `results/hmm_bmodel/figures/point_accuracy_*.png`, `route_continuity_*.png`, `route_f1_*.png`.

**Cases where HMM decoding did not help:**
- **Student-test trip 00000056** (0.820 vs 0.821): the HMM fixed 4.5% of points and broke 4.6%. 84% of its remaining errors are on an arc that *is* part of the reference route, just not the one the label assigns to that point. This trip drives some roads more than once. The label rule "nearest reference-route arc" ignores the order of the route, so a point on the second pass can be labelled with the arc from the first pass, which may be the opposite direction. The HMM follows the drivable order and is marked wrong there. The same effect explains much of the error on 00000065 and 00000077, where 19% of points get a different label under the two rules.
- **Train:** the HMM was worse on 8 of 78 trips.
  - The worst is 00000081 (0.777 vs 0.865). This trip has 16 time-gap breaks and 57 points without candidates. Restarting after each gap lets a wrong-direction choice persist until the next connected evidence: the HMM broke 10.1% of points and fixed 1.3%.
  - 00000016 (0.522 vs 0.533) was already hard for the classifier. 83% of its remaining errors are on an unconnected road about 39 m away (median), a parallel road that neither the emission nor the transition can separate.
  - 00000061 (0.613 vs 0.622) is different: 60% of its remaining errors are wrong-direction errors, the same pattern as the gap-heavy trip above.
- **Misaligned trips:** 00000033 and 00000054, whose GPS tracks are offset from the reference route, are excluded from training by the classifier pipeline and so are not in these numbers.
- **Continuity below 1.0:** continuity stays below 1.0 even with the HMM. At 1 Hz and high speed, a vehicle can cross a short arc (27% of route arcs are under 20 m) between two fixes. The decoded sequence then skips that arc, which the continuity metric counts as a break even though the route is drivable.

**Sensitivity check on the labels** (not used for any choice). Re-scoring the same predictions against labels that follow the route in order (`src/hmm_viterbi/labels.py`, a monotone alignment of points to `.route`) gives:

| Method | Student-test, A's labels | Student-test, order-following labels | Train, A's labels | Train, order-following labels |
|---|---|---|---|---|
| Classifier only | 0.863 | 0.865 | 0.850 | 0.854 |
| Classifier + HMM | 0.916 | 0.928 | 0.899 | 0.923 |

The HMM's apparent errors shrink more than the classifier's under order-following labels: on trip 00000056 it goes from 0.820 to 0.949. Part of the remaining "error" is therefore label ambiguity rather than wrong routes. We report A's labels as the official numbers because the classifier was trained on them.

**Development baseline.** Before the classifier was trained, the decoder was built and tested with a distance-only emission (a Gaussian on point-to-arc distance, σ = 5 m, whose top choice is the nearest arc). On student-test that gave:

| Emission | Point accuracy, classifier only | Point accuracy, + HMM |
|---|---|---|
| Distance only | 0.551 | 0.923 |

Those numbers use the order-following labels and the earlier decoder settings (`results/hmm_vs_classifier_summary.csv`). They show that the HMM alone fixes most direction errors: in a two-way street both directions are equally near, so 67–84% of the distance-only errors were wrong-direction errors. The trained classifier brings the classifier-only accuracy up from 0.55 to 0.86, mainly through its heading feature, and the HMM then adds a further 5 points.

### 5.7 Limitations

- **Labels:** point-level labels come from `.route`, which is reference data, not verified truth. About 2% of consecutive route arcs are disconnected in some trips, two trips don't match their GPS track, and the nearest-route-arc rule is ambiguous on roads driven twice (5.6).
- **Chain breaks:** a break keeps the decoder robust, but the route loses continuity at that point, and on gap-heavy trips (00000081) a wrong choice can persist after a restart.
- **Parallel roads:** unconnected parallel roads in the same direction remain the main source of genuine errors.

---

## 6. Geographical Visualisation Using Folium

Maps are generated from the actual pipeline output by `src/viz_folium/maps.py`, using the tuned classifier's probabilities and the selected decoder settings:

```bash
python -m src.viz_folium.maps --probs results/final_model_probs_student_test.csv --trip data/student_test/00000052
```

The interactive maps are committed in `results/maps/`: student-test trips 00000009, 00000052, 00000065 and 00000076.

**Layers** (each can be switched on and off in the layer control):

| Layer | Style | Shown by default |
|---|---|---|
| Road network near the trip | grey thin lines | yes |
| GPS observations | blue dots; tooltip shows index and timestamp | yes |
| Reference route (`.route`) | green, thick, semi-transparent, so the predicted route is visible on top | yes |
| Predicted route, classifier + HMM | red | yes |
| Classifier-only route | orange, dashed | no |
| HMM error points (prediction ≠ label) | black dots | no |
| Start and end markers | green and red pins | yes |

**Other features:**
- **Legend:** a fixed legend in the lower left.
- **Initial view:** the map opens zoomed to the extent of the GPS track.
- **Map size:** only arcs within 300 m of the track's bounding box are drawn, so maps of trips with up to 300k arcs stay usable.
- **UI version:** `trip_map(..., show_reference=False)` leaves out the reference layer, so ground truth never reaches the evaluator UI.

**Figure 6.1 — Student-test trip 00000052, full trip** (`results/figures/map_trip_00000052_overview.png`). The red HMM route follows the green reference route along its whole length; the HMM error layer reports 1 mismatched point out of 262.

![Trip 00000052 overview](../results/figures/map_trip_00000052_overview.png)

**Figure 6.2 — Student-test trip 00000076, interchange close-up** (`results/figures/map_trip_00000076_interchange.png`). The classifier-only route (orange dashes) leaves the main road for short pieces of the ramps and of the road crossing underneath, because those arcs score highest for some individual points. The HMM route (red) stays on the main road, because those arcs are not reachable from the previous arc at the observed distance. On this trip the HMM cuts the point error from 14.1% to 4.8%.

![Trip 00000076 interchange](../results/figures/map_trip_00000076_interchange.png)

**A limitation of the maps:** a wrong-direction prediction is drawn exactly on top of the correct road, so those errors are invisible on a map even though they are counted. The HMM error layer, the black dots, shows these points.
