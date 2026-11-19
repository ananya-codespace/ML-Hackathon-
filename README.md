# PathSnap — GPS Map-Matching System

**PathSnap** is an end-to-end machine learning and probabilistic sequence-decoding system that snaps noisy GPS trajectories onto road network segments (`arc_id`). It combines spatial-kinematic feature engineering, a tuned **HistGradientBoosting** candidate classifier (selected via grouped cross-validation and Bayesian optimization), and a **Hidden Markov Model (HMM) with Viterbi decoding** to reconstruct topologically continuous routes—served through a **FastAPI** backend and an interactive **Streamlit + Folium** web interface.

---

## System Architecture

```text
Raw GPS Traces & Road Network
            │
            ▼
┌───────────────────────────────────────────┐
│ 1. Spatial Candidate Generation & Features│  (src/data/, src/features/)
│    14 geometric, kinematic & temporal     │
│    features per (GPS point, candidate arc)│
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ 2. ML Candidate Scoring & Model Selection │  (src/models/)
│    5-Fold Trip-Grouped CV across 5 models │
│    + GP-EI Bayesian Optimization (20 runs)│
│    Winner: Tuned HistGradientBoosting     │
└───────────────────────────────────────────┘
            │  Calibrated Emission Probabilities
            ▼
┌───────────────────────────────────────────┐
│ 3. HMM + Viterbi Sequence Decoding        │  (src/hmm_viterbi/)
│    Enforces road network connectivity &   │
│    route continuity across observations   │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ 4. FastAPI Backend & Streamlit + Folium UI│  (src/api/, src/ui/, src/viz_folium/)
│    REST endpoints + interactive route map │
└───────────────────────────────────────────┘
```

---

## Repository Structure

```text
ML-Hackathon-/
├── artifacts/                     # Serialized production model & metadata
│   ├── final_model.joblib         # Tuned HistGradientBoosting classifier
│   ├── final_model_card.json      # Model metadata, feature order & decision threshold (0.44)
│   ├── hmm_config.json            # Tuned HMM transition/emission parameters
│   └── scaler.joblib              # Feature scaler artifact
├── docs/                          # Technical report sections & documentation
├── results/                       # Metrics, CV tables, interactive HTML maps & plots
│   ├── figures/                   # Evaluation plots (CV F1, confusion matrices, BO curves)
│   ├── hmm_bmodel/                # HMM vs. standalone classifier benchmarks
│   ├── maps/                      # Pre-rendered Folium route visualizations
│   ├── comparison_table.csv       # Full model zoo comparison table
│   ├── final_selection.json       # Automated 1-SE model selection log
│   └── model_writeup_facts.md     # Per-model evaluation summary
├── src/
│   ├── api/                       # FastAPI service (main.py, routes.py, pipeline.py, config.py)
│   ├── data/                      # Data loaders, EDA, geospatial checks & diagnostics
│   ├── features/                  # Feature engineering & parquet table generation
│   ├── hmm_viterbi/               # Road network graph, candidate filtering & Viterbi decoder
│   ├── models/                    # Model zoo, grouped CV, Bayesian optimization & inference
│   ├── ui/                        # Streamlit frontend application (app.py, config.py)
│   └── viz_folium/                # Interactive Folium map builders
├── tests/                         # Feature pipeline unit tests & synthetic data generator
├── requirements.txt               # Core dependencies
└── run_all_B.sh                   # End-to-end model training, tuning & plotting pipeline
```

---

## Methodology & Key Results

### 1. Dataset & Feature Engineering (`src/data/`, `src/features/`)
* **Formulation:** Binary classification over `(trip_id, obs_idx, arc_id)` candidate pairs (~1.45M training rows with a 12.8% positive class rate; ~150k student-test rows).
* **Data Hygiene:** 2 misaligned training trips (where the reference route lies >50 m from most GPS observations) are excluded during training/CV (`DROP_FLAGGED_TRAIN_TRIPS`), while student-test trips remain untouched. Candidate recall is **99.01%** on clean training trips (0.99% miss rate) and **99.83%** on student-test trips (0.17% miss rate).
* **Engineered Features (14):**
  * **Spatial proximity:** `dist`, `rank`, `dist_min`, `dist_gap`, `n_cand`
  * **Arc geometry & alignment:** `t_along`, `arc_len`, `cos_head`, `head_ok`
  * **Kinematics & temporal context:** `speed`, `dt_prev`, `dt_next`, `dist_prev`, `dist_next`
* **Leakage Prevention:** All cross-validation splits are strictly grouped by `trip_id` (5 folds) so no single trip is ever split across train and validation folds.

### 2. Model Comparison (`src/models/run_comparison.py`)
Models were evaluated using **5-fold trip-grouped cross-validation** with **Macro F1** as the primary metric (accounting for heavy class imbalance) alongside **Log Loss** (to ensure well-calibrated emission probabilities for the downstream HMM) and **Top-1 Point Accuracy**:

| Model | Family | CV Macro F1 | Student-Test Macro F1 | Student-Test Top-1 | Route Continuity (Top-1 picks) | Student-Test Log Loss | Inference Time (149,946 rows) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Gradient Boosting | 0.9003 ± 0.0214 | 0.9028 | 0.8392 | 0.9228 | **0.1020** | 0.055s |
| **Random Forest** | Bagged Ensemble | **0.9020 ± 0.0221** | **0.9059** | 0.8425 | **0.9237** | 0.1039 | 0.193s |
| **Decision Tree** | Single Tree | 0.8979 ± 0.0221 | 0.9032 | **0.8439** | 0.9227 | 0.1105 | 0.011s |
| **Logistic Regression** | Linear | 0.8849 ± 0.0191 | 0.8888 | 0.8288 | 0.9146 | 0.1360 | **0.007s** |
| **Gaussian Naive Bayes** | Generative | 0.8305 ± 0.0173 | 0.8277 | 0.7920 | 0.9101 | 0.9985 | 0.011s |

Source: `results/comparison_table.csv`. KNN and an MLP are also implemented in `src/models/zoo.py`; they were left out of the final comparison because they were too slow for 5-fold CV on 1.4M rows. Class-balanced weights were tested for every model that supports them (`results/imbalance_ablation.csv`): they did not improve macro F1 and made log loss worse, so all models are trained unweighted.

### 3. Model Selection & Bayesian Optimization (`select_final.py`, `run_bayes.py`)
* **1-SE Selection Rule:** the pre-specified rule keeps every model within one standard error of the best CV macro F1 (Random Forest, HistGradientBoosting, Decision Tree), then picks the lowest CV log loss. On the final run this favours Random Forest by 0.0003 (0.1156 vs 0.1159), a statistical tie. An earlier run of the same rule picked **HistGradientBoosting**, which was carried into Bayesian optimisation. We kept it because, after tuning, its CV macro F1 equals Random Forest's (0.9017 vs 0.9020, against a fold standard deviation of 0.022), its CV log loss is clearly lower (0.1117 vs 0.1156, which matters for the HMM emissions), and it predicts 3.5× faster.
* **Bayesian Optimization:** 20 trials of a custom Gaussian Process (`Matérn 5/2` kernel) with Expected Improvement, started from 6 scrambled-Sobol points, maximising 5-fold grouped-CV macro F1 over `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf` and `l2_regularization` (`results/bayes_trials.csv`). Best: learning_rate 0.0375, max_iter 371, max_leaf_nodes 28, min_samples_leaf 181, l2 0.0012. The decision threshold is frozen at **`0.44`** on out-of-fold predictions.

| HistGradientBoosting | CV Macro F1 | CV Log Loss | Student-Test Top-1 | Student-Test Macro F1 | Student-Test Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Untuned | 0.9003 | 0.1159 | 0.8392 | 0.9028 | 0.1020 |
| **Tuned (shipped)** | **0.9017** | **0.1117** | **0.8407** | **0.9047** | **0.1001** |

### 4. HMM / Viterbi Decoding (`src/hmm_viterbi/`)
While point-wise classifiers score each GPS observation independently, road trips must follow connected network edges. The HMM layer takes the classifier's probabilities as emissions (hidden states = each GPS point's candidate arcs) and combines them with network-topology transitions, `log T = -|d_route - d_gps| / β` (driving distance by bounded Dijkstra against straight-line GPS distance). The **Viterbi algorithm** (log space, with backpointers) then picks the most likely connected route. The chain restarts after time gaps over 60 s or when no candidate is reachable. Backward GPS jitter of up to 60 m on the same arc is treated as standing still. The decoder settings (β = 2 m, same_arc_tol = 60 m) were tuned on the training trips' out-of-fold probabilities (`artifacts/hmm_config.json`).

### 5. Final Results

| Partition | Method | Point Accuracy (trip mean) | Point Accuracy (pooled) | Route Continuity | Route F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Student-test (10 trips) | Classifier only | 0.863 | 0.841 | 0.924 | 0.905 |
| Student-test (10 trips) | **Classifier + HMM** | **0.916** | **0.889** | **0.977** | **0.980** |
| Evaluator (10 trips) | Classifier only | 0.863 | 0.786 | 0.920 | 0.887 |
| Evaluator (10 trips) | **Classifier + HMM** | **0.913** | **0.894** | **0.960** | **0.965** |

Sources: `results/hmm_bmodel/hmm_vs_classifier_summary.csv` and `results/evaluator_test/summary.csv`. The evaluator trips were run once through the frozen pipeline; nothing was tuned on them.

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/ananya-codespace/ML-Hackathon-.git
cd ML-Hackathon-
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The versions are pinned on purpose: `artifacts/final_model.joblib` only loads with the scikit-learn version that saved it (1.9.1).

### 2. Add the Data (not in git)
Put one folder per trip, each with `<id>.track`, `<id>.nodes`, `<id>.arcs` and `<id>.route`, under:
```text
data/train/<trip_id>/            # 80 training trips
data/student_test/<trip_id>/     # 10 student-test trips
data/evaluator_test/<trip_id>/   # 10 evaluator trips (served by the API)
```

### 3. Reproduce the Results
```bash
python src/features/make_tables.py                          # feature tables -> data/processed/
bash run_all_B.sh                                           # 5-model comparison, selection, 20-trial Bayesian optimisation, plots
python -m src.hmm_viterbi.experiment --probs-source b       # HMM decoder tuning + classifier-only vs HMM -> results/hmm_bmodel/
python -m src.hmm_viterbi.plots --results results/hmm_bmodel
python -m src.evaluate_trips --partition evaluator_test     # frozen pipeline on the evaluator trips -> results/evaluator_test/
python -m src.viz_folium.maps --probs results/final_model_probs_student_test.csv --trip data/student_test/00000052
```

### 4. Run the Two-Laptop Demo
**Laptop 1 (API)** listens on the network and serves the trips in `data/evaluator_test/`:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
* **GET `/trips`** — lists the trips in the served partition (set `PATHSNAP_TRIPS_PARTITION=student_test` to rehearse on student-test).
* **POST `/predict/{trip_id}`** — runs candidate generation, `FinalClassifier` scoring and HMM/Viterbi decoding using only `.track/.nodes/.arcs`. It then scores the route against `.route` on the server and returns the predicted `arc_id`s, point accuracy, route continuity, route F1 and a Folium map without the reference route. Each request is logged with the caller's IP.

**Laptop 2 (UI)**, on the same network (a phone hotspot avoids campus Wi-Fi blocking traffic between devices), points the UI at Laptop 1's IP address:
```bash
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 streamlit run src/ui/app.py
```
Find Laptop 1's IP with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows).

---

## Testing
The tests check the feature tables built by `src/features/make_tables.py` (no trip split across folds, train and student-test trips disjoint), so build the tables first:
```bash
python src/features/make_tables.py
pytest tests/
```