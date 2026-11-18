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
│    5-Fold Trip-Grouped CV across 6 models │
│    + GP-EI Bayesian Optimization (30 runs)│
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
* **Data Hygiene:** 2 misaligned training trips (where the reference route lies >50 m from most GPS observations) are excluded during training/CV (`DROP_FLAGGED_TRAIN_TRIPS`), while student-test trips remain untouched. Candidate recall is **99.01%** on clean training trips (0.99% miss rate) and **99.84%** on student-test trips (0.16% miss rate).
* **Engineered Features (14):**
  * **Spatial proximity:** `dist`, `rank`, `dist_min`, `dist_gap`, `n_cand`
  * **Arc geometry & alignment:** `t_along`, `arc_len`, `cos_head`, `head_ok`
  * **Kinematics & temporal context:** `speed`, `dt_prev`, `dt_next`, `dist_prev`, `dist_next`
* **Leakage Prevention:** All cross-validation splits are strictly grouped by `trip_id` (5 folds) so no single trip is ever split across train and validation folds.

### 2. Model Comparison (`src/models/run_comparison.py`)
Models were evaluated using **5-fold trip-grouped cross-validation** with **Macro F1** as the primary metric (accounting for heavy class imbalance) alongside **Log Loss** (to ensure well-calibrated emission probabilities for the downstream HMM) and **Top-1 Point Accuracy**:

| Model | Family | CV Macro F1 | Student-Test Macro F1 | Student-Test Top-1 | Student-Test Log Loss | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Gradient Boosting | 0.8993 ± 0.0218 | 0.9030 | 0.8385 | **0.1029** | 0.124s (0.8 µs/row) |
| **Random Forest** | Bagged Ensemble | **0.9007 ± 0.0228** | **0.9054** | **0.8412** | 0.1050 | 0.360s (2.4 µs/row) |
| **MLP (2 hidden layers)** | Neural Network | 0.8975 ± 0.0222 | 0.9012 | 0.8383 | 0.1052 | 0.047s (0.3 µs/row) |
| **Logistic Regression** | Linear | 0.8843 ± 0.0199 | 0.8901 | 0.8308 | 0.1353 | **0.012s** (0.1 µs/row) |
| **Gaussian Naive Bayes** | Generative | 0.8301 ± 0.0171 | 0.8280 | 0.8135 | 0.9173 | 0.017s (0.1 µs/row) |

### 3. Model Selection & Bayesian Optimization (`select_final.py`, `run_bayes.py`)
* **1-SE Selection Rule:** Applying the pre-specified one-standard-error rule on CV Macro F1 (`0.9007 - 0.0102`), `HistGradientBoosting`, `Random Forest`, and `MLP` were statistically tied. **HistGradientBoosting** was selected for achieving the **lowest log loss** (critical for HMM emission probability calibration) and 3× faster inference than Random Forest.
* **Bayesian Optimization:** Tuned across 30 trials using a custom Gaussian Process (`Matérn 5/2` kernel) + Expected Improvement optimizer initialized with 8 scrambled-Sobol points, optimizing 5-fold grouped-CV Macro F1. The final decision threshold is frozen at **`0.44`** on out-of-fold predictions.

### 4. HMM / Viterbi Decoding (`src/hmm_viterbi/`)
While point-wise classifiers score each GPS observation independently, road trips must follow connected network edges. The HMM layer takes the classifier's calibrated probabilities as emission likelihoods and combines them with network-topology transition probabilities via the **Viterbi algorithm** to eliminate off-route jumps and maximize **Route Continuity** and **Route F1**.

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/ananya-codespace/ML-Hackathon-.git
cd ML-Hackathon-
pip install -r requirements.txt
```

### 2. Run the Full ML & Evaluation Pipeline
To regenerate feature tables, run the model zoo comparison, execute Bayesian hyperparameter tuning, and produce evaluation plots:
```bash
# 1. Build processed feature tables (requires raw data in data/)
python src/features/make_tables.py

# 2. Run model comparison, selection, 30-trial Bayesian optimization & plotting
bash run_all_B.sh
```

### 3. Launch the FastAPI Backend & Streamlit UI
Start the backend server in one terminal:
```bash
uvicorn src.api.main:app --reload --port 8000
```
* **GET `/trips`** — Lists all available trips in the configured partition.
* **POST `/predict/{trip_id}`** — Runs candidate generation, `FinalClassifier` probability scoring, and HMM/Viterbi decoding for `trip_id`, returning predicted `arc_id`s, point accuracy, route continuity, route F1, and an interactive Folium map.

Start the Streamlit frontend in a second terminal:
```bash
streamlit run src/ui/app.py
```

---

## Testing
Run the synthetic feature-table test suite to verify feature alignment and schema guards:
```bash
pytest tests/
```