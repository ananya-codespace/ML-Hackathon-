"""The model zoo: six classifiers from six different algorithm families.

Every entry is a scikit-learn Pipeline (imputer -> [scaler] -> classifier) so that
imputation/scaling statistics are always learned on the training rows only, and the
whole thing can be saved/loaded as one artifact.

imbalance = "none"     -> keep the true class prior, so predicted probabilities stay
                          honest (needed for log loss and as HMM emissions).  DEFAULT.
imbalance = "balanced" -> class-weighted loss (only for models that support it); used
                          in the imbalance ablation.
Whatever the choice, the decision threshold for macro-F1 is picked on out-of-fold
TRAIN predictions (see metrics.best_threshold); Top-1 is rank-based and unaffected.
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


class CappedKNN(KNeighborsClassifier):
    """KNN that fits on a uniform random subsample of at most `max_train_rows` rows.
    Uniform row sampling keeps the class prior, so probabilities are not distorted;
    it only keeps prediction time bearable on ~1M-row tables."""

    def __init__(self, n_neighbors=25, weights="distance", n_jobs=-1, max_train_rows=200_000, random_state=SEED):
        super().__init__(n_neighbors=n_neighbors, weights=weights, n_jobs=n_jobs)
        self.max_train_rows, self.random_state = max_train_rows, random_state

    def fit(self, X, y):
        if len(X) > self.max_train_rows:
            idx = __import__("numpy").random.RandomState(self.random_state).choice(len(X), self.max_train_rows, replace=False)
            X, y = X[idx], y[idx]
        return super().fit(X, y)


def _pipe(clf, scale):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------- default (untuned) params
DEFAULTS = {
    "logreg": dict(C=1.0, max_iter=500),
    "gaussian_nb": dict(var_smoothing=1e-9),
    "knn": dict(n_neighbors=25, weights="distance", max_train_rows=200_000),
    "random_forest": dict(n_estimators=150, max_depth=None, min_samples_leaf=5, max_features="sqrt", max_samples=0.25),
    "hist_gb": dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=0.0),
    "mlp": dict(hidden=64, hidden2=32, alpha=1e-4, learning_rate_init=1e-3, max_iter=100),
    "decision_tree": dict(max_depth=12, min_samples_leaf=50),
}


def build(name, params=None, imbalance="none"):
    p = {**DEFAULTS[name], **(params or {})}
    cw = "balanced" if imbalance == "balanced" else None
    if name == "logreg":
        return _pipe(LogisticRegression(C=p["C"], max_iter=int(p["max_iter"]), class_weight=cw,
                                        random_state=SEED), scale=True)
    if name == "gaussian_nb":
        return _pipe(GaussianNB(var_smoothing=p["var_smoothing"]), scale=True)
    if name == "knn":
        return _pipe(CappedKNN(n_neighbors=int(p["n_neighbors"]), weights=p["weights"],
                                 n_jobs=-1, max_train_rows=int(p["max_train_rows"])), scale=True)
    if name == "random_forest":
        return _pipe(RandomForestClassifier(
            n_estimators=int(p["n_estimators"]), max_depth=None if p["max_depth"] in (None, 0) else int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
            max_samples=p["max_samples"], class_weight="balanced_subsample" if cw else None, n_jobs=-1, random_state=SEED), scale=False)
    if name == "hist_gb":
        return _pipe(HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"], max_iter=int(p["max_iter"]),
            max_leaf_nodes=int(p["max_leaf_nodes"]), min_samples_leaf=int(p["min_samples_leaf"]),
            l2_regularization=p["l2_regularization"], class_weight=cw, random_state=SEED), scale=False)
    if name == "mlp":
        hidden = (int(p["hidden"]), int(p["hidden2"])) if int(p["hidden2"]) > 0 else (int(p["hidden"]),)
        return _pipe(MLPClassifier(hidden_layer_sizes=hidden, alpha=p["alpha"],
                                   learning_rate_init=p["learning_rate_init"], max_iter=int(p["max_iter"]),
                                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                   random_state=SEED), scale=True)
    if name == "decision_tree":
        return _pipe(DecisionTreeClassifier(max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_samples_leaf"]),
                                            class_weight=cw, random_state=SEED), scale=False)
    raise KeyError(name)


# ---------------------------------------------------------------- documentation metadata
INFO = {
    "logreg": dict(label="Logistic Regression", family="Linear / probabilistic",
                   objective="L2-regularised binary log loss (convex, solved by L-BFGS)",
                   supports_imbalance=True, has_epoch_history=False),
    "gaussian_nb": dict(label="Gaussian Naive Bayes", family="Generative probabilistic",
                        objective="Maximise class-conditional Gaussian likelihood (closed-form, no iterative loss)",
                        supports_imbalance=False, has_epoch_history=False),
    "knn": dict(label="Distance-weighted KNN", family="Instance-based",
                objective="None (lazy learner): probability = distance-weighted vote of k neighbours",
                supports_imbalance=False, has_epoch_history=False),
    "random_forest": dict(label="Random Forest", family="Bagged tree ensemble",
                          objective="Gini impurity reduction per split; probability = mean of tree leaf frequencies",
                          supports_imbalance=True, has_epoch_history=False),
    "hist_gb": dict(label="HistGradientBoosting", family="Gradient boosting",
                    objective="Binary log loss, minimised stage-wise with histogram-binned trees",
                    supports_imbalance=True, has_epoch_history=False),
    "mlp": dict(label="MLP (2 hidden layers)", family="Neural network",
                objective="Binary log loss + L2 penalty (alpha), Adam optimiser, early stopping on a 10% validation split",
                supports_imbalance=False, has_epoch_history=True),
    "decision_tree": dict(label="Decision Tree", family="Single decision tree",
                          objective="Gini impurity reduction per split (greedy); probability = leaf class frequency",
                          supports_imbalance=True, has_epoch_history=False),
}
MODEL_ORDER = ["logreg", "gaussian_nb", "decision_tree", "random_forest", "hist_gb"]
# Not run in the final comparison (too slow on 1.4M rows): "knn", "mlp"

# ---------------------------------------------------------------- Bayesian-optimisation search spaces
# type: float | logfloat | int | logint ; ranges are justified in README_B.md
SEARCH_SPACES = {
    "hist_gb": [
        dict(name="learning_rate", type="logfloat", low=0.02, high=0.3),
        dict(name="max_iter", type="int", low=100, high=500),
        dict(name="max_leaf_nodes", type="int", low=8, high=96),
        dict(name="min_samples_leaf", type="logint", low=5, high=200),
        dict(name="l2_regularization", type="logfloat", low=1e-4, high=10.0),
    ],
    "random_forest": [
        dict(name="n_estimators", type="int", low=100, high=400),
        dict(name="max_depth", type="int", low=6, high=40),
        dict(name="min_samples_leaf", type="logint", low=1, high=50),
    ],
    "mlp": [
        dict(name="hidden", type="logint", low=16, high=128),
        dict(name="hidden2", type="logint", low=8, high=64),
        dict(name="alpha", type="logfloat", low=1e-6, high=1e-1),
        dict(name="learning_rate_init", type="logfloat", low=1e-4, high=1e-2),
    ],
    "logreg": [dict(name="C", type="logfloat", low=1e-3, high=100.0)],
    "knn": [dict(name="n_neighbors", type="int", low=5, high=100)],
}
