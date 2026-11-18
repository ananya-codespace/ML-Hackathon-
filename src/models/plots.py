"""Step 4 - report figures, every one drawn from saved result files.   python -m src.models.plots"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .zoo import INFO


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / name, dpi=160); plt.close(fig)
    print("saved", C.FIG_DIR / name)


def main():
    recs = json.loads((C.RESULTS_DIR / "comparison_results.json").read_text())
    labels = [INFO[r["model"]]["label"].replace(" (2 hidden layers)", "") for r in recs]
    x = np.arange(len(recs))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, [r["cv"]["cv_macro_f1_mean"] for r in recs], yerr=[r["cv"]["cv_macro_f1_std"] for r in recs], capsize=4, color="#4C72B0")
    ax.set_xticks(x, labels, rotation=30, ha="right"); ax.set_ylabel("macro F1"); ax.set_title("Grouped-CV macro F1 (80 train trips, 5 folds; error bar = fold std)")
    _save(fig, "cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.38
    ax.bar(x - w / 2, [r["student_test"]["top1_accuracy"] for r in recs], w, label="Top-1 accuracy")
    ax.bar(x + w / 2, [r["student_test"]["macro_f1"] for r in recs], w, label="macro F1")
    ax.set_xticks(x, labels, rotation=30, ha="right"); ax.legend(); ax.set_title("Student-test (10 trips): Top-1 accuracy and macro F1")
    _save(fig, "student_test_top1_macro_f1.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, [r["student_test"]["log_loss"] for r in recs], color="#C44E52")
    ax.set_xticks(x, labels, rotation=30, ha="right"); ax.set_ylabel("log loss (lower = better)"); ax.set_title("Student-test log loss")
    _save(fig, "student_test_log_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.27
    for k, (key, lab) in enumerate([("train", "train (in-sample)"), ("cv", "grouped CV"), ("student_test", "student-test")]):
        vals = [r["train"]["macro_f1"] if key == "train" else r["cv"]["cv_macro_f1_mean"] if key == "cv" else r["student_test"]["macro_f1"] for r in recs]
        ax.bar(x + (k - 1) * w, vals, w, label=lab)
    ax.set_xticks(x, labels, rotation=30, ha="right"); ax.legend(); ax.set_ylabel("macro F1"); ax.set_title("Over/underfitting check: train vs CV vs student-test macro F1")
    _save(fig, "overfitting_check.png")

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, lab in [(axs[0], "train_time_s", "train time (s)"), (axs[1], "inference_us_per_row", "inference (us / candidate row)")]:
        ax.bar(x, [r[key] for r in recs], color="#55A868"); ax.set_xticks(x, labels, rotation=40, ha="right"); ax.set_title(lab); ax.set_yscale("log")
    _save(fig, "compute_cost.png")

    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, r, lab in zip(axs.ravel(), recs, labels):
        t = r["student_test"]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(np.log10(cm + 1), cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(lab, fontsize=9)
    fig.suptitle("Student-test confusion matrices (threshold from train OOF; colour = log10 count)")
    _save(fig, "confusion_matrices.png")

    hist = C.RESULTS_DIR / "mlp_history.json"
    if hist.exists():
        h = json.loads(hist.read_text()); fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["loss_curve"], label="training loss (log loss + L2)")
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
        if h.get("validation_accuracy"):
            ax2 = ax.twinx(); ax2.plot(h["validation_accuracy"], color="orange", label="validation accuracy (10% split)"); ax2.set_ylabel("validation accuracy")
        ax.set_title("MLP epoch history (only model with a real one)"); ax.legend(loc="upper center")
        _save(fig, "mlp_loss_curve.png")

    tr = C.RESULTS_DIR / "bayes_trials.csv"
    if tr.exists():
        t = pd.read_csv(tr); fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(t.trial + 1, t.cv_macro_f1, c=np.where(t.phase == "sobol_init", "#999999", "#4C72B0"))
        ax.plot(t.trial + 1, t.cv_macro_f1.cummax(), color="red", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel("grouped-CV macro F1"); ax.set_title("Bayesian optimisation (grey = Sobol init, blue = GP-EI)"); ax.legend()
        _save(fig, "bayes_convergence.png")
    cp = C.RESULTS_DIR / "bayes_comparison.csv"
    if cp.exists():
        c = pd.read_csv(cp); cols = ["CV macro F1 (mean)", "Student-test Top-1", "Student-test macro F1"]
        fig, axs = plt.subplots(1, 2, figsize=(9, 4)); w = 0.35; xx = np.arange(len(cols))
        for k, v in enumerate(["untuned", "tuned"]): axs[0].bar(xx + (k - .5) * w, c.loc[c.variant == v, cols].iloc[0], w, label=v)
        axs[0].set_xticks(xx, ["CV macro F1", "test Top-1", "test macro F1"]); axs[0].legend(); axs[0].set_title("Tuned vs untuned")
        axs[1].bar(c.variant, c["Student-test log loss"], color=["#999999", "#4C72B0"]); axs[1].set_title("Student-test log loss")
        _save(fig, "tuned_vs_untuned.png")


if __name__ == "__main__":
    main()
