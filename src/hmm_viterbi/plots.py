"""Figures for report Section 5, generated from the saved result CSVs.

    python -m src.hmm_viterbi.plots --results results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"classifier_only": "Classifier only", "classifier_hmm": "Classifier + HMM"}
COLORS = {"classifier_only": "#ef6c00", "classifier_hmm": "#c62828"}


def per_trip(results: Path, part: str, metric: str, out: Path):
    df = pd.read_csv(results / f"hmm_vs_classifier_{part}.csv", dtype={"trip_id": str})
    wide = df.pivot(index="trip_id", columns="method", values=metric)
    wide = wide.sort_values("classifier_only")
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(wide) + 3), 3.6))
    x = range(len(wide))
    w = 0.42
    for k, (m, off) in enumerate([("classifier_only", -w / 2), ("classifier_hmm", w / 2)]):
        ax.bar([i + off for i in x], wide[m], width=w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(wide.index, rotation=90, fontsize=6 if len(wide) > 20 else 8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} per trip ({part})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"{metric}_{part}.png", dpi=150)
    plt.close(fig)


def tuning(results: Path, out: Path):
    g = pd.read_csv(results / "decoder_tuning_train.csv")
    best = g.iloc[0]
    g = g[(g["same_arc_tol"] == best["same_arc_tol"])]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for s, sub in g.groupby("sigma"):
        sub = sub.sort_values("beta")
        ax.plot(sub["beta"], sub["mean_point_accuracy"], marker="o", label=f"sigma={s:g} m")
    ax.set_xscale("log")
    ax.set_xlabel("beta (m)")
    ax.set_ylabel("mean point accuracy (train)")
    ax.set_title(f"Decoder tuning, 80 train trips (tol={best['same_arc_tol']:g} m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "decoder_tuning_train.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for part in ["train", "student_test"]:
        for metric in ["point_accuracy", "route_continuity", "route_f1"]:
            per_trip(results, part, metric, out)
    tuning(results, out)
    print("figures in", out)


if __name__ == "__main__":
    main()
