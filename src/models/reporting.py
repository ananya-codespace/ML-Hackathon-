"""Writes the result files that every report number must trace back to."""
import json

import numpy as np
import pandas as pd

from . import config as C
from .zoo import INFO, MODEL_ORDER


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def short_config(cfg):
    return ", ".join(f"{k}={v}" for k, v in cfg["params"].items()) + f", imbalance={cfg['imbalance']}"


def comparison_table(records):
    """Table 3.1 of the report (one row per model), straight from the records."""
    rows = []
    for r in records:
        i = INFO[r["model"]]
        rows.append({
            "Model": i["label"], "Algorithm family": i["family"],
            "Key configuration": short_config(r["config"]),
            "CV macro F1 (mean)": r["cv"]["cv_macro_f1_mean"], "CV macro F1 (std)": r["cv"]["cv_macro_f1_std"],
            "CV Top-1": r["cv"]["cv_top1_mean"],
            "Route continuity (student-test, Top-1 picks)": r["route_continuity_student_test_top1"],
            "Student-test macro F1": r["student_test"]["macro_f1"],
            "Student-test Top-1": r["student_test"]["top1_accuracy"],
            "Student-test precision": r["student_test"]["precision"],
            "Student-test recall": r["student_test"]["recall"],
            "Student-test F1 (positive class)": r["student_test"]["f1"],
            "Log loss (student-test)": r["student_test"]["log_loss"],
            "Train time (s)": r["train_time_s"], "Inference time (s)": r["inference_time_s"],
            "Inference (us/row)": r["inference_us_per_row"], "Decision threshold (from train OOF)": r["cv"]["threshold"],
        })
    return pd.DataFrame(rows)


def writeup_facts(records, path):
    """Auto-generated FACT SHEET per model (answers to Deliverables Q5-Q9 that come from numbers).
    The 'why it suits PathSnap / strengths / limits' reasoning must be written by the team from these facts."""
    L = ["# Per-model fact sheet (generated from results/comparison_results.json)\n",
         "Numbers below are copied from saved result files. The heuristic flags are only prompts for you to look at the "
         "evidence - write your own interpretation for the report.\n"]
    for r in records:
        i, c = INFO[r["model"]], r["config"]
        tr, cv, te = r["train"], r["cv"], r["student_test"]
        gap_tc = tr["macro_f1"] - cv["cv_macro_f1_mean"]; gap_ct = cv["cv_macro_f1_mean"] - te["macro_f1"]
        hint = []
        if gap_tc > 0.05: hint.append(f"train macro F1 exceeds CV by {gap_tc:.3f} (possible overfitting)")
        if cv["cv_macro_f1_mean"] < 0.7 and tr["macro_f1"] < 0.7: hint.append("train AND CV macro F1 both low (possible underfitting)")
        if abs(gap_ct) > 0.05: hint.append(f"CV vs student-test macro F1 differ by {gap_ct:+.3f}")
        if cv["cv_macro_f1_std"] > 0.03: hint.append(f"fold std of macro F1 is {cv['cv_macro_f1_std']:.3f} (unstable across trips)")
        L += [f"\n## {i['label']}  ({i['family']})\n",
              f"- Hyperparameters (initial values): {short_config(c)}",
              f"- Objective / loss: {i['objective']}",
              f"- Student-test: Top-1 = {te['top1_accuracy']:.4f}, macro F1 = {te['macro_f1']:.4f}, "
              f"precision = {te['precision']:.4f}, recall = {te['recall']:.4f}, log loss = {te['log_loss']:.4f}",
              f"- Train (in-sample): Top-1 = {tr['top1_accuracy']:.4f}, macro F1 = {tr['macro_f1']:.4f}, log loss = {tr['log_loss']:.4f}",
              f"- Grouped CV (5 folds by trip): macro F1 = {cv['cv_macro_f1_mean']:.4f} +/- {cv['cv_macro_f1_std']:.4f}, "
              f"Top-1 = {cv['cv_top1_mean']:.4f} +/- {cv['cv_top1_std']:.4f}",
              f"- Timing: train {r['train_time_s']:.2f}s, inference {r['inference_time_s']:.3f}s ({r['inference_us_per_row']:.1f} us/row)",
              f"- Over/underfitting evidence: train->CV macro-F1 gap {gap_tc:+.4f}; CV->student-test gap {gap_ct:+.4f}",
              f"- Heuristic flags: {'; '.join(hint) if hint else 'none triggered'}"]
    path.write_text("\n".join(L) + "\n")
