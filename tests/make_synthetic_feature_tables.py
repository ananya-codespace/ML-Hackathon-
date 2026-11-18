"""SMOKE-TEST ONLY.  Writes fake feature tables with the agreed schema so the pipeline can be exercised
before Person A's real tables exist.  Numbers produced from this data are meaningless - never put them in the report.
    python tests/make_synthetic_feature_tables.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_trips, prefix, rng, k=6):
    rows = []
    for t in range(n_trips):
        trip = f"{prefix}_{t:03d}"; n_obs = int(rng.randint(80, 220)); noise = rng.uniform(0.7, 1.5)
        prev_d = 10.0
        for o in range(n_obs):
            has_true = rng.rand() < 0.96; true_pos = rng.randint(k) if has_true else -1
            d_true = abs(rng.normal(9, 6)) * noise; speed = abs(rng.normal(9, 4)); gap = 1 if rng.rand() < .97 else rng.randint(2, 30)
            d_all = np.where(np.arange(k) == true_pos, d_true, rng.uniform(4, 70, k))
            rank = d_all.argsort().argsort()
            for c in range(k):
                is_t = int(c == true_pos)
                rows.append((trip, o, int(rng.randint(10**5)), is_t, d_all[c], rng.rand(),
                             abs(rng.normal(20, 18)) if is_t else rng.uniform(0, 180), speed, gap,
                             rng.uniform(20, 400), rank[c], prev_d, d_all[c] - d_all.min()))
            prev_d = d_true
    return pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "label", "dist_to_arc_m", "pos_along_arc",
                                       "heading_diff_deg", "speed_mps", "time_gap_s", "arc_length_m",
                                       "cand_rank", "prev_obs_dist_m", "dist_minus_min_m"])


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rng = np.random.RandomState(0)
    make(80, "train", rng).to_csv(out / "train_features.csv", index=False)
    make(10, "stest", rng).to_csv(out / "student_test_features.csv", index=False)
    print("wrote synthetic tables to", out)
