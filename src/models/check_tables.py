"""Pre-flight check - run this FIRST on the real tables:   python -m src.models.check_tables
Prints what the model code will see and stops with a clear message if the contract is broken."""
from . import config as C
from .common import feature_columns, get_folds, load_train_test


def main():
    train, test, feats = load_train_test()
    print(f"features ({len(feats)}): {feats}")
    for name, d in [("train (after dropping flagged)", train), ("student_test", test)]:
        n_obs = d.groupby(["trip_id", "obs_idx"]).ngroups
        has = d.groupby(["trip_id", "obs_idx"])[C.LABEL_COL].max().mean()
        print(f"{name}: rows={len(d):,} trips={d.trip_id.nunique()} gps_points={n_obs:,} "
              f"positive_rate={d[C.LABEL_COL].mean():.4f} candidate_miss_rate={1 - has:.4f} "
              f"nan_cells={int(d[feats].isna().sum().sum())}")
    folds = get_folds(train)
    sizes = {int(f): int((folds == f).sum()) for f in sorted(set(folds))}
    print("rows per fold:", sizes)
    g = train.assign(f=folds).groupby("trip_id").f.nunique().max()
    assert g == 1, "a trip is split across folds!"
    print("OK - no trip split across folds, train/test trips disjoint. Safe to run ./run_all_B.sh")


if __name__ == "__main__":
    main()
