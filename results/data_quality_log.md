# Data-quality log (Person A)

| # | Issue / observation | Evidence file | Decision |
|---|---|---|---|
| 1 | Trips come from many countries, each with its own road network | projection_and_route_distance.csv | Per-trip UTM zone (25 zones used); one global zone caused inf values |
| 2 | Huge timestamp gaps in some train trips (up to 153,064 s) | eda_train_trips.csv | Mostly parked vehicle (<25 m jump); kept; heading disabled across gaps (head_ok) |
| 3 | Real position jumps at gaps in trips 55 (884 m) and 70 (538 m) | eda_train_trips.csv | Kept; noted as anomalies |
| 4 | Fragmented networks (median 20 components, up to 421); reference route always inside one | eda_train_trips.csv | No change |
| 5 | Reference route breaks in 10 trips, up to 1.9% of consecutive arcs | eda_train_trips.csv | Labels treated as reference, not perfect truth |
| 6 | Duplicate arcs (up to 322) and self-loops (up to 8) | eda_train_trips.csv | Kept as provided |
| 7 | Trips 33 and 54: GPS shifted ~115-137 m from reference route | far_points_diagnosis.csv | Kept, flagged (trip_flagged), reported with and without; not realigned (would use labels) |
| 8 | Trip 2: far points mostly at trip start/end | far_points_diagnosis.csv | Kept |
| 9 | Label = nearest reference-route arc to each GPS point | candidate_miss_grid_clean.csv | Limits: junction ambiguity, parallel roads, misaligned trips |
| 10 | Candidates: r = 75 m, k = 10 | candidate_miss_grid_clean.csv | Miss rate 0.99% (clean train), 0.16% (student test), 10.92% (all train) |
| 11 | Scaler and folds use train data only | cv_folds.csv, scaler.joblib | No evaluator data used anywhere |
