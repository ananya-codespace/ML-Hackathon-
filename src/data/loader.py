from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def list_trips(partition):
    """partition: 'train' or 'student_test'"""
    return sorted(p.name for p in (DATA / partition).iterdir() if p.is_dir())

def _read(path, names, dtypes=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes, engine="python")

def load_trip(partition, trip_id, with_route=True):
    d = DATA / partition / trip_id
    out = {
        "track": _read(d / f"{trip_id}.track", ["lon", "lat", "t"]),
        "nodes": _read(d / f"{trip_id}.nodes", ["lon", "lat"]),
        "arcs":  _read(d / f"{trip_id}.arcs", ["from_node", "to_node"]),
    }
    out["nodes"].index.name = "node_id"
    out["arcs"].index.name = "arc_id"   # arc_id = 0-based row index
    if with_route:
        out["route"] = _read(d / f"{trip_id}.route", ["arc_id"])["arc_id"]
    return out