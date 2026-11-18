import numpy as np, shapely
from shapely import STRtree

def label_points(net, pts_xy, route):
    """Label of each GPS point = nearest arc among the reference-route arcs.
    Returns (label_arc_id, distance_to_that_arc_m)."""
    route = np.asarray(route)
    tree = STRtree(net.geoms[route])
    # all_matches=True returns every equally-near route arc; keep the earliest one in the route
    # so ties (e.g. a street driven in both directions) are resolved the same way every run.
    (pt, pos), dist = tree.query_nearest(shapely.points(pts_xy), return_distance=True, all_matches=True)
    best = np.full(len(pts_xy), np.iinfo(np.int64).max)
    np.minimum.at(best, pt, pos)
    d = np.zeros(len(pts_xy))
    d[pt] = dist
    return route[best], d