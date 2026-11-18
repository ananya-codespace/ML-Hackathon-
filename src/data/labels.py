import numpy as np, shapely
from shapely import STRtree

def label_points(net, pts_xy, route):
    """Label of each GPS point = nearest arc among the reference-route arcs.
    Returns (label_arc_id, distance_to_that_arc_m)."""
    route = np.asarray(route)
    tree = STRtree(net.geoms[route])
    idx, dist = tree.query_nearest(shapely.points(pts_xy), return_distance=True, all_matches=False)
    return route[idx[1]], dist