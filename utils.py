"""
Utility functions for silkworm tracking
"""

def ccw(A, B, C):
    """Check if three points are in counter-clockwise order"""
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def bbox_overlap(b1, b2):
    """Check if two bounding boxes overlap"""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def point_segment_distance(P, A, B):
    """Compute the shortest Euclidean distance from point P to segment AB."""
    px, py = float(P[0]), float(P[1])
    ax, ay = float(A[0]), float(A[1])
    bx, by = float(B[0]), float(B[1])

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx*abx + aby*aby
    if denom == 0.0:
        dx = px - ax
        dy = py - ay
        return (dx*dx + dy*dy) ** 0.5

    t = (apx*abx + apy*aby) / denom
    if t < 0.0:
        qx, qy = ax, ay
    elif t > 1.0:
        qx, qy = bx, by
    else:
        qx = ax + t * abx
        qy = ay + t * aby

    dx = px - qx
    dy = py - qy
    return (dx*dx + dy*dy) ** 0.5
