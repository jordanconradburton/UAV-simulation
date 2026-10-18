"""
core/mountains.py

Mountain geometry. A mountain is an axis-aligned square prism standing on the
ground, given as (mountain_id, mx, my, side_length, height): centered on
(mx, my), side_length across, from z = 0 up to height.
"""


def point_in_mountain(point, mountain):
    """
    Check if a 3D point lies inside a mountain (within its square base and
    under its height).
    """
    x, y, z = point
    _, mx, my, side, h = mountain
    half = side / 2
    if (mx - half) <= x <= (mx + half) and (my - half) <= y <= (my + half) and 0 <= z <= h:
        return True
    return False

def segment_intersects_mountain(a, b, mountain, eps=1e-9):
    """
    Check if the line segment from a to b intersects the axis‐aligned square prism of the mountain.
    Uses the slab method for AABB intersection. Returns True if any intersection.
    """
    ax, ay, az = a
    bx, by, bz = b
    _, mx, my, side, h = mountain
    half = side / 2.0
    # Define min/max coords
    xmin, xmax = mx - half, mx + half
    ymin, ymax = my - half, my + half
    zmin, zmax = 0.0, h

    def slab(minc, maxc, a_coord, b_coord):
        d = b_coord - a_coord
        if abs(d) < eps:
            # Parallel to slab; must be inside
            if a_coord < minc or a_coord > maxc:
                return None
            return (0.0, 1.0)
        t0 = (minc - a_coord) / d
        t1 = (maxc - a_coord) / d
        return (min(t0, t1), max(t0, t1))

    slabs = []
    for (minc, maxc, a_coord, b_coord) in [
        (xmin, xmax, ax, bx),
        (ymin, ymax, ay, by),
        (zmin, zmax, az, bz)
    ]:
        s = slab(minc, maxc, a_coord, b_coord)
        if s is None:
            return False
        slabs.append(s)
    t_enter = max(s[0] for s in slabs)
    t_exit  = min(s[1] for s in slabs)
    return (t_enter <= t_exit) and (t_exit >= 0.0) and (t_enter <= 1.0)

def point_clear_of_mountains(point, mountain_list):
    """
    Return True if `point` is not inside any mountain in `mountain_list` —
    i.e. a sensor can be placed there, or a drone can start there.
    """
    return not any(point_in_mountain(point, m) for m in mountain_list)


def reject_if_mountain(a, b, mountain_list):
    """
    Return True if the segment from a to b intersects any mountain in
    `mountain_list`, i.e. a drone step from a to b must be rejected.
    """
    return any(segment_intersects_mountain(a, b, m) for m in mountain_list or [])
