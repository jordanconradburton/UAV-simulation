def platesCalc(Mountain):
    # each plate is a tupel of (normal vector, constant point, range(A, B)), or (N, C, (A, B))
    (x, y, r, h) = Mountain
    plate1 = ((0, 0, 1), (x, y, h), ((x - r/2, y - r/2, h), (x + r/2, y + r/2, h)))
    plate2 = ((1, 1, 0), (x, y, 0), ((x - r/2, y + r/2, 0), (x + r/2, y - r/2, h)))
    plate3 = ((1, -1, 0), (x, y, 0), ((x - r/2, y - r/2, 0), (x + r/2, y + r/2, h)))

    return (plate1, plate2, plate3)

def inRange(p, r):
    '''
    p: point -> (x, y, z)
    r: range, 2 3D vectors
    '''
    (p1, p2, p3) = p
    (a, b) = r
    (a1, a2, a3) = a
    (b1, b2, b3) = b

    if(p1 > max(b1, a1) or p1 < min(b1, a1)):
        return False
    
    if(p2 > max(b2, a2) or p2 < min(b2, a2)):
        return False
    
    if(p3 > max(b3, a3) or p3 < min(b3, a3)):
        return False
    
    return True

def plateTouchPoint(a, b, plate):
    '''
    a, b: points -> (x, y, z);
    plate: (3D normal vector -> (x, y, z), point on the plate -> (x, y, z), range(2 3D vectors));
    '''
    # L: {A + tB} (A = (x1, y1, z1), B = (x2, y2, z2))
    # P: {N*T = C} where {T: (x, y, z)}
    # need to find a 'T' which belongs to P and to L, or find a 't' where {N*(A + tB) = C}
    # t = (C - A1*N1 - A2*N2 - A3*N3)/(B1*N1 + B2*N2 + B3*N3)
    (x1, y1, z1) = a
    (x2, y2, z2) = b
    (x2, y2, z2) = (x2 - x1, y2 - y1, z2 - z1)

    ((N1, N2, N3), (p1, p2, p3), range) = plate
    c = N1*p1 + N2*p2 + N3*p3
    
    if (x2*N1 + y2*N2 + z2*N3) == 0:
        return False
    
    t = (c - x1*N1 - y1*N2 - z1*N3)/(x2*N1 + y2*N2 + z2*N3)

    if(t > 1 or t < 0):
        return False
    
    touch = (x1 + x2*t, y1 + y2*t, z1 + z2*t)

    if(inRange(touch, range)):
        return True
    return False

def MountainSightCheck(a, b, mount):
    '''
    a, b: points -> (x, y, z);
    mount: (x, y, radius, height);
    '''
    plate = [0, 0, 0]
    (plate[0], plate[1], plate[2]) = platesCalc(mount)

    for p in plate:
        if(plateTouchPoint(a, b, p)):
            return False
    
    return True

def point_in_mountain(point, mountain):
    """
    Check if a 3D point lies inside the square base of a mountain (and under its height).
    mountain: (mx, my, side_length, height)
    """
    x, y, z = point
    id, mx, my, side, h = mountain
    half = side / 2
    if (mx - half) <= x <= (mx + half) and (my - half) <= y <= (my + half) and 0 <= z <= h:
        return True
    return False

def segment_intersects_mountain(a, b, mountain, eps=1e-9):
    """
    Check if the line segment from a to b intersects the axis‐aligned square prism of the mountain.
    Uses the slab method for AABB intersection:
      mountain: (mx, my, side_length, height)
    Returns True if any intersection.
    """
    ax, ay, az = a
    bx, by, bz = b
    id, mx, my, side, h = mountain
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

def can_place_sensor(point, mountain_list):
    """
    Return True if `point` is not inside any mountain in `mountain_list`.
    mountain_list: iterable of (mx, my, side_length, height)
    """
    return not any(point_in_mountain(point, m) for m in mountain_list)


def is_drone_path_valid(a, b, mountain_list):
    """
    Return True if the segment from a to b does not intersect any mountain in mountain_list.
    """
    return not any(segment_intersects_mountain(a, b, m) for m in mountain_list)

# example: MountainSightCheck((0, 0, 0), (100, 100, 20), (10, 0, 20, 10)) = False
