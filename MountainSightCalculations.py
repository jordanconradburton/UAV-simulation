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

# example: MountainSightCheck((0, 0, 0), (100, 100, 20), (10, 0, 20, 10)) = False
