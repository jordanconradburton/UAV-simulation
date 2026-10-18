"""
variants/mountain_sampling.py

The geometry, sampling and masking used by `add-mountains`, which adds
mountains to an already-generated mountain-free dataset without re-simulating.
`inject_mountains.py` is the driver that reads and writes the dataset; this
module draws the mountains and works out what they block.

By construction we only ever accept mountains that would not have deflected
the drone. A real mountains-on run also contains trajectories that do get
deflected (a hover step where the drone would otherwise have flown through a
mountain). Those trajectories cannot be reproduced here without changing the
drone positions.

This variant is valid for single-drone datasets only.
"""
import hashlib
import math
import random

import numpy as np

# The simulator's mountain defaults.
MOUNTAIN_COUNT_RANGE = (1, 5)
MOUNTAIN_SIDE_RANGE = (50.0, 1000.0)
MOUNTAIN_HEIGHT_RANGE = (50.0, 500.0)

# Fallback sensor and drone-start ceilings, the simulator's defaults.
SENSOR_PLACEMENT_HEIGHT = 100.0
TARGET_HEIGHT = 5000.0

SLAB_EPS = 1e-9                  # segment direction below this counts as parallel to an axis
MAX_PLACEMENT_ATTEMPTS = 10_000  # per mountain, before the region is declared too small


def mountains_to_array(mountains):
    """
    (x, y, side_length, height) tuples -> float64 [M, 4].

    Accepts either the 4-tuple form used in this module or the DB's 5-tuple
    (id, x, y, side, height) form returned by
    DatabaseManager.get_mountains_by_experiment, so callers can feed
    simulator rows straight in.
    """
    rows = []
    for m in mountains:
        if len(m) == 5:
            _, x, y, side, h = m
        else:
            x, y, side, h = m
        rows.append((x, y, side, h))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def points_in_mountains(points, mountains):
    """
    Vectorized core.mountains.point_in_mountain, OR-ed over every
    mountain. points: [P, 3]; returns bool [P].

    Bounds are inclusive on every axis and z is tested against [0, height],
    exactly as the scalar version does.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    marr = mountains_to_array(mountains)
    if len(marr) == 0 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)

    mx, my, side, h = marr[:, 0], marr[:, 1], marr[:, 2], marr[:, 3]
    half = side / 2.0
    x, y, z = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]          # [P, 1]
    inside = (
        (x >= (mx - half)) & (x <= (mx + half)) &
        (y >= (my - half)) & (y <= (my + half)) &
        (z >= 0.0) & (z <= h)
    )                                                          # [P, M]
    return inside.any(axis=1)


def segments_blocked(a, b, mountains, eps=SLAB_EPS):
    """
    Vectorized core.mountains.segment_intersects_mountain, OR-ed
    over every mountain. a, b: [S, 3] segment endpoints; returns bool [S].
    """
    A = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    marr = mountains_to_array(mountains)
    if len(marr) == 0 or len(A) == 0:
        return np.zeros(len(A), dtype=bool)

    half = marr[:, 2] / 2.0
    mins = np.stack([marr[:, 0] - half, marr[:, 1] - half, np.zeros(len(marr))], axis=1)  # [M, 3]
    maxs = np.stack([marr[:, 0] + half, marr[:, 1] + half, marr[:, 3]], axis=1)          # [M, 3]

    t_enter = np.zeros((len(A), len(marr)))
    t_exit = np.ones((len(A), len(marr)))
    feasible = np.ones((len(A), len(marr)), dtype=bool)

    for k in range(3):
        ac = A[:, k:k + 1]                     # [S, 1]
        d = (B[:, k:k + 1] - ac)               # [S, 1]
        parallel = np.abs(d) < eps

        d_safe = np.where(parallel, 1.0, d)
        t0 = (mins[None, :, k] - ac) / d_safe  # [S, M]
        t1 = (maxs[None, :, k] - ac) / d_safe
        lo = np.where(parallel, 0.0, np.minimum(t0, t1))
        hi = np.where(parallel, 1.0, np.maximum(t0, t1))

        outside = parallel & ((ac < mins[None, :, k]) | (ac > maxs[None, :, k]))
        feasible &= ~outside
        t_enter = np.maximum(t_enter, lo)
        t_exit = np.minimum(t_exit, hi)

    hit = feasible & (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)
    return hit.any(axis=1)


def sensor_drone_blocked_matrix(sensor_xyz, drone_xyz, mountains):
    """
    [T, N] bool: is the LOS from sensor n to the drone's position at
    timestep t blocked? sensor_xyz: [N, 3] (static over the experiment),
    drone_xyz: [T, 3].
    """
    sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64).reshape(-1, 3)
    drone_xyz = np.asarray(drone_xyz, dtype=np.float64).reshape(-1, 3)
    T, N = len(drone_xyz), len(sensor_xyz)
    if T == 0 or N == 0:
        return np.zeros((T, N), dtype=bool)
    a = np.repeat(sensor_xyz[None, :, :], T, axis=0).reshape(-1, 3)
    b = np.repeat(drone_xyz[:, None, :], N, axis=1).reshape(-1, 3)
    return segments_blocked(a, b, mountains).reshape(T, N)


def sensor_pair_blocked(sensor_xyz, pairs, mountains):
    """
    bool [len(pairs)]: is the segment between the two sensors of each
    (i, j) pair blocked?
    """
    sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64).reshape(-1, 3)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros(0, dtype=bool)
    return segments_blocked(sensor_xyz[pairs[:, 0]], sensor_xyz[pairs[:, 1]], mountains)


# ---------------------------------------------------------------------------
# Mountain sampling
# ---------------------------------------------------------------------------
def _rects_overlap(r1, r2):
    """Same test as core.environment._rects_overlap — strict inequalities, so
    exactly-touching footprints are allowed."""
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or
                r1[3] <= r2[1] or r1[1] >= r2[3])


def propose_mountain_set(rng, region_size, num_mountains):
    """
    One draw from the simulator's mountain prior.

    Returns a list of (x, y, side_length, height).
    """
    mountains = []
    existing = []
    for _ in range(num_mountains):
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            mx = rng.uniform(0, region_size[0])
            my = rng.uniform(0, region_size[1])
            side = rng.uniform(*MOUNTAIN_SIDE_RANGE)
            half = side / 2.0
            rect = (mx - half, my - half, mx + half, my + half)
            if not any(_rects_overlap(rect, other) for other in existing):
                existing.append(rect)
                break
        else:
            raise RuntimeError(
                f"could not place mountain {len(mountains) + 1}/{num_mountains} without "
                f"overlap after {MAX_PLACEMENT_ATTEMPTS} attempts in region {region_size} — "
                f"region too small for this many mountains of side "
                f"{MOUNTAIN_SIDE_RANGE}, refusing to spin forever")
        mountains.append((mx, my, side, rng.uniform(*MOUNTAIN_HEIGHT_RANGE)))
    return mountains


def _trajectory_admissible(mountains, seg_start, seg_end, hover_mask):
    """
    True if this mountain set is consistent with the trajectory we already
    have — i.e. simulate_random_trajectory, given these mountains, would
    have produced the same positions.
    """
    moving = ~hover_mask
    if moving.any() and segments_blocked(seg_start[moving], seg_end[moving], mountains).any():
        return False
    if hover_mask.any() and points_in_mountains(seg_start[hover_mask], mountains).any():
        return False
    return True


def _free_volume(mountains, region_size, ceiling):
    """
    Volume of the placement box [0,W] x [0,H] x [0,ceiling] still available
    once `mountains` are present.
    """
    w, h = region_size
    occluded = sum(s * s * min(mh, ceiling) for _, _, s, mh in mountains)
    return w * h * ceiling - occluded


def _log_accept_ratio(mountains, region_size, n_sensors, sensor_height, target_height):
    """
    log of the rejection-sampling acceptance probability w(m)/w_max, always <= 0.

    The simulator places mountains first and then draws each sensor and the
    drone's start uniformly, rejecting points inside a mountain. So the chance
    of the geometry we observed, given mountains m, is proportional to
    w(m) = v_sensor(m)^-n_sensors * v_start(m)^-1, where v is the free volume.
    w_max uses the most volume mountains could ever occupy.
    """
    k_max = MOUNTAIN_COUNT_RANGE[1]
    side_max = MOUNTAIN_SIDE_RANGE[1]
    height_max = MOUNTAIN_HEIGHT_RANGE[1]
    worst = [(0.0, 0.0, side_max, height_max)] * k_max

    v_sensor = _free_volume(mountains, region_size, sensor_height)
    v_start = _free_volume(mountains, region_size, target_height)
    v_sensor_min = _free_volume(worst, region_size, sensor_height)
    v_start_min = _free_volume(worst, region_size, target_height)

    return (n_sensors * (math.log(v_sensor_min) - math.log(v_sensor))
            + (math.log(v_start_min) - math.log(v_start)))


def sample_mountains_for_experiment(rng, region_size, sensor_xyz, seg_start, seg_end,
                                     hover_mask, num_mountains=None, max_attempts=100_000,
                                     sensor_height=SENSOR_PLACEMENT_HEIGHT,
                                     target_height=TARGET_HEIGHT,
                                     require_blocked_detection=False):
    """
    Draw the mountain set for one already-simulated experiment, from the
    simulator's own prior conditioned on the geometry we already have.

    Rejection stages:
      1. no sensor inside a mountain
      2. the trajectory unaffected
      3. the importance correction (_log_accept_ratio)
      4. only with require_blocked_detection: at least one detection blocked.
         This changes the distribution, so it is not faithful to the simulator.

    Returns (mountains, number of proposals drawn).
    """
    lo, hi = MOUNTAIN_COUNT_RANGE
    sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64).reshape(-1, 3)
    n_sensors = len(sensor_xyz)

    for attempt in range(1, max_attempts + 1):
        k = num_mountains if num_mountains is not None else rng.randint(lo, hi)
        mountains = propose_mountain_set(rng, region_size, k)

        if points_in_mountains(sensor_xyz, mountains).any():
            continue
        if not _trajectory_admissible(mountains, seg_start, seg_end, hover_mask):
            continue
        if math.log(rng.random()) > _log_accept_ratio(
                mountains, region_size, n_sensors, sensor_height, target_height):
            continue
        if require_blocked_detection and not sensor_drone_blocked_matrix(
                sensor_xyz, seg_start, mountains).any():
            continue
        return mountains, attempt

    raise RuntimeError(
        f"no admissible mountain set after {max_attempts} attempts "
        f"(num_mountains={num_mountains}, require_blocked_detection={require_blocked_detection})"
    )


def experiment_rng(*key_parts):
    """
    Deterministic, process-independent RNG for one experiment.
    """
    key = "|".join(str(p) for p in key_parts).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Applying the mask to exported artifacts
# ---------------------------------------------------------------------------
def apply_blocked_mask_to_graphs(exp_graphs, blocked_tn, theta_power_slice, phi_power_slice):
    """
    Zero the per-sector power columns of every (timestep, sensor) whose LOS
    is blocked. Mutates each Data object's x in place. phi_power_slice is None
    for a dataset without elevation sectors.

    Returns the number of (timestep, sensor) detections removed.
    """
    import torch

    removed = 0
    for t, g in enumerate(exp_graphs):
        mask = torch.from_numpy(blocked_tn[t])
        if not bool(mask.any()):
            continue

        had = (g.x[:, theta_power_slice] != 0).any(dim=1)
        removed += int((mask & had).sum())
        g.x[mask, theta_power_slice] = 0.0
        if phi_power_slice is not None:
            g.x[mask, phi_power_slice] = 0.0
    return removed


def filter_edges(edge_index, edge_attr, blocked_pair):
    """
    Drop both directions of every mountain-blocked sensor pair.
    """
    import torch

    keep_pairs = np.repeat(~blocked_pair, 2)
    if not keep_pairs.any():
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float))
    keep = torch.from_numpy(keep_pairs)
    return edge_index[:, keep].contiguous(), edge_attr[keep].contiguous()
