"""
variants/elevation_schemes.py

Five ways to divide elevation angle into sectors.

A sensor reports which elevation band it heard the drone in. Where to put the
band boundaries is a design choice with real consequences for how well the
altitude can be recovered, so this module offers five schemes, and `rebin.py` 
converts a dataset between them exactly.

Every scheme has the same shape: six ascending edges [0, e1, e2, e3, e4, 90]
defining five bins over [0, 90) degrees, plus a separate catch-all bin for
angles below the sensor's horizon. Only the four interior boundaries vary —
the bin count and the below-horizon bin are fixed.

Each scheme also carries six center angles, used as a feature value so a model
can associate a bin index with a physical angle. Bins 2-6 take the midpoint of
their edges. Bin 1's center is -9.0 for every scheme: the below-horizon bin is
unbounded below in principle, so it has no real centroid, and -9.0 is chosen
only for even spacing with the others.

    A_uniform_angle    equal 18-degree slices. The default.
    B_uniform_sin      equal steps in sin(elevation) — concentrates resolution 
                       near the horizon, and it collapses the near-zenith region 
                       into one wide sector.
    C_uniform_tan      equal altitude bands up to a cap, with the final bin
                       left as a coarse overflow.
    D_quantile         equal occupancy under the simulator's own geometry.
    E_fisher_optimal   maximizes Fisher information about elevation.

A and B are closed-form and computed here from their definitions. C, D and E
were derived from a reference set of simulated geometries and are recorded as
constants.
"""
import math

N_BINS = 5           # bins over [0,90); the below-horizon catch-all is a 6th, separate bin
BELOW_HORIZON_CENTER_DEG = -9.0


def _centers_from_edges(edges_deg):
    """[-9.0] + midpoints of consecutive (e_i, e_{i+1}) pairs -> 6-element center list."""
    mids = [(edges_deg[i] + edges_deg[i + 1]) / 2.0 for i in range(len(edges_deg) - 1)]
    return [BELOW_HORIZON_CENTER_DEG] + mids


def uniform_angle_edges():
    """Scheme A (current design): K equal 18-degree slices of [0,90]."""
    step = 90.0 / N_BINS
    return [round(k * step, 6) for k in range(N_BINS + 1)]


def uniform_sin_edges():
    """Scheme B: sin(edge) uniform in [0,1]."""
    return [round(math.degrees(math.asin(k / N_BINS)), 6) for k in range(N_BINS + 1)]


# --- Final schemes: A/B computed here, C/D/E recorded as constants ---
_A_EDGES = uniform_angle_edges()
_B_EDGES = uniform_sin_edges()

# C: tan(edge) evenly spaced from 0 up to a cap of 76.835245 degrees, the 95th
# percentile elevation angle of the reference geometries, so the first four
# bins are equal altitude bands and [cap, 90] is the coarse overflow bin. The
# edges were computed from the unrounded cap.
_C_EDGES = [0.0, 46.905726, 64.929901, 72.67911, 76.835245, 90.0]
_D_EDGES = [0.0, 18.605951, 34.041009, 47.696634, 60.605912, 90.0]
_E_EDGES = [0.0, 19.032199, 34.622841, 47.447162, 60.079496, 90.0]

SCHEME_BOUNDARIES = {
    "A_uniform_angle": (_A_EDGES, _centers_from_edges(_A_EDGES)),
    "B_uniform_sin": (_B_EDGES, _centers_from_edges(_B_EDGES)),
    "C_uniform_tan": (_C_EDGES, _centers_from_edges(_C_EDGES)),
    "D_quantile": (_D_EDGES, _centers_from_edges(_D_EDGES)),
    "E_fisher_optimal": (_E_EDGES, _centers_from_edges(_E_EDGES)),
}
