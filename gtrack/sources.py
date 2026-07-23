"""Source-cloud construction for downstream nearest-neighbour interpolation.

This module builds an *indicator source*: a set of rotated seed points (e.g. a
masked craton or continental field) unioned with a FRESH, uniform background
grid regenerated at the target age.

Why a fresh background instead of rotating a fixed present-day grid?
-------------------------------------------------------------------
Back-rotating a fixed present-day grid *deforms* it: the grid stretches open at
divergent boundaries and piles up at trenches, opening gaps that GROW with age.
For a "zero outside the region" indicator field that deformation is meaningless
and actively harmful — a downstream nearest-neighbour interpolator (with a large
distance threshold) can then find no nearby background point across a stretched
gap and reach across it to pull in a distant seed value. A fresh uniform grid
generated *at the target age* keeps the background dense and hole-free at any
age, so the interpolation stays well-posed. The fresh background is therefore a
correctness requirement, not a workaround for the (removed) rigid engine.

The rotated seeds themselves are advected topologically by ``PointRotator`` and
need no plate ids: the topological engine keeps every seed (including
out-of-circuit ones) and moves each by whatever plate or network it sits in.
No plate-id reattachment or sliver inheritance is required.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .mesh import create_sphere_mesh_latlon
from .point_rotation import PointCloud, PointRotator


def _unit(xyz: np.ndarray) -> np.ndarray:
    """Return unit vectors of the rows of ``xyz`` (safe against zero norm)."""
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(norms, 1e-30)


def build_indicator_source(
    seed_cloud: PointCloud,
    rotator: PointRotator,
    target_age: float,
    *,
    from_age: float = 0.0,
    background_n: int = 40000,
    background_value: float = 0.0,
    exclusion_factor: float = 1.0,
    time_step: float = 1.0,
) -> PointCloud:
    """Rotate seeds and union them with a fresh uniform background grid.

    Advects ``seed_cloud`` topologically from ``from_age`` to ``target_age``,
    then unions the rotated seeds with a FRESH uniform Fibonacci background grid
    generated at ``target_age``, dropping background points that land within one
    grid-spacing of a rotated seed (set-union collision removal). This guarantees
    a dense, hole-free background so a downstream nearest-neighbour interpolation
    always has nearby background points outside the seed regions, at ANY age.

    Motion is topological and needs no plate ids; no out-of-circuit reattachment
    or sliver-plate-id inheritance is performed (both are redundant under the
    topological engine).

    Parameters
    ----------
    seed_cloud : PointCloud
        Present-day (or ``from_age``) seed points with their properties. The
        seeds are typically the in-region points of a masked indicator field.
    rotator : PointRotator
        A topological rotator (built with ``topology_files``).
    target_age : float
        Geological age to rotate the seeds to and to build the background at (Ma).
    from_age : float, default=0.0
        Source geological age of ``seed_cloud`` (Ma).
    background_n : int, default=40000
        Number of points in the fresh uniform background grid.
    background_value : float, default=0.0
        Value assigned to every seed property on the background points.
    exclusion_factor : float, default=1.0
        Collision-removal radius as a multiple of the background grid spacing.
        A background point within ``exclusion_factor`` grid-spacings of the
        nearest rotated seed is dropped.
    time_step : float, default=1.0
        Internal stepping granularity for the topological reconstruction (Myr).

    Returns
    -------
    PointCloud
        ``concatenate([rotated_seeds, background])`` — rotated seeds first (their
        properties/plate_ids preserved in order), then the surviving background
        points (each seed property filled with ``background_value``; plate_ids 0
        if the seeds carry plate_ids, else absent).

    Notes
    -----
    Contrast with rotating a fixed present-day grid: that path deforms with age
    (gaps grow), which this helper deliberately avoids by regenerating a uniform
    grid at the target age.
    """
    if background_n <= 0:
        raise ValueError(f"background_n must be positive, got {background_n!r}")
    if exclusion_factor < 0:
        raise ValueError(
            f"exclusion_factor must be non-negative, got {exclusion_factor!r}"
        )

    # 1. Rotate the seeds topologically (keeps every seed; needs no plate ids).
    rotated = rotator.rotate(
        seed_cloud, from_age=from_age, to_age=target_age, time_step=time_step
    )

    # 2. Fresh uniform background grid AT the target age.
    lats, lons = create_sphere_mesh_latlon(background_n)
    background = PointCloud.from_latlon(np.column_stack([lats, lons]))
    for name in rotated.properties:
        background.add_property(
            name, np.full(background.n_points, background_value, dtype=float)
        )
    background.plate_ids = (
        np.zeros(background.n_points, dtype=int)
        if rotated.plate_ids is not None
        else None
    )

    # 3. Collision removal: drop background points within one grid-spacing (times
    #    exclusion_factor) of the nearest rotated seed. Distances are chord
    #    lengths on the unit sphere.
    if rotated.n_points > 0 and background.n_points > 0 and exclusion_factor > 0:
        from scipy.spatial import cKDTree

        theta = exclusion_factor * np.sqrt(4.0 * np.pi / background_n)  # rad
        chord = 2.0 * np.sin(theta / 2.0)                              # unit sphere
        tree = cKDTree(_unit(rotated.xyz))
        dist, _ = tree.query(_unit(background.xyz), k=1)
        background = background.subset(dist > chord)

    # 4. Union: rotated seeds first, then background (order/lockstep preserved).
    return PointCloud.concatenate([rotated, background], warn=False)
