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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging import get_logger
from .mesh import create_sphere_mesh_latlon
from .point_rotation import PointCloud, PointRotator

logger = get_logger(__name__)


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


# ---------------------------------------------------------------------------
# Polygon-bounded indicator source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolygonIndicatorConfig:
    """Knobs for :class:`PolygonIndicatorSource`.

    Parameters
    ----------
    background_n : int, default=20000
        Number of points in the fresh uniform background grid built at each
        age. This is not only a cost knob: it sets the collision-removal
        radius, ``exclusion_factor * sqrt(4*pi / background_n)`` radians, so a
        coarser background deletes background points further out from every
        seed and leaves an annulus around the region holding no background at
        all. A downstream interpolator's membership estimate is pulled towards
        1 in that annulus, so the region is effectively DILATED by roughly that
        radius:

            background_n =  5000  ->  0.050 rad  ~ 320 km on Earth
            background_n = 20000  ->  0.025 rad  ~ 160 km
            background_n = 40000  ->  0.018 rad  ~ 113 km

        The dilation is geometric — a hole in the point cloud — so sharpening
        the interpolation kernel does not shrink it. Match this to the seed
        spacing unless you are coarsening deliberately, and keep the resulting
        radius below the resolution of whatever consumes the cloud.
    seed_fallback_n : int, default=20000
        Sphere-mesh size used ONLY when ``thickness_data`` is a scalar and has
        to be broadcast somewhere. Every other input brings its own points and
        ignores this. It has nothing to do with ``background_n`` and does not
        affect the dilation above.
    exclusion_factor : float, default=1.0
        Collision-removal radius as a multiple of the background grid spacing.

    Notes
    -----
    ``background_n`` and ``seed_fallback_n`` are separate fields because they
    are separate quantities that happened to share one knob. One sizes a grid
    rebuilt at every age and controls how far the region is dilated; the other
    is a fallback that is inert unless the thickness input is a bare number.
    A single knob for both means tuning the dilation silently reshapes a
    scalar-seeded cloud, and vice versa.
    """

    background_n: int = 20000
    seed_fallback_n: int = 20000
    exclusion_factor: float = 1.0

    def __post_init__(self):
        if self.background_n <= 0:
            raise ValueError(
                f"background_n must be positive, got {self.background_n}"
            )
        if self.seed_fallback_n <= 0:
            raise ValueError(
                f"seed_fallback_n must be positive, got {self.seed_fallback_n}"
            )
        if self.exclusion_factor < 0:
            raise ValueError(
                f"exclusion_factor must be non-negative, "
                f"got {self.exclusion_factor}"
            )


class PolygonIndicatorSource:
    """A polygon-bounded region as a labelled cloud at any age.

    Satisfies :class:`~gtrack.age_sources.AgeCloudSource`.

    Keeps only the in-polygon seeds at present day, each carrying the
    data-derived thickness, then at every requested age rotates those seeds
    topologically and unions them with a fresh uniform background grid built at
    that age, via :func:`build_indicator_source`.

    Parameters
    ----------
    rotation_files : list of str or Path
        Rotation files for the plate model.
    topology_files : list of str or Path
        Topology files for the plate model.
    polygons : str or Path or list
        Polygon file(s) bounding the region, e.g. a shapefile or gpml.
    static_polygons : str or Path
        Static plate polygons, used by the point rotator.
    thickness_data : PointCloud or str or Path or tuple or int or float
        Thickness carried by the seeds inside the polygons, in anything
        :meth:`~gtrack.point_rotation.PointCloud.from_data` accepts.
    config : PolygonIndicatorConfig, optional
        Background, fallback and collision knobs.

    Attributes
    ----------
    provides : frozenset of str
        ``{"masked_thickness", "membership"}``. Coordinates are carried by the
        cloud itself and are deliberately not listed.
    monotonic_backward : bool
        False. There is no walk and no state, so ages may be requested in any
        order, repeatedly, and the answer depends only on the age asked for.

    Notes
    -----
    The thickness channel is called ``masked_thickness`` rather than
    ``thickness``, and the distinction is not cosmetic. On a bounded source the
    channel holds the region's depth multiplied by its membership once the
    downstream blend has run, which is a different physical quantity from an
    unbounded thickness field. Giving it its own name lets a consumer's
    required-versus-provided check reject a pairing that would read it as a
    plain depth, instead of silently producing a field that is wrong near every
    region edge.

    ``membership`` is published alongside it for the same reason. A single
    channel holding thickness inside and zero outside is the product of two
    independent facts — where the region is, and how thick it is there — and an
    interpolator smooths the product rather than the factors. Blending
    membership on its own gives the local in-region weight fraction, which
    makes the two separable again.

    Construction is cheap and touches no reconstruction data; the polygon
    filter, rotator and present-day seeds are built on the first
    :meth:`at_age`.

    Examples
    --------
    >>> source = PolygonIndicatorSource(
    ...     rotation_files=rotations,
    ...     topology_files=topologies,
    ...     polygons='cratons.shp',
    ...     static_polygons=static_polys,
    ...     thickness_data=200.0,
    ... )
    >>> cloud = source.at_age(100.0)
    """

    provides = frozenset({"masked_thickness", "membership"})
    monotonic_backward = False

    #: Name the thickness is attached under. Fixed rather than configurable,
    #: because ``provides`` names it and the two must agree.
    PROPERTY_NAME = "masked_thickness"

    #: Name of the membership channel. Passed explicitly to
    #: ``build_indicator_source`` rather than left to its default, so the two
    #: sides of this seam are coupled by an argument rather than by two
    #: defaults that happen to match.
    MEMBERSHIP_PROPERTY = "membership"

    def __init__(
        self,
        rotation_files,
        topology_files,
        polygons,
        static_polygons,
        thickness_data,
        config: Optional[PolygonIndicatorConfig] = None,
    ):
        if polygons is None:
            raise ValueError("polygons is required")
        if static_polygons is None:
            raise ValueError("static_polygons is required")

        self.rotation_files = rotation_files
        self.topology_files = topology_files
        self.polygons = polygons
        self.static_polygons = static_polygons
        self.config = config or PolygonIndicatorConfig()

        self._thickness_data = thickness_data

        # Built on first use, never in __init__.
        self._polygon_filter = None
        self._rotator = None
        self._seeds = None

    def _ensure_loaded(self) -> None:
        """Build the polygon filter, rotator and present-day seeds once."""
        if self._rotator is not None:
            return
        from .polygon_filter import PolygonFilter

        self._polygon_filter = PolygonFilter(
            polygon_files=self.polygons,
            rotation_files=self.rotation_files,
        )
        self._rotator = PointRotator(
            rotation_files=self.rotation_files,
            topology_files=self.topology_files,
            static_polygons=self.static_polygons,
        )
        self._seeds = self._build_seeds()

    def _build_seeds(self) -> PointCloud:
        """Load the thickness data and keep only the in-polygon points.

        Only the real in-region seeds are kept. The background is rebuilt at
        every age by :func:`build_indicator_source`, so there is no present-day
        grid to mask and relabel, and no plate ids to assign: the topological
        rotator advects each seed by whatever plate or network contains it.
        """
        cloud = PointCloud.from_data(
            self._thickness_data,
            self.PROPERTY_NAME,
            n_points_fallback=self.config.seed_fallback_n,
        )
        n_before = cloud.n_points
        seeds = self._polygon_filter.filter_inside(cloud, at_age=0.0)
        logger.debug("kept %d in-polygon seeds out of %d present-day points",
                     seeds.n_points, n_before)
        return seeds

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` is negative.

        There is no walk and no plate-model range known here, so this is the
        only check available. Ages may otherwise be requested in any order.

        Parameters
        ----------
        age : float
            Geological age in Ma.

        Raises
        ------
        ValueError
            If ``age`` is negative.
        """
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future)."
            )

    def at_age(self, age: float) -> PointCloud:
        """Return the region cloud at ``age``.

        Parameters
        ----------
        age : float
            Geological age in Ma.

        Returns
        -------
        PointCloud
            Rotated in-region seeds followed by the fresh background,
            carrying ``masked_thickness`` and ``membership``.
        """
        self.validate_age(age)
        self._ensure_loaded()

        cloud = build_indicator_source(
            self._seeds,
            self._rotator,
            target_age=age,
            background_n=self.config.background_n,
            background_value=0.0,
            exclusion_factor=self.config.exclusion_factor,
            membership_property=self.MEMBERSHIP_PROPERTY,
        )
        logger.debug("%d source points (seeds + fresh background) at %.2f Ma",
                     cloud.n_points, age)
        return cloud
