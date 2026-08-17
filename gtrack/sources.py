"""Build source-point clouds for indicator interpolation.

Each indicator source contains rotated source points and a uniform background.
The module creates a new background at each requested geological age.
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
    """Return row-wise unit vectors with a floor for zero norms."""
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(norms, 1e-30)


def build_indicator_source(
    source_cloud: PointCloud,
    rotator: PointRotator,
    target_age: float,
    *,
    from_age: float = 0.0,
    background_point_count: int = 40000,
    background_value: float = 0.0,
    exclusion_factor: float = 1.0,
    rotation_step_myr: float = 1.0,
    membership_property: Optional[str] = "membership",
) -> PointCloud:
    """Rotate source points and add a uniform background.

    Parameters
    ----------
    source_cloud : PointCloud
        Source points and their channels at ``from_age``.
    rotator : PointRotator
        A topological rotator (built with ``topology_files``).
    target_age : float
        Geological age for the output source, in Ma.
    from_age : float, default=0.0
        Geological age of ``source_cloud``, in Ma.
    background_point_count : int, default=40000
        Number of points in the background distribution.
    background_value : float, default=0.0
        Value assigned to each source channel on the background points.
    exclusion_factor : float, default=1.0
        Collision-removal radius as a multiple of the background grid spacing.
        A background point inside this radius from a source point is removed.
    rotation_step_myr : float, default=1.0
        Step for the topological rotation, in Myr.
    membership_property : str or None, default="membership"
        Name of the membership channel. Source points receive one. Background
        points receive zero. Use ``None`` to omit the channel.

    Returns
    -------
    PointCloud
        Rotated source points followed by the remaining background points.
    """
    if background_point_count <= 0:
        raise ValueError(
            "background_point_count must be positive, "
            f"got {background_point_count!r}"
        )
    if exclusion_factor < 0:
        raise ValueError(
            f"exclusion_factor must be non-negative, got {exclusion_factor!r}"
        )

    rotated = rotator.rotate(
        source_cloud,
        from_age=from_age,
        to_age=target_age,
        time_step=rotation_step_myr,
    )

    # Mark the source points as members before the background is created.
    if membership_property is not None:
        rotated.add_property(
            membership_property, np.ones(rotated.n_points, dtype=float)
        )

    # Create a uniform background at the target age.
    lats, lons = create_sphere_mesh_latlon(background_point_count)
    background = PointCloud.from_latlon(np.column_stack([lats, lons]))
    for name in rotated.properties:
        background.add_property(
            name, np.full(background.n_points, background_value, dtype=float)
        )
    if membership_property is not None:
        background.add_property(
            membership_property, np.zeros(background.n_points, dtype=float)
        )
    background.plate_ids = (
        np.zeros(background.n_points, dtype=int)
        if rotated.plate_ids is not None
        else None
    )

    # Remove background points near a rotated source point.
    if rotated.n_points > 0 and background.n_points > 0 and exclusion_factor > 0:
        from scipy.spatial import cKDTree

        theta = exclusion_factor * np.sqrt(
            4.0 * np.pi / background_point_count
        )
        chord = 2.0 * np.sin(theta / 2.0)                              # unit sphere
        tree = cKDTree(_unit(rotated.xyz))
        dist, _ = tree.query(_unit(background.xyz), k=1)
        background = background.subset(dist > chord)

    # Keep the rotated source points before the background points.
    return PointCloud.concatenate([rotated, background], warn=False)


# ---------------------------------------------------------------------------
# Polygon-bounded indicator source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolygonIndicatorConfig:
    """Configure :class:`PolygonIndicatorSource`.

    Parameters
    ----------
    background_point_count : int, default=20000
        Number of points in the background distribution for each age.
    scalar_input_point_count : int, default=20000
        Source-point count when ``thickness_data`` is a scalar.
    exclusion_factor : float, default=1.0
        Collision-removal radius as a multiple of the background grid spacing.

    """

    background_point_count: int = 20000
    scalar_input_point_count: int = 20000
    exclusion_factor: float = 1.0

    def __post_init__(self):
        if self.background_point_count <= 0:
            raise ValueError(
                "background_point_count must be positive, "
                f"got {self.background_point_count}"
            )
        if self.scalar_input_point_count <= 0:
            raise ValueError(
                "scalar_input_point_count must be positive, "
                f"got {self.scalar_input_point_count}"
            )
        if self.exclusion_factor < 0:
            raise ValueError(
                f"exclusion_factor must be non-negative, "
                f"got {self.exclusion_factor}"
            )


class PolygonIndicatorSource:
    """A polygon-bounded region as a labelled cloud at any age.

    Satisfies :class:`~gtrack.age_sources.AgeCloudSource`.

    The source rotates the in-polygon source points to each requested age. It
    adds a new uniform background at that age.

    Parameters
    ----------
    rotation_files : list of str or Path
        Rotation files for the plate model.
    topology_files : list of str or Path
        Topology files for the plate model.
    polygons : str or Path or list
        Polygon files that bound the region, for example, a shapefile or GPML.
    static_polygons : str or Path
        Static plate polygons, used by the point rotator.
    thickness_data : PointCloud or str or Path or tuple or int or float
        Thickness carried by source points inside the polygons, in anything
        :meth:`~gtrack.point_rotation.PointCloud.from_data` accepts.
    config : PolygonIndicatorConfig, optional
        Point-count and collision parameters.

    Attributes
    ----------
    provides : frozenset of str
        ``{"masked_thickness", "membership"}``. Coordinates are carried by the
        cloud itself and are deliberately not listed.
    monotonic_backward : bool
        False. There is no walk and no state, so ages can be requested in any
        order, repeatedly, and the answer depends only on the age asked for.

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

    #: Name of the masked-thickness channel.
    PROPERTY_NAME = "masked_thickness"

    #: Name of the membership channel.
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

        # Delay plate-model I/O until the first source request.
        self._polygon_filter = None
        self._rotator = None
        self._source_points = None

    def _ensure_loaded(self) -> None:
        """Build the polygon filter, rotator, and source points once."""
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
        self._source_points = self._build_source_points()

    def _build_source_points(self) -> PointCloud:
        """Load the thickness channel and keep the in-polygon source points."""
        cloud = PointCloud.from_data(
            self._thickness_data,
            self.PROPERTY_NAME,
            n_points_fallback=self.config.scalar_input_point_count,
        )
        n_before = cloud.n_points
        source_points = self._polygon_filter.filter_inside(cloud, at_age=0.0)
        logger.debug(
            "kept %d in-polygon source points out of %d present-day points",
            source_points.n_points,
            n_before,
        )
        return source_points

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` is negative.

        This source has no tracker state or known plate-model age limit.
        It accepts non-negative ages in any order.

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
            Rotated in-region source points followed by the background,
            carrying ``masked_thickness`` and ``membership``.
        """
        self.validate_age(age)
        self._ensure_loaded()

        cloud = build_indicator_source(
            self._source_points,
            self._rotator,
            target_age=age,
            background_point_count=self.config.background_point_count,
            background_value=0.0,
            exclusion_factor=self.config.exclusion_factor,
            membership_property=self.MEMBERSHIP_PROPERTY,
        )
        logger.debug("%d source points with a fresh background at %.2f Ma",
                     cloud.n_points, age)
        return cloud
