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

#: Ways a source can move its seed points back in time.
#: "topological" advects every seed with the plate or network that contains it at
#: each time step. "rigid_plate_id" rotates every seed with the total rotation of
#: the plate ID of the continent polygon that contains it at the present day.
ROTATION_MODES = ("topological", "rigid_plate_id")

#: What the rigid mode does with a seed whose polygon plate ID is not in the rotation
#: tree at the target age (or is 0). "in_place" keeps the seed where it is, as
#: ``pygplates.reconstruct`` does for the polygon. "topological_plate" rotates it
#: with the plate that contains it at 0 Ma in the topological model.
MISSING_PLATE_POLICIES = ("in_place", "topological_plate")

#: Tolerance in Myr for comparing an age with the begin age of a polygon.
#: pygplates compares times with an epsilon of 1e-9, and callers hand over ages
#: that carry float round-off, so an exact comparison would remove seeds that
#: ``pygplates.reconstruct`` keeps.
BEGIN_AGE_TOLERANCE_MYR = 1e-9


def _unit(xyz: np.ndarray) -> np.ndarray:
    """Return row-wise unit vectors with a floor for zero norms."""
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(norms, 1e-30)


def _rotate_rigid_by_plate_id(
    cloud: PointCloud,
    rotation_model,
    plate_ids: np.ndarray,
    begin_ages: np.ndarray,
    from_age: float,
    to_age: float,
    fallback_plate_ids: Optional[np.ndarray] = None,
    stats: Optional[dict] = None,
) -> PointCloud:
    """Rotate points rigidly with the rotation chain of a plate ID each.

    This applies to every point the rule that ``pygplates.reconstruct`` applies to
    a feature with a reconstruction plate ID: the total rotation of that plate ID
    from the present day to ``to_age``, taken from the reconstruction tree at
    ``to_age`` only. A plate ID that is not in the tree at ``to_age`` gets the
    identity rotation, so its points stay where they are. The continent polygon
    mask in ``boundaries.py`` follows the same rule, so rotated points and
    reconstructed polygons agree.

    A point whose polygon is not valid at ``to_age`` (``to_age`` older than the
    polygon's begin age) is removed, because ``pygplates.reconstruct`` does not
    reconstruct such a polygon either.

    If ``fallback_plate_ids`` is given, a point whose plate ID is not in the tree
    at ``to_age`` (or is 0) is rotated with its fallback plate ID instead. The
    choice is made again at every age, so a point follows its own plate while
    that plate has a rotation sequence and the fallback plate otherwise. A
    fallback plate that is also missing leaves the point in place.

    The two-argument form ``get_rotation(to_age, plate_id)`` is used on purpose.
    The three-argument stage form builds a second tree at ``from_age`` and returns
    the identity if the plate is missing from either tree, which drops the motion
    of plates that are not in the rotation tree at ``from_age``.

    Parameters
    ----------
    cloud : PointCloud
        Points at ``from_age``, which must be the present day.
    rotation_model : pygplates.RotationModel
        Rotation model whose default anchor plate is the reference frame.
    plate_ids : np.ndarray
        Plate ID per point, shape ``(N,)``. Plate ID 0 counts as missing.
    begin_ages : np.ndarray
        Begin of the valid time, in Ma, of the polygon that gave each plate ID,
        shape ``(N,)``. ``inf`` marks a polygon that exists in the distant past.
    from_age : float
        Age of ``cloud`` in Ma. Must be zero within ``ZERO_SPAN_TOLERANCE_MYR``.
    to_age : float
        Target age in Ma.
    fallback_plate_ids : np.ndarray, optional
        Plate ID per point, shape ``(N,)``, used where ``plate_ids`` is missing at
        ``to_age``. Without it, such points stay in place.
    stats : dict, optional
        Filled with the counts ``n_kept``, ``n_in_place``, ``n_fallback`` and
        ``n_removed`` of this call.

    Returns
    -------
    PointCloud
        The rotated points that are valid at ``to_age``, in input order, with
        their properties subset to match.
    """
    from .geometry import LatLon2XYZ, XYZ2LatLon
    from .point_rotation import ZERO_SPAN_TOLERANCE_MYR, _rotate_points_batch

    plate_ids = np.asarray(plate_ids)
    begin_ages = np.asarray(begin_ages, dtype=float)
    if fallback_plate_ids is not None:
        fallback_plate_ids = np.asarray(fallback_plate_ids)
        if fallback_plate_ids.shape != (cloud.n_points,):
            raise ValueError(
                "seed_fallback_plate_ids must have one entry per source point "
                f"({cloud.n_points}); got shape {fallback_plate_ids.shape}"
            )
    if plate_ids.shape != (cloud.n_points,) or begin_ages.shape != (cloud.n_points,):
        raise ValueError(
            "seed_plate_ids and seed_begin_ages must each have one entry per "
            f"source point ({cloud.n_points}); got shapes {plate_ids.shape} and "
            f"{begin_ages.shape}"
        )
    if abs(from_age) >= ZERO_SPAN_TOLERANCE_MYR:
        raise ValueError(
            'rotation="rigid_plate_id" rotates from the present day, so from_age '
            f"must be 0, got {from_age!r}"
        )

    lats, lons = XYZ2LatLon(cloud.xyz)
    new_lats = lats.copy()
    new_lons = lons.copy()

    # A polygon older than its begin age does not exist yet. The comparison is
    # inclusive and has a tolerance, as in pygplates.reconstruct.
    valid = to_age <= begin_ages + BEGIN_AGE_TOLERANCE_MYR

    def total_rotation(plate_id):
        """Total rotation of a plate ID at ``to_age``, or None if it has none.

        With use_identity_for_missing_plate_ids=False, pygplates returns None for
        a plate that is not in the reconstruction tree at ``to_age``. Plate ID 0
        is the anchor plate, which is no polygon plate, so it counts as missing.
        """
        if int(plate_id) == 0:
            return None
        return rotation_model.get_rotation(
            to_age, int(plate_id), use_identity_for_missing_plate_ids=False
        )

    n_in_place = 0
    n_fallback = 0
    needs_fallback = np.zeros(cloud.n_points, dtype=bool)

    # One rotation per plate ID, applied to all of that plate's points at once.
    for plate_id in np.unique(plate_ids[valid]):
        on_plate = valid & (plate_ids == plate_id)
        rotation = total_rotation(plate_id)
        if rotation is None:
            # Without a fallback the points keep their position, which is the
            # identity rotation that pygplates.reconstruct applies to the polygon.
            if fallback_plate_ids is None:
                n_in_place += int(on_plate.sum())
            else:
                needs_fallback |= on_plate
            continue
        new_lats[on_plate], new_lons[on_plate] = _rotate_points_batch(
            lats[on_plate], lons[on_plate], rotation
        )

    # Points whose own plate is missing follow their fallback plate. A fallback
    # plate that is missing too leaves the point in place.
    if needs_fallback.any():
        for plate_id in np.unique(fallback_plate_ids[needs_fallback]):
            on_plate = needs_fallback & (fallback_plate_ids == plate_id)
            rotation = total_rotation(plate_id)
            if rotation is None:
                n_in_place += int(on_plate.sum())
                continue
            n_fallback += int(on_plate.sum())
            new_lats[on_plate], new_lons[on_plate] = _rotate_points_batch(
                lats[on_plate], lons[on_plate], rotation
            )

    n_kept = int(valid.sum())
    n_removed = int((~valid).sum())
    if stats is not None:
        stats.update(n_kept=n_kept, n_in_place=n_in_place,
                     n_fallback=n_fallback, n_removed=n_removed)
    logger.info(
        "rigid_plate_id at %.2f Ma: %d source points kept, %d of them in place "
        "(plate not in the rotation tree), %d of them on the fallback plate, "
        "%d removed (polygon not yet valid)",
        to_age, n_kept, n_in_place, n_fallback, n_removed,
    )

    new_xyz = LatLon2XYZ(np.column_stack([new_lats[valid], new_lons[valid]]))
    return PointCloud(
        xyz=new_xyz,
        properties={k: v[valid].copy() for k, v in cloud.properties.items()},
        plate_ids=(cloud.plate_ids[valid].copy()
                   if cloud.plate_ids is not None else None),
    )


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
    rotation: str = "topological",
    seed_plate_ids: Optional[np.ndarray] = None,
    seed_begin_ages: Optional[np.ndarray] = None,
    seed_fallback_plate_ids: Optional[np.ndarray] = None,
    rigid_stats: Optional[dict] = None,
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
        Geological age of ``source_cloud``, in Ma. The rigid mode requires zero.
    background_point_count : int, default=40000
        Number of points in the background distribution.
    background_value : float, default=0.0
        Value assigned to each source channel on the background points.
    exclusion_factor : float, default=1.0
        Collision-removal radius as a multiple of the background grid spacing.
        A background point inside this radius from a source point is removed.
    rotation_step_myr : float, default=1.0
        Step for the topological rotation, in Myr. The rigid mode ignores it.
    membership_property : str or None, default="membership"
        Name of the membership channel. Source points receive one. Background
        points receive zero. Use ``None`` to omit the channel.
    rotation : {"topological", "rigid_plate_id"}, default="topological"
        How the source points move. ``"topological"`` calls ``rotator.rotate``.
        ``"rigid_plate_id"`` rotates each point with the rotation chain of
        ``seed_plate_ids`` (see ``_rotate_rigid_by_plate_id``) and removes the
        points whose polygon is not valid at ``target_age``.
    seed_plate_ids : np.ndarray, optional
        Plate ID per source point. Required for ``rotation="rigid_plate_id"``.
    seed_begin_ages : np.ndarray, optional
        Begin age in Ma of each point's polygon. Required for
        ``rotation="rigid_plate_id"``.
    seed_fallback_plate_ids : np.ndarray, optional
        Plate ID per source point to use where ``seed_plate_ids`` is missing from
        the rotation tree at ``target_age``. Used only with
        ``rotation="rigid_plate_id"``.
    rigid_stats : dict, optional
        Filled with the point counts of the rigid rotation, as in
        ``_rotate_rigid_by_plate_id``. Used only with ``rotation="rigid_plate_id"``.

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

    if rotation not in ROTATION_MODES:
        raise ValueError(
            f"rotation must be one of {ROTATION_MODES}, got {rotation!r}"
        )

    if rotation == "topological":
        rotated = rotator.rotate(
            source_cloud,
            from_age=from_age,
            to_age=target_age,
            time_step=rotation_step_myr,
        )
    else:
        if seed_plate_ids is None or seed_begin_ages is None:
            raise ValueError(
                'rotation="rigid_plate_id" requires seed_plate_ids and '
                "seed_begin_ages"
            )
        rotated = _rotate_rigid_by_plate_id(
            source_cloud,
            rotator.rotation_model,
            seed_plate_ids,
            seed_begin_ages,
            from_age,
            target_age,
            fallback_plate_ids=seed_fallback_plate_ids,
            stats=rigid_stats,
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
    rotation : {"topological", "rigid_plate_id"}, default="topological"
        How the in-polygon source points move back in time.

        ``"topological"`` advects each point with the resolved plate or network
        that contains it at every 1 Myr step. A point near a plate boundary can
        change plate and then follow a neighbouring oceanic plate.

        ``"rigid_plate_id"`` gives each point the plate ID of the polygon that
        contains it at the present day, and rotates it with the total rotation of
        that plate ID. The points then move like the polygons that
        ``pygplates.reconstruct`` rotates. A point whose plate ID has no rotation
        sequence at the target age stays in place with the default policy, as its polygon
        does (see ``rigid_missing_plate``). A point
        is removed at ages older than the begin age of its polygon. This mode
        ignores deforming networks.
    rigid_missing_plate : {"in_place", "topological_plate"}, default="in_place"
        What ``rotation="rigid_plate_id"`` does with a point whose plate ID is not
        in the rotation tree at the target age, or is 0.

        ``"in_place"`` leaves the point where it is, as its polygon is left by
        ``pygplates.reconstruct``. The mode then needs polygons with non-zero
        plate IDs.

        ``"topological_plate"`` rotates the point with the plate that contains it
        at 0 Ma in the topological model, whenever its own plate ID is missing.
        The point follows its own plate while that plate has a rotation sequence.
        This also serves polygon files without plate IDs, such as shapefiles. A
        point can then follow a plate other than the one of its polygon.

    """

    background_point_count: int = 20000
    scalar_input_point_count: int = 20000
    exclusion_factor: float = 1.0
    rotation: str = "topological"
    rigid_missing_plate: str = "in_place"

    def __post_init__(self):
        if self.rotation not in ROTATION_MODES:
            raise ValueError(
                f"rotation must be one of {ROTATION_MODES}, got {self.rotation!r}"
            )
        if self.rigid_missing_plate not in MISSING_PLATE_POLICIES:
            raise ValueError(
                f"rigid_missing_plate must be one of {MISSING_PLATE_POLICIES}, "
                f"got {self.rigid_missing_plate!r}"
            )
        if self.rotation == "topological" and self.rigid_missing_plate != "in_place":
            raise ValueError(
                'rigid_missing_plate only applies to rotation="rigid_plate_id"'
            )
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
        Point-count and collision parameters, and the rotation mode
        (``rotation="topological"`` by default, or ``"rigid_plate_id"``).

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

        # Plate ID and polygon begin age of every source point. Filled only for
        # rotation="rigid_plate_id". They are kept off the cloud so that the
        # emitted channels stay exactly the ones in ``provides``.
        self._seed_plate_ids = None
        self._seed_begin_ages = None
        self._seed_fallback_plate_ids = None

        # Point counts of the latest rigid rotation, and whether the one-time
        # warning about points that stay in place has been logged.
        self._rigid_stats = {}
        self._warned_in_place = False

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
        if self.config.rotation == "rigid_plate_id":
            (
                self._seed_plate_ids,
                self._seed_begin_ages,
                self._seed_fallback_plate_ids,
            ) = self._partition_seeds(self._source_points)

    def _partition_seeds(self, seeds: PointCloud):
        """Find the polygon that contains each seed at the present day.

        Parameters
        ----------
        seeds : PointCloud
            Source points, all inside the polygons at 0 Ma.

        Returns
        -------
        plate_ids : np.ndarray
            Reconstruction plate ID of each seed's polygon, shape ``(N,)``.
        begin_ages : np.ndarray
            Begin of the valid time of each seed's polygon in Ma, shape ``(N,)``.
        fallback_plate_ids : np.ndarray or None
            Plate ID of the topological plate that contains each seed at 0 Ma,
            shape ``(N,)``. None unless ``rigid_missing_plate="topological_plate"``.

        Raises
        ------
        ValueError
            If a seed lies in no polygon at 0 Ma. With ``rigid_missing_plate=
            "in_place"`` also if a polygon has plate ID 0, which is the anchor
            plate and would make the seed stationary at every age. Shapefile
            polygons often carry no plate ID. With ``"topological_plate"`` also if
            a seed lies in no topological plate at 0 Ma.
        """
        import pygplates

        # PlatePartitioner only sees the features valid at the reconstruction
        # time, so the polygons here are exactly the ones that contain seeds at
        # 0 Ma. Where polygons overlap it returns one of them by its own rule.
        partitioner = pygplates.PlatePartitioner(
            self._polygon_filter.polygon_features,
            self._rotator.rotation_model,
            reconstruction_time=0.0,
        )
        plate_ids = np.zeros(seeds.n_points, dtype=int)
        begin_ages = np.zeros(seeds.n_points, dtype=float)
        n_outside = 0
        for i, (lat, lon) in enumerate(seeds.latlon):
            hit = partitioner.partition_point(pygplates.PointOnSphere(lat, lon))
            if hit is None:
                n_outside += 1
                continue
            feature = hit.get_feature()
            plate_ids[i] = feature.get_reconstruction_plate_id()
            begin_ages[i] = feature.get_valid_time()[0]

        if n_outside > 0:
            raise ValueError(
                f"{n_outside} source points lie in no polygon at 0 Ma, so they "
                "have no plate ID for rotation='rigid_plate_id'"
            )
        if self.config.rigid_missing_plate == "topological_plate":
            # Plate ID 0 is allowed here: such seeds use the fallback plate.
            # The private lookup is used because assign_plate_ids only warns about
            # a point in no plate, and this method raises for it below.
            fallback_plate_ids = np.asarray(
                self._rotator._topology_plate_ids(seeds.xyz, 0.0)
            )
            n_none = int((fallback_plate_ids == 0).sum())
            if n_none > 0:
                raise ValueError(
                    f"{n_none} source points lie in no topological plate at 0 Ma, "
                    'so rigid_missing_plate="topological_plate" has no plate for them'
                )
            return plate_ids, begin_ages, fallback_plate_ids

        n_zero = int((plate_ids == 0).sum())
        if n_zero > 0:
            raise ValueError(
                f"{n_zero} of {seeds.n_points} source points are in polygons with "
                "plate ID 0. rotation='rigid_plate_id' needs polygons with "
                "reconstruction plate IDs, for example a GPML continent file, or "
                "rigid_missing_plate='topological_plate'."
            )
        return plate_ids, begin_ages, None

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

    def _warn_once_if_points_stay_in_place(self, age: float) -> None:
        """Log one warning if the rigid mode left source points in place.

        The per-call counts are logged at INFO, which a default run does not show.
        This warning appears once per source, at the first age that has such
        points, so a long run is not flooded.

        Parameters
        ----------
        age : float
            Geological age in Ma of the call that has just finished. It appears in
            the message.
        """
        n_in_place = self._rigid_stats.get("n_in_place", 0)
        if n_in_place > 0 and not self._warned_in_place:
            self._warned_in_place = True
            if self.config.rigid_missing_plate == "in_place":
                advice = (
                    "Use rigid_missing_plate='topological_plate' to rotate them with "
                    "the plate that contains them at 0 Ma."
                )
            else:
                advice = (
                    "Their fallback plate, the plate that contains them at 0 Ma, is "
                    "missing from the rotation tree as well."
                )
            logger.warning(
                "rotation='rigid_plate_id': %d of %d source points have a plate ID "
                "that is not in the rotation tree at %.2f Ma and stay in place. %s",
                n_in_place, self._rigid_stats.get("n_kept", 0), age, advice,
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
            # The extra arguments are passed only for the opt-in mode, so the
            # default call is the same as it has always been.
            **(
                {
                    "rotation": "rigid_plate_id",
                    "seed_plate_ids": self._seed_plate_ids,
                    "seed_begin_ages": self._seed_begin_ages,
                    "seed_fallback_plate_ids": self._seed_fallback_plate_ids,
                    "rigid_stats": self._rigid_stats,
                }
                if self.config.rotation == "rigid_plate_id"
                else {}
            ),
        )
        self._warn_once_if_points_stay_in_place(age)
        logger.debug("%d source points with a fresh background at %.2f Ma",
                     cloud.n_points, age)
        return cloud
