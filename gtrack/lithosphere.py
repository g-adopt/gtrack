"""Lithosphere source points through geological time.

The source combines oceanic and continental source points. The ocean tracker
advances towards younger geological ages and cannot rewind.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .checkpoints import CheckpointPolicy
from .config import TracerConfig
from .hpc_integration import SeafloorAgeTracker
from .logging import get_logger
from .point_rotation import PointCloud, PointRotator
from .polygon_filter import PolygonFilter

logger = get_logger(__name__)


def quintic_smoothstep(x: np.ndarray) -> np.ndarray:
    """Return a clamped quintic transition with two continuous derivatives."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


class OceanicLidWeight:
    """Map oceanic thickness to a dimensionless lateral weight.

    The result is zero at and below `zero_weight_thickness_km`.
    It is one at and above `full_weight_thickness_km`.
    A quintic transition joins these limits.

    Parameters
    ----------
    oceanic_thickness_from_age : callable
        Map an oceanic material age in Myr to a thickness in km.
    full_weight_age_myr : float, default=15.0
        Material age used to calculate `full_weight_thickness_km`.
    zero_weight_age_myr : float, default=1.0
        Material age used to calculate `zero_weight_thickness_km`.

    Examples
    --------
    >>> weight = OceanicLidWeight(oceanic_thickness_from_age=half_space_cooling)
    >>> source = LithosphereCloudSource(
    ...     ..., oceanic_weight_from_thickness=weight
    ... )
    """

    def __init__(
        self,
        oceanic_thickness_from_age: Callable[[np.ndarray], np.ndarray],
        full_weight_age_myr: float = 15.0,
        zero_weight_age_myr: float = 1.0,
    ):
        if not (zero_weight_age_myr < full_weight_age_myr):
            raise ValueError(
                "zero_weight_age_myr must be less than full_weight_age_myr "
                f"(got {zero_weight_age_myr} and {full_weight_age_myr})"
            )

        def _thickness_at(age_myr: float) -> float:
            return float(np.ravel(oceanic_thickness_from_age(
                np.array([age_myr], dtype=float)
            ))[0])

        self.full_weight_age_myr = float(full_weight_age_myr)
        self.zero_weight_age_myr = float(zero_weight_age_myr)
        self.zero_weight_thickness_km = _thickness_at(zero_weight_age_myr)
        self.full_weight_thickness_km = _thickness_at(full_weight_age_myr)

        # A large material age estimates the thickness cap of the cooling map.
        cap_km = _thickness_at(1.0e6)
        if not (self.full_weight_thickness_km > self.zero_weight_thickness_km):
            raise ValueError(
                "full_weight_thickness_km must exceed zero_weight_thickness_km"
            )
        if self.full_weight_thickness_km >= cap_km - 1e-9:
            raise ValueError(
                f"full_weight_age_myr={full_weight_age_myr} gives a thickness at "
                f"or above the {cap_km:.1f} km thickness cap"
            )

        logger.info(
            "OceanicLidWeight: zero<=%.3g Myr (%.1f km), full>=%.3g Myr "
            "(%.1f km), cap=%.1f km",
            self.zero_weight_age_myr, self.zero_weight_thickness_km,
            self.full_weight_age_myr, self.full_weight_thickness_km, cap_km,
        )

    def __call__(self, thickness_km: np.ndarray) -> np.ndarray:
        return quintic_smoothstep(
            (np.asarray(thickness_km, dtype=float) - self.zero_weight_thickness_km)
            / (self.full_weight_thickness_km - self.zero_weight_thickness_km)
        )

    def __repr__(self) -> str:
        return (
            f"OceanicLidWeight(full_weight_age_myr={self.full_weight_age_myr:g}, "
            f"zero_weight_age_myr={self.zero_weight_age_myr:g}, "
            f"zero_weight_thickness_km={self.zero_weight_thickness_km:.1f}, "
            f"full_weight_thickness_km={self.full_weight_thickness_km:.1f})"
        )


@dataclass(frozen=True)
class LithosphereCloudConfig:
    """Configure :class:`LithosphereCloudSource`.

    Parameters
    ----------
    tracer : TracerConfig
        Configuration for the seafloor age tracker.
    tracker_rebuild_interval_myr : float, default=50.0
        Elapsed time between tracker rebuilds.
    checkpoint : CheckpointPolicy, optional
        Checkpoint directory and interval for the ocean tracker. `None` disables
        checkpoints. The directory must exist before source construction.
    continental_material_age_myr : float, default=500.0
        Material age for continental source points.
    oldest_requested_age_ma : float, optional
        Oldest geological age that the source will receive.
    scalar_continental_point_count : int, default=20000
        Source-point count for a scalar continental thickness. Other input
        forms supply their own source points.

    Examples
    --------
    >>> config = LithosphereCloudConfig(
    ...     tracer=TracerConfig(tracker_step_myr=2.0, tracker_point_count=10000),
    ...     tracker_rebuild_interval_myr=50.0,
    ... )
    """

    tracer: TracerConfig = field(default_factory=TracerConfig)
    tracker_rebuild_interval_myr: float = 50.0
    checkpoint: Optional[CheckpointPolicy] = None
    continental_material_age_myr: float = 500.0
    oldest_requested_age_ma: Optional[float] = None
    scalar_continental_point_count: int = 20000

    def __post_init__(self):
        if self.tracker_rebuild_interval_myr <= 0:
            raise ValueError(
                "tracker_rebuild_interval_myr must be positive, "
                f"got {self.tracker_rebuild_interval_myr}"
            )
        if self.oldest_requested_age_ma is not None and self.oldest_requested_age_ma < 0:
            raise ValueError(
                "oldest_requested_age_ma must be non-negative, "
                f"got {self.oldest_requested_age_ma}"
            )
        if self.scalar_continental_point_count < 1:
            raise ValueError(
                "scalar_continental_point_count must be positive, "
                f"got {self.scalar_continental_point_count}"
            )


class LithosphereCloudSource:
    """Oceanic plus continental lithosphere as a labelled cloud per age.

    Satisfies :class:`~gtrack.age_sources.AgeCloudSource`.

    Parameters
    ----------
    rotation_files : list of str or Path
        Rotation files for the plate model.
    topology_files : list of str or Path
        Topology files for the plate model.
    continental_polygons : str or Path
        Present-day continental polygons. Used both to exclude continental
        area from the oceanic tracker and to clip the continental cloud.
    static_polygons : str or Path
        Static plate polygons, used by the point rotator.
    continental_data : PointCloud or str or Path or tuple or int or float
        Present-day continental thickness, in anything
        :meth:`~gtrack.point_rotation.PointCloud.from_data` accepts.
    oceanic_thickness_from_age : callable
        Maps an array of seafloor ages in Myr to the tracked property,
        typically thickness in km via half-space cooling.
    plate_model_max_age_ma : float
        Oldest age the plate model covers, in Ma. The tracker starts its walk
        here when no checkpoint is available.
    config : LithosphereCloudConfig, optional
        Tracker, rebuild, checkpoint, and continental parameters.
    oceanic_weight_from_thickness : callable, optional
        Maps oceanic thickness in km to a lateral weight in ``[0, 1]``.
        Continental source points receive a lateral weight of one. If the
        argument is ``None``, the source does not publish this channel.

    Attributes
    ----------
    provides : frozenset of str
        ``{"thickness", "age"}``, plus ``"lateral_weight"`` when
        ``oceanic_weight_from_thickness`` is given. The ``age`` channel contains
        the material age in Myr.
    monotonic_backward : bool
        True. Each ``at_age`` call must request an age no older than the last.

    Examples
    --------
    >>> source = LithosphereCloudSource(
    ...     rotation_files=rotations,
    ...     topology_files=topologies,
    ...     continental_polygons=cont_polys,
    ...     static_polygons=static_polys,
    ...     continental_data=100.0,
    ...     oceanic_thickness_from_age=lambda ages: 2.32 * np.sqrt(ages),
    ...     plate_model_max_age_ma=400.0,
    ... )
    >>> cloud = source.at_age(200.0)
    """

    provides = frozenset({"thickness", "age"})
    monotonic_backward = True

    #: Name of the thickness channel.
    PROPERTY_NAME = "thickness"

    def __init__(
        self,
        rotation_files,
        topology_files,
        continental_polygons,
        static_polygons,
        continental_data,
        oceanic_thickness_from_age: Callable[[np.ndarray], np.ndarray],
        plate_model_max_age_ma: float,
        config: Optional[LithosphereCloudConfig] = None,
        oceanic_weight_from_thickness: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if continental_polygons is None:
            raise ValueError("continental_polygons is required")
        if static_polygons is None:
            raise ValueError("static_polygons is required")
        if plate_model_max_age_ma <= 0:
            raise ValueError(
                f"plate_model_max_age_ma must be positive, got {plate_model_max_age_ma}"
            )

        self.config = config or LithosphereCloudConfig()
        if (self.config.oldest_requested_age_ma is not None
                and self.config.oldest_requested_age_ma > plate_model_max_age_ma):
            raise ValueError(
                f"oldest_requested_age_ma ({self.config.oldest_requested_age_ma}) "
                f"exceeds plate_model_max_age_ma ({plate_model_max_age_ma})"
            )

        self.rotation_files = rotation_files
        self.topology_files = topology_files
        self.continental_polygons = continental_polygons
        self.static_polygons = static_polygons
        self.plate_model_max_age_ma = plate_model_max_age_ma
        self.oceanic_thickness_from_age = oceanic_thickness_from_age

        # This map supplies the lateral weight for oceanic source points.
        # Continental source points use a lateral weight of one.
        self._oceanic_weight_from_thickness = oceanic_weight_from_thickness
        self.provides = frozenset({"thickness", "age"}) | (
            {"lateral_weight"} if oceanic_weight_from_thickness is not None else set()
        )

        self._continental_data = continental_data

        # Delay plate-model I/O until the first source request.
        self._tracker = None
        self._rotator = None
        self._continental_filter = None
        self._continental_present = None

        self._initialized = False
        self._last_walked_age = None
        self._last_tracker_rebuild_age = None
        self._last_checkpoint_age = None

    # -- construction ------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Build the plate machinery and present-day continental cloud once."""
        if self._tracker is not None:
            return
        self._tracker = SeafloorAgeTracker(
            rotation_files=self.rotation_files,
            topology_files=self.topology_files,
            continental_polygons=self.continental_polygons,
            config=self.config.tracer,
        )
        self._rotator = PointRotator(
            rotation_files=self.rotation_files,
            topology_files=self.topology_files,
            static_polygons=self.static_polygons,
        )
        self._continental_filter = PolygonFilter(
            polygon_files=self.continental_polygons,
            rotation_files=self.rotation_files,
        )
        self._continental_present = self._build_continental_cloud()

    def _build_continental_cloud(self) -> PointCloud:
        """Load the present-day continental cloud and clip it to the polygons."""
        cloud = PointCloud.from_data(
            self._continental_data,
            self.PROPERTY_NAME,
            n_points_fallback=self.config.scalar_continental_point_count,
        )
        n_before = cloud.n_points
        cloud = self._continental_filter.filter_inside(cloud, at_age=0.0)
        if n_before > 0:
            logger.debug(
                "continental data filtered %d -> %d points (%.1f%% retained)",
                n_before, cloud.n_points, 100 * cloud.n_points / n_before,
            )
        # Do not assign static plate IDs here. The topological rotator finds the
        # plate or network that contains each source point at each time.
        return cloud

    # -- age contract ------------------------------------------------------

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` cannot be served.

        The method checks the model range and `oldest_requested_age_ma`.
        It also rejects a request that requires the tracker to rewind.

        Parameters
        ----------
        age : float
            Geological age in Ma.

        Raises
        ------
        ValueError
            If any of the three checks fails.
        """
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future)."
            )
        if age > self.plate_model_max_age_ma:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's "
                f"maximum age ({self.plate_model_max_age_ma:.2f} Ma)."
            )
        oldest_requested_age_ma = self.config.oldest_requested_age_ma
        if oldest_requested_age_ma is not None and age > oldest_requested_age_ma:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the declared "
                f"oldest_requested_age_ma ({oldest_requested_age_ma:.2f} Ma)"
            )
        if self._last_walked_age is not None and age > self._last_walked_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the last walked "
                f"age ({self._last_walked_age:.2f} Ma). The tracker walks "
                f"forward only, towards decreasing age, and cannot rewind."
            )

    # -- the cloud ---------------------------------------------------------

    def at_age(self, age: float) -> PointCloud:
        """Return the lithosphere cloud at ``age``.

        Parameters
        ----------
        age : float
            Geological age in Ma.

        Returns
        -------
        PointCloud
            Oceanic source points followed by continental source points. The
            source points carry `thickness` and `age`.
        """
        self.validate_age(age)
        self._ensure_loaded()

        ocean = self._step_ocean_to(age)
        ocean.add_property(
            self.PROPERTY_NAME,
            self.oceanic_thickness_from_age(ocean.get_property("age")),
        )

        continental = self._rotator.rotate(
            self._continental_present, from_age=0.0, to_age=age
        )
        continental.add_property(
            "age",
            np.full(continental.n_points, self.config.continental_material_age_myr),
        )

        if self._oceanic_weight_from_thickness is not None:
            ocean.add_property(
                "lateral_weight",
                np.clip(
                    self._oceanic_weight_from_thickness(
                        ocean.get_property(self.PROPERTY_NAME)
                    ),
                    0.0,
                    1.0,
                ),
            )
            # Continental source points have a uniform lateral weight.
            continental.add_property(
                "lateral_weight",
                np.ones(continental.n_points, dtype=float),
            )

        self._last_walked_age = age
        return PointCloud.concatenate([ocean, continental], warn=False)

    # -- the walk ----------------------------------------------------------

    def _initialize_walk(self, age: float) -> None:
        """Start the tracker from a checkpoint or the model maximum age."""
        if self._initialized:
            return
        loaded = False
        policy = self.config.checkpoint
        best = policy.best_at_or_before(age) if policy is not None else None
        if best is not None:
            try:
                self._tracker.load_checkpoint(best)
                loaded_age = self._tracker.current_age
                logger.debug("loaded ocean checkpoint at %s Ma from %s",
                             loaded_age, best)
                self._last_tracker_rebuild_age = loaded_age
                self._last_checkpoint_age = loaded_age
                loaded = True
            except Exception as exc:
                # A full tracker walk recovers from a damaged or incompatible
                # checkpoint without changing the calculated state.
                logger.info("failed to load checkpoint %s: %s. "
                            "Falling back to a full walk.", best, exc)
        if not loaded:
            # pyGplates works in whole Ma.
            starting_age = int(self.plate_model_max_age_ma)
            logger.debug("initialising ocean tracker at %d Ma", starting_age)
            self._tracker.initialize(starting_age=starting_age)
            self._last_tracker_rebuild_age = starting_age
        self._initialized = True

    def _step_ocean_to(self, age: float) -> PointCloud:
        """Advance the tracker to ``age`` and do scheduled maintenance."""
        self._initialize_walk(age)

        if (self._last_tracker_rebuild_age is not None
                and abs(self._last_tracker_rebuild_age - age)
                >= self.config.tracker_rebuild_interval_myr):
            logger.debug("rebuilding the ocean tracker at %.2f Ma", age)
            self._tracker.tracker_rebuild(
                n_points=self.config.tracer.tracker_point_count
            )
            self._last_tracker_rebuild_age = age

        cloud = self._tracker.step_to(int(round(age)))
        self._save_checkpoint_if_due(age)
        return cloud

    def _save_checkpoint_if_due(self, age: float) -> None:
        """Write a checkpoint when the policy says one is due."""
        policy = self.config.checkpoint
        if policy is None:
            return
        if not policy.is_due(age, self._last_checkpoint_age):
            return
        rounded_age = int(round(age))
        filepath = policy.path_for(age)
        try:
            self._tracker.save_checkpoint(filepath)
            self._last_checkpoint_age = rounded_age
            logger.debug("saved ocean checkpoint at %d Ma -> %s",
                         rounded_age, filepath)
        except Exception as exc:
            # A failed checkpoint write affects restart cost, not this result.
            logger.info("failed to save checkpoint at %d Ma: %s",
                        rounded_age, exc)
