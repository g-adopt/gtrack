"""Lithospheric thickness clouds through geological time.

Combines two populations into one labelled cloud per age: oceanic seeds carried
by a forward-walking seafloor age tracker, and continental seeds rotated back
from their present-day positions. The oceanic half carries a real seafloor age
and a thickness derived from it; the continental half carries a nominal age and
its present-day thickness.

The ocean tracker is the only stateful piece. It advances forward in geological
time — towards decreasing age — and cannot rewind, which is why this source
declares ``monotonic_backward``.

This module is serial and knows nothing about MPI. A parallel consumer is
expected to run it on one rank and broadcast the result itself.
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


@dataclass(frozen=True)
class LithosphereCloudConfig:
    """Knobs for :class:`LithosphereCloudSource`.

    Parameters
    ----------
    tracer : TracerConfig
        Configuration for the seafloor age tracker, nested rather than
        flattened. Every tracker knob has exactly one home here, including
        ``time_step`` and ``default_mesh_points``.
    reinit_interval_myr : float, default=50.0
        Age separation after which the tracker's mesh is regenerated and ages
        re-interpolated onto it.
    checkpoint : CheckpointPolicy, optional
        Where to write ocean-tracker checkpoints and how often. None disables
        checkpointing entirely. The directory is not created here; create it
        before constructing the source so a mistyped path fails at wiring time.
    default_continental_age_myr : float, default=500.0
        Age assigned to continental seeds, which carry no seafloor age of
        their own.
    walk_start_age : float, optional
        The oldest age that will ever be requested. Declaring it makes an
        out-of-range request fail immediately rather than silently fixing the
        walk's ceiling at whatever age happens to be asked for first. It does
        NOT permit revisiting ages already stepped past.
    continental_fallback_n : int, default=20000
        Number of points in the sphere mesh a SCALAR continental thickness is
        broadcast onto. Ignored for every other input shape, which brings its
        own points.

    Notes
    -----
    ``tracer`` being a nested config rather than a dict of overrides is the
    substance of this class, not a stylistic choice. A design in which some
    tracker knobs are fields here and the rest arrive in a passthrough dict
    lets the dict silently win, so a field set on this object is discarded
    without warning. One home per knob makes that unexpressible.

    Examples
    --------
    >>> config = LithosphereCloudConfig(
    ...     tracer=TracerConfig(time_step=2.0, default_mesh_points=10000),
    ...     reinit_interval_myr=50.0,
    ... )
    """

    tracer: TracerConfig = field(default_factory=TracerConfig)
    reinit_interval_myr: float = 50.0
    checkpoint: Optional[CheckpointPolicy] = None
    default_continental_age_myr: float = 500.0
    walk_start_age: Optional[float] = None
    continental_fallback_n: int = 20000

    def __post_init__(self):
        if self.reinit_interval_myr <= 0:
            raise ValueError(
                f"reinit_interval_myr must be positive, "
                f"got {self.reinit_interval_myr}"
            )
        if self.walk_start_age is not None and self.walk_start_age < 0:
            raise ValueError(
                f"walk_start_age must be non-negative, got {self.walk_start_age}"
            )
        if self.continental_fallback_n < 1:
            raise ValueError(
                f"continental_fallback_n must be positive, "
                f"got {self.continental_fallback_n}"
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
    age_to_property : callable
        Maps an array of seafloor ages in Myr to the tracked property,
        typically thickness in km via half-space cooling.
    oldest_age : float
        Oldest age the plate model covers, in Ma. The tracker starts its walk
        here when no checkpoint is available.
    config : LithosphereCloudConfig, optional
        Tracker, reinit, checkpoint and continental knobs.
    oceanic_amplitude : callable, optional
        Maps an array of oceanic thickness (km) to a lateral amplitude in
        ``[0, 1]``, applied to the oceanic seeds only. When given, the source
        publishes a ``surface_amplitude`` channel (oceanic value clipped to
        ``[0, 1]``, continental value fixed at ``1.0``) so a consumer can
        weaken thin/young seafloor without touching continents. ``None`` (the
        default) publishes no such channel.

    Attributes
    ----------
    provides : frozenset of str
        ``{"thickness", "age"}``, plus ``"surface_amplitude"`` when
        ``oceanic_amplitude`` is given. Coordinates are carried by the cloud
        itself and are deliberately not listed.
    monotonic_backward : bool
        True. Each ``at_age`` call must request an age no older than the last.

    Notes
    -----
    Construction is cheap and touches no reconstruction data. The tracker,
    rotator, polygon filter and present-day continental cloud are all built on
    the first :meth:`at_age`, so a source can be constructed and inspected —
    and a consumer wired up and tested — without any plate files being read.

    Examples
    --------
    >>> source = LithosphereCloudSource(
    ...     rotation_files=rotations,
    ...     topology_files=topologies,
    ...     continental_polygons=cont_polys,
    ...     static_polygons=static_polys,
    ...     continental_data=100.0,
    ...     age_to_property=lambda ages: 2.32 * np.sqrt(ages),
    ...     oldest_age=400.0,
    ... )
    >>> cloud = source.at_age(200.0)
    """

    provides = frozenset({"thickness", "age"})
    monotonic_backward = True

    #: Name the thickness is attached under. Fixed rather than configurable,
    #: because ``provides`` names it and the two must agree.
    PROPERTY_NAME = "thickness"

    def __init__(
        self,
        rotation_files,
        topology_files,
        continental_polygons,
        static_polygons,
        continental_data,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        oldest_age: float,
        config: Optional[LithosphereCloudConfig] = None,
        oceanic_amplitude: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if continental_polygons is None:
            raise ValueError("continental_polygons is required")
        if static_polygons is None:
            raise ValueError("static_polygons is required")
        if oldest_age <= 0:
            raise ValueError(f"oldest_age must be positive, got {oldest_age}")

        self.config = config or LithosphereCloudConfig()
        if (self.config.walk_start_age is not None
                and self.config.walk_start_age > oldest_age):
            raise ValueError(
                f"walk_start_age ({self.config.walk_start_age}) is older than "
                f"the plate model's oldest age ({oldest_age})."
            )

        self.rotation_files = rotation_files
        self.topology_files = topology_files
        self.continental_polygons = continental_polygons
        self.static_polygons = static_polygons
        self.oldest_age = oldest_age
        self.age_to_property = age_to_property

        # Optional lateral amplitude for OCEANIC lithosphere only: maps oceanic
        # thickness (km) to an amplitude in [0, 1]. Published as the
        # ``surface_amplitude`` channel so a consumer can weaken thin/young
        # seafloor (e.g. ridges) without touching continents, which are always
        # 1.0. None keeps today's behaviour and does not publish the channel.
        self._oceanic_amplitude = oceanic_amplitude
        self.provides = frozenset({"thickness", "age"}) | (
            {"surface_amplitude"} if oceanic_amplitude is not None else set()
        )

        self._continental_data = continental_data

        # Built on first use, never in __init__.
        self._tracker = None
        self._rotator = None
        self._continental_filter = None
        self._continental_present = None

        self._initialized = False
        self._last_walked_age = None
        self._last_reinit_age = None
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
            n_points_fallback=self.config.continental_fallback_n,
        )
        n_before = cloud.n_points
        cloud = self._continental_filter.filter_inside(cloud, at_age=0.0)
        if n_before > 0:
            logger.debug(
                "continental data filtered %d -> %d points (%.1f%% retained)",
                n_before, cloud.n_points, 100 * cloud.n_points / n_before,
            )
        # Deliberately no plate-id assignment: the topological rotator advects
        # every seed by whatever plate or network contains it and does not need
        # ids, and dropping seeds whose static-polygon id has no rotation
        # circuit would bias the result at exactly the passive margins that
        # matter.
        return cloud

    # -- age contract ------------------------------------------------------

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` cannot be served.

        Three checks, in order of how early they can fire.

        The range check rejects a negative age or one older than the plate
        model covers. The promise check rejects an age older than a declared
        ``walk_start_age``, and fires before any walking has happened, so a
        mis-wired consumer fails at setup rather than mid-run. The floor check
        rejects an age older than one already walked past, which is
        unreachable because the tracker cannot rewind.

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
        if age > self.oldest_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's "
                f"oldest age ({self.oldest_age:.2f} Ma)."
            )
        walk_start_age = self.config.walk_start_age
        if walk_start_age is not None and age > walk_start_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the declared "
                f"walk_start_age ({walk_start_age:.2f} Ma). Raise "
                f"walk_start_age to the oldest age you will request, or "
                f"request a younger age. The tracker walks forward only, "
                f"towards decreasing age."
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
            Oceanic seeds followed by continental seeds, carrying
            ``thickness`` and ``age``.
        """
        self.validate_age(age)
        self._ensure_loaded()

        ocean = self._step_ocean_to(age)
        ocean.add_property(
            self.PROPERTY_NAME, self.age_to_property(ocean.get_property("age"))
        )

        continental = self._rotator.rotate(
            self._continental_present, from_age=0.0, to_age=age
        )
        continental.add_property(
            "age",
            np.full(continental.n_points, self.config.default_continental_age_myr),
        )

        if self._oceanic_amplitude is not None:
            ocean.add_property(
                "surface_amplitude",
                np.clip(
                    self._oceanic_amplitude(ocean.get_property(self.PROPERTY_NAME)),
                    0.0,
                    1.0,
                ),
            )
            # Continents never fade; carry the channel so concatenate keeps it.
            continental.add_property(
                "surface_amplitude",
                np.ones(continental.n_points, dtype=float),
            )

        self._last_walked_age = age
        return PointCloud.concatenate([ocean, continental], warn=False)

    # -- the walk ----------------------------------------------------------

    def _initialize_walk(self, age: float) -> None:
        """Start the walk, from a checkpoint when one is usable.

        Idempotent. Distinct from building the machinery: this establishes the
        tracker's age state, which the machinery does not have until it is
        either initialised at the model's oldest age or restored from a file.
        """
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
                self._last_reinit_age = loaded_age
                self._last_checkpoint_age = loaded_age
                loaded = True
            except Exception as exc:
                # A damaged or incompatible checkpoint must not end the run:
                # the full walk reproduces the same state, only slower.
                logger.info("failed to load checkpoint %s: %s. "
                            "Falling back to a full walk.", best, exc)
        if not loaded:
            # pyGplates works in whole Ma.
            starting_age = int(self.oldest_age)
            logger.debug("initialising ocean tracker at %d Ma", starting_age)
            self._tracker.initialize(starting_age=starting_age)
            self._last_reinit_age = starting_age
        self._initialized = True

    def _step_ocean_to(self, age: float) -> PointCloud:
        """Advance the tracker to ``age``, reinitialising and saving as due."""
        self._initialize_walk(age)

        if (self._last_reinit_age is not None
                and abs(self._last_reinit_age - age) >= self.config.reinit_interval_myr):
            logger.debug("reinitialising ocean tracker at %.2f Ma", age)
            self._tracker.reinitialize(
                n_points=self.config.tracer.default_mesh_points
            )
            self._last_reinit_age = age

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
            # Losing a checkpoint costs time on a later run, not correctness
            # of this one.
            logger.info("failed to save checkpoint at %d Ma: %s",
                        rounded_age, exc)
