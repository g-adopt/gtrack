"""
Seafloor age tracking using Lagrangian particle tracking.

This module provides the main API for computing seafloor ages
using pygplates' C++ backend for efficient reconstruction.
"""

import gc
import numpy as np
from typing import Callable, Dict, List, Optional, Union

import pygplates

from .config import TracerConfig
from .point_rotation import PointCloud
from .mesh import create_sphere_mesh_latlon
from .mor_seeds import generate_mor_seeds
from .boundaries import ContinentalPolygonCache
from .initial_conditions import compute_initial_ages, default_age_distance_law
from .logging import get_logger

logger = get_logger(__name__)


class SeafloorAgeTracker:
    """
    Seafloor age tracker using Lagrangian particle tracking with C++ backend.

    Uses pygplates.TopologicalModel.reconstruct_geometry() for efficient
    point advection with built-in collision detection.

    Key features:
    - GPlately-compatible: Matches GPlately's SeafloorGrid output
    - C++ backend: Fast reconstruction using pygplates internals
    - Icosahedral initialization: Full ocean coverage from start
    - Continental filtering: Via polygon queries with caching
    - Checkpointing: Save/restore state for restarts

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot).
    topology_files : list of str
        Paths to topology/plate boundary files (.gpml/.gpmlz).
    continental_polygons : str, optional
        Path to continental polygon file. If None, tracers are not
        removed when they enter continental regions.
    config : TracerConfig, optional
        Configuration parameters. If None, uses defaults.
    verbose : bool, default=True
        Deprecated. Use GTRACK_LOGLEVEL environment variable instead.
        Set GTRACK_LOGLEVEL=INFO for progress messages or DEBUG for details.

    Examples
    --------
    >>> # Initialize tracker
    >>> tracker = SeafloorAgeTracker(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     continental_polygons='continents.gpmlz'
    ... )
    >>>
    >>> # Initialize with sphere mesh (GPlately-compatible)
    >>> tracker.initialize(starting_age=200)
    >>>
    >>> # Step forward (decreasing geological age toward present)
    >>> for target_age in range(199, -1, -1):
    ...     cloud = tracker.step_to(target_age)
    ...     xyz = cloud.xyz
    ...     ages = cloud.get_property('age')
    """

    def __init__(
        self,
        rotation_files: Union[str, List[str]],
        topology_files: Union[str, List[str]],
        continental_polygons: Optional[str] = None,
        config: Optional[TracerConfig] = None,
    ):
        from .geometry import ensure_list, load_features

        self._config = config if config else TracerConfig()

        # Handle single file or Path as list
        rotation_files = ensure_list(rotation_files)
        topology_files = ensure_list(topology_files)

        self._rotation_files = rotation_files
        self._topology_files = topology_files
        self._continental_polygons_path = continental_polygons

        logger.info("Initializing SeafloorAgeTracker...")

        # Load rotation model
        self._rotation_model = pygplates.RotationModel(rotation_files)

        # Load topology features (accepts single filename, list, or features)
        self._topology_features = load_features(topology_files)

        # Create TopologicalModel for C++ backend reconstruction
        self._topological_model = pygplates.TopologicalModel(
            self._topology_features, self._rotation_model
        )

        # Set up continental polygon cache.  During forward time-stepping we
        # reconstruct polygons at the exact time each step, so a small cache
        # of recent reconstructions avoids redundant pygplates calls.
        if continental_polygons is not None:
            self._continental_cache = ContinentalPolygonCache(
                continental_polygons,
                self._rotation_model,
                max_cache_size=5,
            )
        else:
            self._continental_cache = None

        # State variables
        self._current_age: Optional[float] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        self._ages: Optional[np.ndarray] = None  # Material ages (time since formation)
        self._initialized = False
        self._gc_step_counter: int = 0

        logger.info("  Initialization complete.")

    def initialize(
        self,
        starting_age: float,
        method: str = 'mesh',
        n_points: Optional[int] = None,
        initial_ocean_mean_spreading_rate: Optional[float] = None,
        age_distance_law: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    ) -> int:
        """
        Initialize tracers for given geological age.

        Parameters
        ----------
        starting_age : float
            Starting geological age (Ma). Tracers are placed based on
            ocean structure at this age.
        method : str, default='mesh'
            Initialization method:
            - 'mesh': Full ocean mesh with computed ages (GPlately-compatible)
            - 'ridge_only': Tracers only at ridges with age=0 (legacy gtrack)
        n_points : int, optional
            Number of points for the sphere mesh. If None, uses config default.
        initial_ocean_mean_spreading_rate : float, optional
            Spreading rate for age calculation (mm/yr). If None, uses config default.
        age_distance_law : callable, optional
            Custom function to convert distance to age.
            Signature: (distances_km, spreading_rate_mm_yr) -> ages_myr

        Returns
        -------
        int
            Number of tracers initialized.

        Examples
        --------
        >>> # GPlately-compatible initialization
        >>> tracker.initialize(starting_age=200)
        >>>
        >>> # Higher resolution
        >>> tracker.initialize(starting_age=200, n_points=40000)
        >>>
        >>> # Custom age calculation
        >>> def my_age_law(distances, rate):
        ...     return distances / (rate / 2) * 1.1  # 10% older
        >>> tracker.initialize(starting_age=200, age_distance_law=my_age_law)
        """
        if n_points is None:
            n_points = self._config.default_mesh_points
        if initial_ocean_mean_spreading_rate is None:
            initial_ocean_mean_spreading_rate = self._config.initial_ocean_mean_spreading_rate

        logger.info(f"Initializing tracers at {starting_age} Ma (method='{method}')...")

        if method == 'mesh':
            self._initialize_mesh(
                starting_age,
                n_points,
                initial_ocean_mean_spreading_rate,
                age_distance_law,
            )
        elif method == 'ridge_only':
            self._initialize_ridge_only(starting_age)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        self._current_age = starting_age
        self._initialized = True
        self._gc_step_counter = 0

        logger.info(f"  Initialized with {len(self._lats)} tracers at {starting_age} Ma")

        return len(self._lats)

    def _initialize_mesh(
        self,
        starting_age: float,
        n_points: int,
        spreading_rate: float,
        age_distance_law: Optional[Callable],
    ):
        """Initialize with sphere mesh (GPlately-compatible)."""
        logger.debug(f"  Creating sphere mesh ({n_points:,} points)...")

        # Create sphere mesh and get lat/lon coordinates directly
        mesh_lats, mesh_lons = create_sphere_mesh_latlon(n_points)

        logger.debug(f"  Created mesh with {len(mesh_lats)} points")

        # Resolve topologies at starting time (needed for compute_initial_ages)
        resolved_topologies = []
        shared_boundary_sections = []
        pygplates.resolve_topologies(
            self._topology_features,
            self._rotation_model,
            resolved_topologies,
            starting_age,
            shared_boundary_sections,
        )

        # Filter out continental points if we have continental polygons
        if self._continental_cache is not None:
            continental_mask = self._continental_cache.get_continental_mask(
                mesh_lats, mesh_lons, starting_age
            )

            # Filter to ocean points
            ocean_mask = ~continental_mask
            ocean_lats = mesh_lats[ocean_mask]
            ocean_lons = mesh_lons[ocean_mask]

            logger.debug(f"  Filtered to {len(ocean_lats)} ocean points (removed {continental_mask.sum()} continental)")

            # Create ocean point MultiPointOnSphere for compute_initial_ages
            ocean_points = pygplates.MultiPointOnSphere(zip(ocean_lats, ocean_lons))
        else:
            ocean_lats = mesh_lats
            ocean_lons = mesh_lons
            ocean_points = pygplates.MultiPointOnSphere(zip(ocean_lats, ocean_lons))

        # Compute initial ages from distance to ridge (plate-based approach)
        logger.debug("  Computing initial ages from distance to ridge...")

        lons, lats, ages = compute_initial_ages(
            ocean_points,
            resolved_topologies,
            shared_boundary_sections,
            initial_ocean_mean_spreading_rate=spreading_rate,
            age_distance_law=age_distance_law,
        )

        self._lats = lats
        self._lons = lons
        self._ages = ages

    def _initialize_ridge_only(self, starting_age: float):
        """Initialize with tracers only at ridges (legacy gtrack approach)."""
        logger.debug("  Generating ridge seed points...")

        lats, lons = generate_mor_seeds(
            starting_age,
            self._topology_features,
            self._rotation_model,
            ridge_sampling_degrees=self._config.ridge_sampling_degrees,
            spreading_offset_degrees=self._config.spreading_offset_degrees,
        )

        self._lats = lats
        self._lons = lons
        self._ages = np.zeros(len(lats))  # All tracers start with age 0

    def initialize_from_cloud(
        self,
        cloud: PointCloud,
        current_age: float,
    ) -> int:
        """
        Initialize from existing PointCloud.

        Use this to restart from a checkpoint or to provide
        custom initial tracer positions.

        Parameters
        ----------
        cloud : PointCloud
            Point cloud with 'age' property containing material
            age of each tracer (time since ridge formation).
        current_age : float
            Current geological age (Ma).

        Returns
        -------
        int
            Number of tracers initialized.

        Raises
        ------
        ValueError
            If cloud does not have 'age' property.
        """
        if 'age' not in cloud.properties:
            raise ValueError(
                "PointCloud must have 'age' property. "
                "This should contain the material age of each tracer."
            )

        logger.info(f"Initializing from PointCloud at {current_age} Ma...")

        # Convert XYZ to lat/lon
        lonlat = cloud.lonlat
        self._lons = lonlat[:, 0]
        self._lats = lonlat[:, 1]
        self._ages = cloud.get_property('age').copy()
        self._current_age = current_age
        self._initialized = True
        self._gc_step_counter = 0

        logger.info(f"  Initialized with {len(self._lats)} tracers at {current_age} Ma")

        return len(self._lats)

    def step_to(self, target_age: float) -> PointCloud:
        """
        Evolve tracers to target geological age using C++ backend.

        Can only step forward (decreasing geological age toward 0).

        Parameters
        ----------
        target_age : float
            Target geological age (Ma). Must be less than current_age.

        Returns
        -------
        PointCloud
            Point cloud with 'age' property containing material ages.

        Raises
        ------
        RuntimeError
            If tracker is not initialized.
        ValueError
            If target_age > current_age (can only go forward).
        """
        if not self._initialized:
            raise RuntimeError(
                "Must call initialize() or initialize_from_cloud() first!"
            )

        if target_age > self._current_age:
            raise ValueError(
                f"Can only step forward (decreasing age). "
                f"Current age: {self._current_age}, target: {target_age}"
            )

        if target_age == self._current_age:
            logger.debug(f"Already at {target_age} Ma, no update needed")
            return self.get_current_state()

        logger.info(f"Evolving: {self._current_age} Ma -> {target_age} Ma")

        # Process in time_step increments
        time = self._current_age
        while time > target_age:
            next_time = max(time - self._config.time_step, target_age)

            if len(self._lats) == 0:
                logger.warning("  No points to reconstruct")
                break

            # Create MultiPointOnSphere from lat/lon tuples (faster than individual PointOnSphere)
            points = pygplates.MultiPointOnSphere(
                zip(self._lats, self._lons)
            )

            # Each outer iteration is a single pygplates reconstruction step
            # of length `span`, so we pass `time_increment=span` and the
            # divisibility constraint `(initial_time - youngest_time) /
            # time_increment == 1` is satisfied by construction. This lets
            # any positive float time_step work, including the clamped final
            # step where span may be smaller than the nominal time_step.
            span = time - next_time
            reconstructed_time_span = self._topological_model.reconstruct_geometry(
                points,
                initial_time=time,
                youngest_time=next_time,
                time_increment=span,
                deactivate_points=pygplates.ReconstructedGeometryTimeSpan.DefaultDeactivatePoints(
                    threshold_velocity_delta=self._config.velocity_delta_threshold_cm_yr,
                    threshold_distance_to_boundary=self._config.distance_threshold_per_myr,
                    deactivate_points_that_fall_outside_a_network=True,
                ),
            )

            # Get reconstructed points (inactive points are None)
            reconstructed_points = reconstructed_time_span.get_geometry_points(
                next_time, return_inactive_points=True
            )

            # Update coordinates and ages, removing inactive points
            self._update_from_reconstructed(reconstructed_points, time - next_time)

            # Remove continental points
            if self._continental_cache is not None:
                self._remove_continental_points(next_time)

            # Add new MOR seed points
            self._add_mor_seeds(next_time)

            # Periodically force GC to reclaim pygplates C++ wrapper objects
            # that may hold circular references and accumulate between steps.
            self._gc_step_counter += 1
            freq = self._config.gc_collect_frequency
            if freq is not None and self._gc_step_counter % freq == 0:
                gc.collect()

            time = next_time

        self._current_age = target_age

        logger.info(f"  Current tracer count: {len(self._lats)}")

        return self.get_current_state()

    def _update_from_reconstructed(
        self,
        reconstructed_points: List,
        delta_time: float,
    ):
        """Update coordinates from reconstruction, removing inactive points."""
        # Create boolean mask for active (non-None) points
        active_mask = np.array([p is not None for p in reconstructed_points], dtype=bool)
        n_active = active_mask.sum()

        if n_active == 0:
            self._lats = np.array([])
            self._lons = np.array([])
            self._ages = np.array([])
            return

        # Pre-allocate arrays
        new_lats = np.empty(n_active)
        new_lons = np.empty(n_active)

        # Extract coordinates from active points
        j = 0
        for i, point in enumerate(reconstructed_points):
            if point is not None:
                new_lats[j], new_lons[j] = point.to_lat_lon()
                j += 1

        # Update ages using vectorized operation
        self._lats = new_lats
        self._lons = new_lons
        self._ages = self._ages[active_mask] + delta_time

    def _remove_continental_points(self, time: float):
        """Remove points inside continental polygons.

        Continental polygons are reconstructed at the exact requested time
        using pygplates' full float-precision interpolation (SLERP between
        rotation pole samples). The ContinentalPolygonCache handles LRU
        caching to avoid redundant pygplates.reconstruct() calls.
        """
        if len(self._lats) == 0:
            return

        continental_mask = self._continental_cache.get_continental_mask(
            self._lats, self._lons, time
        )

        if continental_mask.any():
            ocean_mask = ~continental_mask
            self._lats = self._lats[ocean_mask]
            self._lons = self._lons[ocean_mask]
            self._ages = self._ages[ocean_mask]

            logger.debug(f"    Removed {continental_mask.sum()} continental points")

    def _add_mor_seeds(self, time: float):
        """Add new MOR seed points."""
        new_lats, new_lons = generate_mor_seeds(
            time,
            self._topology_features,
            self._rotation_model,
            ridge_sampling_degrees=self._config.ridge_sampling_degrees,
            spreading_offset_degrees=self._config.spreading_offset_degrees,
        )

        if len(new_lats) > 0:
            self._lats = np.concatenate([self._lats, new_lats])
            self._lons = np.concatenate([self._lons, new_lons])
            self._ages = np.concatenate([self._ages, np.zeros(len(new_lats))])

            logger.debug(f"    Added {len(new_lats)} new MOR seed points")

    def get_current_state(self) -> PointCloud:
        """
        Get current tracers as PointCloud without evolving.

        Returns
        -------
        PointCloud
            Current tracer positions with 'age' property.
        """
        if self._lats is None or len(self._lats) == 0:
            return PointCloud(
                xyz=np.zeros((0, 3)),
                properties={'age': np.array([])}
            )

        # Convert lat/lon to XYZ
        lats_rad = np.radians(self._lats)
        lons_rad = np.radians(self._lons)
        r = self._config.earth_radius

        x = r * np.cos(lats_rad) * np.cos(lons_rad)
        y = r * np.cos(lats_rad) * np.sin(lons_rad)
        z = r * np.sin(lats_rad)

        xyz = np.column_stack([x, y, z])

        return PointCloud(
            xyz=xyz,
            properties={'age': self._ages.copy()}
        )

    def reinitialize(
        self,
        n_points: Optional[int] = None,
        max_distance_km: Optional[float] = None,
        k_neighbors: int = 3,
    ) -> PointCloud:
        """
        Reinitialize the tracer field with a new sphere mesh.

        Generates a fresh sphere mesh and interpolates ages from existing
        tracers using inverse distance weighting. Points without nearby tracers
        (beyond max_distance_km) are dropped.

        This is useful for resampling the ocean when point density becomes
        uneven due to spreading patterns. The reinitialized points will be
        assigned plate IDs during the next step_to() call.

        Parameters
        ----------
        n_points : int, optional
            Number of points for the mesh. If None, uses config.default_mesh_points.
        max_distance_km : float, optional
            Maximum distance in kilometers to search for neighbors. Points with
            no neighbors within this distance are dropped (no age data means gap).
            If None, calculated as 2× the mesh spacing for the given number of points.
        k_neighbors : int, default=3
            Number of nearest neighbors for inverse distance weighting.
            Use k_neighbors=1 for simple nearest-neighbor interpolation.

        Returns
        -------
        PointCloud
            The reinitialized point cloud with interpolated ages.

        Raises
        ------
        RuntimeError
            If tracker is not initialized.
        ValueError
            If all points are filtered out (max_distance_km too small or no data).

        Examples
        --------
        >>> tracker.initialize(starting_age=200)
        >>> tracker.step_to(150)
        >>> # Reinitialize with higher resolution mesh
        >>> cloud = tracker.reinitialize(n_points=40000)
        >>> tracker.step_to(100)  # Continue time-stepping
        """
        if not self._initialized:
            raise RuntimeError(
                "Must call initialize() or initialize_from_cloud() before reinitialize()"
            )

        from scipy.spatial import cKDTree
        from .geometry import inverse_distance_weighted_interpolation, compute_mesh_spacing_km

        # Set defaults
        if n_points is None:
            n_points = self._config.default_mesh_points

        if max_distance_km is None:
            # Default: 2× mesh spacing
            max_distance_km = 2.0 * compute_mesh_spacing_km(n_points)

        # Convert max_distance to meters for KDTree queries
        max_distance_m = max_distance_km * 1000.0

        logger.info(
            f"Reinitializing to sphere mesh ({n_points:,} points, "
            f"max_distance={max_distance_km:.1f} km, k={k_neighbors})..."
        )

        # Check we have existing points
        if len(self._lats) == 0:
            raise ValueError("No existing tracers to interpolate from")

        # Create new sphere mesh
        mesh_lats, mesh_lons = create_sphere_mesh_latlon(n_points)
        n_mesh = len(mesh_lats)
        logger.debug(f"  Created mesh with {n_mesh} points")

        # Helper function for lat/lon to XYZ conversion
        def latlon_to_xyz(lats, lons, r):
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            x = r * np.cos(lats_rad) * np.cos(lons_rad)
            y = r * np.cos(lats_rad) * np.sin(lons_rad)
            z = r * np.sin(lats_rad)
            return np.column_stack([x, y, z])

        # Convert to XYZ for KDTree
        current_xyz = latlon_to_xyz(self._lats, self._lons, self._config.earth_radius)
        mesh_xyz = latlon_to_xyz(mesh_lats, mesh_lons, self._config.earth_radius)

        # Build KDTree from existing tracers
        tree = cKDTree(current_xyz)

        # Handle case where k_neighbors > number of existing points
        k = min(k_neighbors, len(self._lats))

        # Query K nearest neighbors for each mesh point
        distances, indices = tree.query(mesh_xyz, k=k)

        # Ensure 2D arrays even for k=1
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Determine which mesh points have valid neighbors (within max_distance)
        # A point is valid if at least one neighbor is within max_distance
        min_distances = distances[:, 0]  # Distance to nearest neighbor
        valid_mask = min_distances < max_distance_m

        n_valid = valid_mask.sum()
        if n_valid == 0:
            raise ValueError(
                f"All mesh points filtered out. No existing tracers within "
                f"{max_distance_km:.1f} km of any mesh point. "
                f"Try increasing max_distance_km or check tracer distribution."
            )

        logger.debug(f"  {n_valid}/{n_mesh} mesh points have nearby tracers")

        # Filter to valid points only
        valid_mesh_lats = mesh_lats[valid_mask]
        valid_mesh_lons = mesh_lons[valid_mask]
        valid_distances = distances[valid_mask]
        valid_indices = indices[valid_mask]

        # For IDW, only use neighbors within max_distance
        # Build values array from ages
        ages_for_interp = np.zeros_like(valid_distances)
        for i in range(n_valid):
            ages_for_interp[i] = self._ages[valid_indices[i]]

        # Mask out neighbors beyond max_distance (set distance to inf so they get zero weight)
        beyond_threshold = valid_distances >= max_distance_m
        valid_distances_masked = valid_distances.copy()
        valid_distances_masked[beyond_threshold] = np.inf

        # Compute IDW interpolation
        new_ages = inverse_distance_weighted_interpolation(ages_for_interp, valid_distances_masked)

        # Update internal state
        self._lats = valid_mesh_lats
        self._lons = valid_mesh_lons
        self._ages = new_ages

        logger.info(f"  Reinitialized to {len(self._lats)} points")

        return self.get_current_state()

    # Keep old name as alias for backwards compatibility
    def reinitialize_to_mesh(
        self,
        n_points: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_distance: Optional[float] = None,
    ) -> int:
        """
        Deprecated: Use reinitialize() instead.

        This method is kept for backwards compatibility.
        """
        import warnings
        warnings.warn(
            "reinitialize_to_mesh() is deprecated, use reinitialize() instead",
            DeprecationWarning,
            stacklevel=2
        )
        # Convert max_distance from meters to km for new API
        max_distance_km = max_distance / 1000.0 if max_distance is not None else None
        k = k_neighbors if k_neighbors is not None else 3

        self.reinitialize(
            n_points=n_points,
            max_distance_km=max_distance_km,
            k_neighbors=k,
        )
        return len(self._lats)

    @property
    def current_age(self) -> Optional[float]:
        """Current geological age."""
        return self._current_age

    @property
    def n_tracers(self) -> int:
        """Number of tracers."""
        return len(self._lats) if self._lats is not None else 0

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save state to checkpoint file.

        Parameters
        ----------
        filepath : str
            Path to save checkpoint (.npz format).
        """
        from .io_formats import PointCloudCheckpoint

        if not self._initialized:
            raise RuntimeError("No state to save - not initialized")

        cloud = self.get_current_state()
        checkpoint = PointCloudCheckpoint()
        checkpoint.save(
            cloud,
            filepath,
            geological_age=self._current_age,
            metadata={
                'time_step': self._config.time_step,
            }
        )

        logger.info(f"Saved checkpoint to {filepath} ({self._current_age} Ma)")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load state from checkpoint file.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file.
        """
        from .io_formats import PointCloudCheckpoint

        checkpoint = PointCloudCheckpoint()
        cloud, metadata = checkpoint.load(filepath)

        geological_age = metadata.get('geological_age')
        if geological_age is None:
            raise ValueError("Checkpoint missing 'geological_age' in metadata")

        self.initialize_from_cloud(cloud, geological_age)

        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"  Geological age: {self._current_age} Ma")
        logger.info(f"  Tracers: {len(self._lats)}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about current tracer state.

        Returns
        -------
        dict
            Statistics including mean age, max age, coverage, etc.
        """
        if not self._initialized or len(self._ages) == 0:
            return {
                'count': 0,
                'mean_age': 0,
                'max_age': 0,
                'min_age': 0,
                'geological_age': self._current_age
            }

        return {
            'count': len(self._ages),
            'mean_age': float(np.mean(self._ages)),
            'max_age': float(np.max(self._ages)),
            'min_age': float(np.min(self._ages)),
            'std_age': float(np.std(self._ages)),
            'geological_age': self._current_age
        }

    @classmethod
    def compute_ages(
        cls,
        target_age: float,
        starting_age: float,
        rotation_files: Union[str, List[str]],
        topology_files: Union[str, List[str]],
        continental_polygons: Optional[str] = None,
        config: Optional[TracerConfig] = None,
    ) -> PointCloud:
        """
        One-shot computation of seafloor ages (functional interface).

        Creates a tracker, initializes at starting_age, and evolves
        to target_age in a single call.

        Parameters
        ----------
        target_age : float
            Target geological age (Ma).
        starting_age : float
            Starting geological age (Ma).
        rotation_files : list of str
            Paths to rotation model files (.rot).
        topology_files : list of str
            Paths to topology/plate boundary files (.gpml/.gpmlz).
        continental_polygons : str, optional
            Path to continental polygon file.
        config : TracerConfig, optional
            Configuration parameters.

        Returns
        -------
        PointCloud
            Point cloud with 'age' property.

        Examples
        --------
        >>> cloud = SeafloorAgeTracker.compute_ages(
        ...     target_age=100,
        ...     starting_age=200,
        ...     rotation_files=['rotations.rot'],
        ...     topology_files=['topologies.gpmlz']
        ... )
        >>> ages = cloud.get_property('age')
        """
        tracker = cls(
            rotation_files=rotation_files,
            topology_files=topology_files,
            continental_polygons=continental_polygons,
            config=config,
        )

        tracker.initialize(starting_age)
        return tracker.step_to(target_age)


# Backwards compatibility aliases
HPCSeafloorAgeTracker = SeafloorAgeTracker
MemoryEfficientSeafloorAgeTracker = SeafloorAgeTracker
