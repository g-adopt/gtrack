"""
Point rotation API for rotating user-provided points through geological time.

This module provides the core API for:
- Managing collections of points with associated properties (PointCloud)
- Rotating points between geological ages using plate reconstructions (PointRotator)

The API uses Cartesian XYZ coordinates internally for compatibility with gadopt,
while interfacing with pygplates using lat/lon coordinates.

Rotation engine
---------------
``PointRotator`` advects points with a single, deforming-aware engine:
``pygplates.TopologicalModel.reconstruct_geometry``. It resolves topologies —
rigid plates *and* deforming networks — and moves each point by whatever plate
or network it sits in, exactly as the ocean ``SeafloorAgeTracker`` does. This
replaces the earlier rigid, per-plate-id engine, which silently dropped points
whose static-polygon plate id had no rigid rotation sequence to the anchor and
which mis-rotated the (majority) continental points whose static-polygon id
disagreed with the topology id. See ``CHANGE-PLAN.md`` for the evidence.

Consequences of the single engine:
- Motion no longer depends on ``plate_ids``. ``rotate`` neither requires nor
  drops points; with ``deactivate_points=None`` every input point is returned.
- ``plate_ids`` are a *labelling* output only (``assign_plate_ids``), no longer
  a motion input, and default to the topology-consistent partition.

Terminology:
- geological_age (or just 'age'): Time before present in Ma (0 = present, 100 = 100 Myr ago)
- from_age / to_age: Source and target geological ages for rotation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pygplates


@dataclass
class PointCloud:
    """
    Container for points with associated properties.

    Stores points in Cartesian XYZ format internally (matches gadopt).
    Properties (lithospheric_depth, etc.) are stored separately from positions.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates, shape (N, 3), in meters.
        Points should lie on Earth's surface (radius ~6.3781e6 m).
    properties : dict, optional
        Dictionary mapping property names to arrays of shape (N,).
        Properties are preserved during rotation operations.
    plate_ids : np.ndarray, optional
        Plate IDs for each point, shape (N,). Required for rotation.

    Examples
    --------
    >>> xyz = np.random.randn(1000, 3)
    >>> from gtrack.geometry import normalize_to_sphere
    >>> xyz = normalize_to_sphere(xyz)  # Project to Earth's surface
    >>> cloud = PointCloud(xyz=xyz)
    >>> cloud.add_property('lithospheric_depth', np.random.rand(1000) * 100e3)
    """

    xyz: np.ndarray
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    plate_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        # Ensure xyz is a numpy array
        self.xyz = np.asarray(self.xyz)

        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(
                f"xyz must have shape (N, 3), got {self.xyz.shape}"
            )

        n_points = len(self.xyz)

        # Validate properties
        for name, prop in self.properties.items():
            prop = np.asarray(prop)
            if len(prop) != n_points:
                raise ValueError(
                    f"Property '{name}' has {len(prop)} values, expected {n_points}"
                )
            self.properties[name] = prop

        # Validate plate_ids if provided
        if self.plate_ids is not None:
            self.plate_ids = np.asarray(self.plate_ids)
            if len(self.plate_ids) != n_points:
                raise ValueError(
                    f"plate_ids has {len(self.plate_ids)} values, expected {n_points}"
                )

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.xyz)

    @property
    def latlon(self) -> np.ndarray:
        """
        Get lat/lon coordinates (computed from XYZ).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with [lat, lon] in degrees.
            Latitude: -90 to 90, Longitude: -180 to 180.
        """
        from .geometry import XYZ2LatLon
        lats, lons = XYZ2LatLon(self.xyz)
        return np.column_stack([lats, lons])

    @property
    def lonlat(self) -> np.ndarray:
        """
        Get lon/lat coordinates (computed from XYZ).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with [lon, lat] in degrees.
            Longitude: -180 to 180, Latitude: -90 to 90.
        """
        from .geometry import XYZ2LatLon
        lats, lons = XYZ2LatLon(self.xyz)
        return np.column_stack([lons, lats])

    @classmethod
    def from_latlon(
        cls,
        latlon: np.ndarray,
        properties: Optional[Dict[str, np.ndarray]] = None
    ) -> "PointCloud":
        """
        Create PointCloud from lat/lon coordinates.

        Parameters
        ----------
        latlon : np.ndarray
            Coordinates, shape (N, 2) with [lat, lon] in degrees.
        properties : dict, optional
            Properties to attach to the points.

        Returns
        -------
        PointCloud
            New PointCloud with XYZ coordinates computed from lat/lon.

        Examples
        --------
        >>> latlon = np.array([[45.0, -120.0], [30.0, 90.0]])
        >>> cloud = PointCloud.from_latlon(latlon)
        """
        from .geometry import LatLon2XYZ
        latlon = np.asarray(latlon)
        xyz = LatLon2XYZ(latlon)
        return cls(xyz=xyz, properties=properties or {})

    @classmethod
    def from_xyz(
        cls,
        xyz: np.ndarray,
        properties: Optional[Dict[str, np.ndarray]] = None,
        normalize_to_earth: bool = False
    ) -> "PointCloud":
        """
        Create PointCloud from Cartesian XYZ coordinates.

        This is a convenience classmethod that provides symmetry with from_latlon.
        It can optionally normalize points to Earth's surface.

        Parameters
        ----------
        xyz : np.ndarray
            Cartesian coordinates, shape (N, 3).
        properties : dict, optional
            Properties to attach to the points.
        normalize_to_earth : bool, default=False
            If True, normalize points to Earth's radius (~6.3781e6 m).
            Useful when input points are on a unit sphere.

        Returns
        -------
        PointCloud
            New PointCloud with the given XYZ coordinates.

        Examples
        --------
        >>> # Points already at Earth's radius
        >>> xyz = np.array([[6378100, 0, 0], [0, 6378100, 0]])
        >>> cloud = PointCloud.from_xyz(xyz)
        >>>
        >>> # Points on unit sphere, scale to Earth
        >>> xyz_unit = np.array([[1, 0, 0], [0, 1, 0]])
        >>> cloud = PointCloud.from_xyz(xyz_unit, normalize_to_earth=True)
        """
        from .geometry import EARTH_RADIUS
        xyz = np.asarray(xyz)

        if normalize_to_earth:
            # Normalize to unit sphere, then scale to Earth's radius
            norms = np.linalg.norm(xyz, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            xyz = xyz / norms * EARTH_RADIUS

        return cls(xyz=xyz, properties=properties or {})

    @classmethod
    def from_data(
        cls,
        data,
        property_name: str,
        n_points_fallback: int = 20000
    ) -> "PointCloud":
        """
        Create PointCloud from whichever of four shapes the caller has.

        Callers configuring a source rarely hold a PointCloud already. They
        hold a gridded data file, or a pair of arrays, or a single number
        meaning "this value everywhere". This dispatches all of them, so the
        choice of input shape stops being every caller's problem.

        Parameters
        ----------
        data : PointCloud or str or Path or tuple or int or float
            One of:

            - a PointCloud, returned unchanged;
            - a path to an HDF5/NetCDF lat/lon grid, loaded via
              :func:`~gtrack.io_formats.load_latlon_grid_hdf5`;
            - a ``(latlon, values)`` tuple, with latlon of shape (N, 2);
            - a scalar, broadcast onto a Fibonacci sphere mesh of
              ``n_points_fallback`` points.
        property_name : str
            Name the property is attached under. For a file this is also the
            dataset looked for inside it.
        n_points_fallback : int, default=20000
            Mesh size used ONLY for the scalar case. Every other input brings
            its own points and ignores this.

        Returns
        -------
        PointCloud
            Cloud carrying ``property_name``.

        Raises
        ------
        TypeError
            If ``data`` is none of the four accepted shapes.

        Examples
        --------
        >>> cloud = PointCloud.from_data(50.0, 'thickness', n_points_fallback=5000)
        >>> cloud = PointCloud.from_data((latlon, values), 'thickness')
        >>> cloud = PointCloud.from_data('thickness_grid.h5', 'thickness')
        """
        from pathlib import Path

        from .io_formats import load_latlon_grid_hdf5
        from .mesh import create_sphere_mesh_latlon

        if isinstance(data, cls):
            return data
        if isinstance(data, (str, Path)):
            return load_latlon_grid_hdf5(data, property_name)
        if isinstance(data, tuple) and len(data) == 2:
            latlon, values = data
            cloud = cls.from_latlon(np.asarray(latlon))
            cloud.add_property(property_name, np.asarray(values))
            return cloud
        # bool is a subclass of int; a boolean here is a caller error, not a
        # thickness of 1 metre everywhere.
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            lats, lons = create_sphere_mesh_latlon(n_points_fallback)
            latlon = np.column_stack([lats, lons])
            cloud = cls.from_latlon(latlon)
            cloud.add_property(property_name, np.full(len(latlon), float(data)))
            return cloud
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected PointCloud, file path, (latlon, values) tuple, or scalar."
        )

    def add_property(self, name: str, values: np.ndarray) -> None:
        """
        Add or update a property.

        Parameters
        ----------
        name : str
            Name of the property.
        values : np.ndarray
            Property values, shape (N,).

        Raises
        ------
        ValueError
            If values length doesn't match number of points.
        """
        values = np.asarray(values)
        if len(values) != self.n_points:
            raise ValueError(
                f"Property '{name}' has {len(values)} values, expected {self.n_points}"
            )
        self.properties[name] = values

    def remove_property(self, name: str) -> None:
        """
        Remove a property.

        Parameters
        ----------
        name : str
            Name of the property to remove.
        """
        if name in self.properties:
            del self.properties[name]

    def get_property(self, name: str) -> np.ndarray:
        """
        Get a property by name.

        Parameters
        ----------
        name : str
            Name of the property.

        Returns
        -------
        np.ndarray
            Property values.

        Raises
        ------
        KeyError
            If property not found.
        """
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found")
        return self.properties[name]

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """
        Create subset of points using boolean mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask, shape (N,). True values are kept.

        Returns
        -------
        PointCloud
            New PointCloud with subset of points.
        """
        mask = np.asarray(mask, dtype=bool)
        new_xyz = self.xyz[mask]
        new_properties = {
            name: prop[mask] for name, prop in self.properties.items()
        }
        new_plate_ids = self.plate_ids[mask] if self.plate_ids is not None else None
        return PointCloud(
            xyz=new_xyz,
            properties=new_properties,
            plate_ids=new_plate_ids
        )

    def copy(self) -> "PointCloud":
        """
        Create a deep copy.

        Returns
        -------
        PointCloud
            Deep copy of this PointCloud.
        """
        return PointCloud(
            xyz=self.xyz.copy(),
            properties={k: v.copy() for k, v in self.properties.items()},
            plate_ids=self.plate_ids.copy() if self.plate_ids is not None else None
        )

    @classmethod
    def concatenate(cls, clouds: List["PointCloud"], warn: bool = True) -> "PointCloud":
        """
        Merge multiple PointClouds into a single PointCloud.

        Parameters
        ----------
        clouds : list of PointCloud
            List of PointCloud instances to merge.
        warn : bool, default=True
            Whether to emit warnings when properties or plate_ids are
            dropped due to inconsistency between clouds. Set to False
            when this is expected behavior.

        Returns
        -------
        PointCloud
            New PointCloud with concatenated data from all input clouds.

        Raises
        ------
        ValueError
            If clouds list is empty.
        TypeError
            If any element is not a PointCloud.

        Notes
        -----
        - The xyz arrays are vertically stacked.
        - Properties are merged: only properties present in ALL clouds are included
          in the result. If clouds have different property sets, a warning is issued
          and only the common properties are kept.
        - plate_ids are concatenated only if ALL clouds have them; otherwise
          the result has plate_ids=None.

        Examples
        --------
        >>> cloud1 = PointCloud(xyz=np.random.randn(100, 3))
        >>> cloud1.add_property('temperature', np.random.rand(100))
        >>> cloud2 = PointCloud(xyz=np.random.randn(50, 3))
        >>> cloud2.add_property('temperature', np.random.rand(50))
        >>> merged = PointCloud.concatenate([cloud1, cloud2])
        >>> merged.n_points
        150
        """
        # Validate input
        if not clouds:
            raise ValueError("Cannot concatenate empty list of PointClouds")

        if len(clouds) == 1:
            return clouds[0].copy()

        # Type check
        for i, cloud in enumerate(clouds):
            if not isinstance(cloud, cls):
                raise TypeError(
                    f"Element {i} is {type(cloud).__name__}, expected PointCloud"
                )

        # Concatenate xyz arrays
        xyz_arrays = [cloud.xyz for cloud in clouds]
        merged_xyz = np.vstack(xyz_arrays)

        # Find common properties across all clouds
        all_property_sets = [set(cloud.properties.keys()) for cloud in clouds]
        common_properties = set.intersection(*all_property_sets) if all_property_sets else set()

        # Check if any properties are being dropped
        all_properties = set.union(*all_property_sets) if all_property_sets else set()
        dropped_properties = all_properties - common_properties

        if dropped_properties and warn:
            warnings.warn(
                f"Properties {dropped_properties} are not present in all clouds "
                f"and will be excluded from the merged result. "
                f"Only common properties {common_properties} are included.",
                UserWarning
            )

        # Merge common properties
        merged_properties = {}
        for prop_name in common_properties:
            prop_arrays = [cloud.properties[prop_name] for cloud in clouds]
            merged_properties[prop_name] = np.concatenate(prop_arrays)

        # Handle plate_ids: only include if ALL clouds have them
        all_have_plate_ids = all(cloud.plate_ids is not None for cloud in clouds)
        if all_have_plate_ids:
            plate_id_arrays = [cloud.plate_ids for cloud in clouds]
            merged_plate_ids = np.concatenate(plate_id_arrays)
        else:
            merged_plate_ids = None
            # Warn if some but not all have plate_ids
            some_have_plate_ids = any(cloud.plate_ids is not None for cloud in clouds)
            if some_have_plate_ids and warn:
                warnings.warn(
                    "Some clouds have plate_ids and some do not. "
                    "The merged cloud will have plate_ids=None.",
                    UserWarning
                )

        return cls(
            xyz=merged_xyz,
            properties=merged_properties,
            plate_ids=merged_plate_ids
        )

    def __len__(self) -> int:
        """Return number of points."""
        return self.n_points

    def __repr__(self) -> str:
        """String representation."""
        props = list(self.properties.keys())
        has_plate_ids = self.plate_ids is not None
        return (
            f"PointCloud(n_points={self.n_points}, "
            f"properties={props}, "
            f"has_plate_ids={has_plate_ids})"
        )


# =============================================================================
# Helper functions for plate operations
# =============================================================================


def _get_plate_ids(
    xyz: np.ndarray,
    partitioning_features,
    rotation_model: pygplates.RotationModel,
    time: float
) -> np.ndarray:
    """
    Assign plate IDs to points based on their positions.

    Uses pygplates.partition_into_plates for efficient bulk plate ID assignment.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates, shape (N, 3).
    partitioning_features : pygplates.FeatureCollection
        Static polygons or topology features for partitioning.
    rotation_model : pygplates.RotationModel
        Rotation model.
    time : float
        Reconstruction time in Ma.

    Returns
    -------
    plate_ids : np.ndarray
        Array of plate IDs, shape (N,). Unassigned points get plate_id=0.
    """
    from .geometry import XYZ2LatLon

    # Convert XYZ to lat/lon
    lats, lons = XYZ2LatLon(xyz)

    # Create point features for partitioning. Tag each with its input index in
    # the feature name: partition_into_plates does NOT preserve input order (it
    # groups partitioned features ahead of unpartitioned ones), so we must map
    # results back to input positions ourselves or the returned plate ids would
    # be misaligned with the points.
    point_features = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        point_feature = pygplates.Feature()
        point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
        point_feature.set_name(str(i))
        point_features.append(point_feature)

    # Partition into plates
    assigned_point_features = pygplates.partition_into_plates(
        partitioning_features,
        rotation_model,
        point_features,
        reconstruction_time=time,
        properties_to_copy=[pygplates.PartitionProperty.reconstruction_plate_id]
    )

    # Extract plate IDs, restoring input order via the index tag.
    plate_ids = np.zeros(len(lats), dtype=int)
    for f in assigned_point_features:
        idx = int(f.get_name())
        plate_ids[idx] = f.get_reconstruction_plate_id()

    return plate_ids


# =============================================================================
# NON-PRODUCTION rigid helpers — retained for regression tests ONLY.
#
# `_rotate_points_batch`, `_move_points_batched` and `_unreconstructable_plate_ids`
# implement the *old* rigid, per-plate-id rotation engine. Production `rotate`
# is now topological (`PointRotator._advect_topological`) and does NOT use any of
# these. They are kept because the regression suite needs them to (a) document
# the old drop behaviour that the new engine fixes and (b) assert that
# topological advection equals rigid rotation when fed the topology's own plate
# ids (the decisive Phase 0 consistency check). Do NOT rewire them into
# `rotate` — that would resurrect the drop/mismatch bug.
# =============================================================================


def _rotate_points_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    rotation: pygplates.FiniteRotation
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation to a batch of points using MultiPointOnSphere.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes in degrees, shape (N,).
    lons : np.ndarray
        Longitudes in degrees, shape (N,).
    rotation : pygplates.FiniteRotation
        Rotation to apply.

    Returns
    -------
    rotated_lats : np.ndarray
        Rotated latitudes, shape (N,).
    rotated_lons : np.ndarray
        Rotated longitudes, shape (N,).
    """
    if len(lats) == 0:
        return np.array([]), np.array([])

    # Create MultiPointOnSphere for batch rotation (single C++ call)
    multi_point = pygplates.MultiPointOnSphere(zip(lats, lons))

    # Apply rotation (single C++ operation)
    rotated_multi_point = rotation * multi_point

    # Extract rotated coordinates
    rotated_lats = np.empty(len(lats))
    rotated_lons = np.empty(len(lats))
    for i, point in enumerate(rotated_multi_point.get_points()):
        rotated_lats[i], rotated_lons[i] = point.to_lat_lon()

    return rotated_lats, rotated_lons


def _unreconstructable_plate_ids(
    plate_ids: np.ndarray,
    rotation_model: pygplates.RotationModel,
    time: float
) -> set:
    """
    Find plate IDs that are not part of the plate circuit at a given time.

    A plate (typically an accreted terrane or a young deforming block) only
    exists back to its appearance age. Before that age its rotation circuit to
    the anchor plate is incomplete, and ``rotation_model.get_rotation`` silently
    returns an *identity* rotation. Applying that identity leaves the point
    pinned at its present-day position while the rest of its continent drifts
    away, producing points stranded in open ocean with no polygon around them.

    This uses the reconstruction tree at ``time`` to ask, unambiguously, whether
    each plate is connected to the anchor. A plate with no edge in the tree (and
    which is not the anchor itself) does not exist at ``time``. This is distinct
    from a plate that exists but happens to be stationary (identity rotation but
    a valid tree edge), which must NOT be flagged.

    Parameters
    ----------
    plate_ids : np.ndarray
        Plate IDs to test, shape (N,).
    rotation_model : pygplates.RotationModel
        Rotation model.
    time : float
        Reconstruction time in Ma.

    Returns
    -------
    set of int
        The subset of unique plate IDs that are not reconstructable at ``time``.
    """
    tree = rotation_model.get_reconstruction_tree(time)
    anchor = tree.get_anchor_plate_id()
    bad = set()
    for pid in np.unique(plate_ids):
        pid = int(pid)
        if pid == anchor:
            continue
        if tree.get_edge(pid) is None:
            bad.add(pid)
    return bad


def _move_points_batched(
    xyz: np.ndarray,
    rotation_model: pygplates.RotationModel,
    plate_ids: np.ndarray,
    from_age: float,
    to_age: float
) -> np.ndarray:
    """
    Move points according to plate motions using batched rotation operations.

    This provides significant speedup by grouping points by plate_id and
    computing rotation once per plate instead of per point.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates, shape (N, 3).
    rotation_model : pygplates.RotationModel
        Rotation model.
    plate_ids : np.ndarray
        Plate IDs for each point, shape (N,).
    from_age : float
        Source time in Ma.
    to_age : float
        Target time in Ma.

    Returns
    -------
    new_xyz : np.ndarray
        Updated positions, shape (N, 3).
    """
    from .geometry import XYZ2LatLon, LatLon2XYZ

    # Convert all positions to lat/lon once
    lats, lons = XYZ2LatLon(xyz)

    # Pre-allocate output arrays
    new_lats = lats.copy()
    new_lons = lons.copy()

    # Group points by unique plate IDs
    unique_plates = np.unique(plate_ids)

    # Process each plate's points in batch
    for plate_id in unique_plates:
        # Get rotation for this plate (computed ONCE per plate)
        rotation = rotation_model.get_rotation(to_age, plate_id, from_age)

        # Find all points on this plate
        mask = (plate_ids == plate_id)

        # Get positions for this plate's points
        plate_lats = lats[mask]
        plate_lons = lons[mask]

        # Apply rotation to all points on this plate
        rotated_lats, rotated_lons = _rotate_points_batch(
            plate_lats, plate_lons, rotation
        )

        # Update positions
        new_lats[mask] = rotated_lats
        new_lons[mask] = rotated_lons

    # Convert back to XYZ
    new_latlon = np.column_stack([new_lats, new_lons])
    new_xyz = LatLon2XYZ(new_latlon)

    return new_xyz


class PointRotator:
    """
    Rotate points between geological ages using plate reconstructions.

    This class provides the main API for rotating user-provided points
    according to plate tectonic reconstructions.

    Motion is deforming-aware: points are advected with
    ``pygplates.TopologicalModel.reconstruct_geometry``, resolving rigid plates
    and deforming networks, exactly like the ocean tracker. Motion does not
    depend on ``plate_ids`` and no point is ever silently dropped.

    Key Features:
    - Cartesian XYZ internal representation (matches gadopt)
    - Properties stored separately from positions and preserved during rotation
    - Single topological engine (no rigid per-plate fallback)
    - No silent drops: ``rotate`` with ``deactivate_points=None`` returns every
      input point

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot).
    topology_files : list of str
        Paths to topology/plate boundary files (.gpml/.gpmlz). **Required** —
        the topological engine is built from these. A clear ``ValueError`` is
        raised if they are absent.
    static_polygons : str, optional
        Path to static polygons. Used only for the optional
        ``assign_plate_ids(source="static")`` labelling path, never for motion.

    Examples
    --------
    >>> rotator = PointRotator(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ... )
    >>>
    >>> # Load user points
    >>> cloud = PointCloud.from_latlon(my_latlon_array)
    >>>
    >>> # Rotate to 50 Ma (no plate_ids needed — motion is topological)
    >>> rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)
    """

    def __init__(
        self,
        rotation_files: Union[str, List[str]],
        topology_files: Optional[Union[str, List[str]]] = None,
        static_polygons: Optional[str] = None
    ):
        import pygplates
        from .geometry import ensure_list, load_features

        # Handle single file or Path as list
        rotation_files = ensure_list(rotation_files)
        topology_files = ensure_list(topology_files)

        self.rotation_model = pygplates.RotationModel(rotation_files)

        # Topology features are required — the deforming-aware engine is built
        # from them. Without them there is no motion path (the old rigid,
        # static-polygon-driven path has been removed; see the module docstring
        # and CHANGE-PLAN.md).
        if not topology_files:
            raise ValueError(
                "topology_files is required: PointRotator advects points with a "
                "topological model built from the plate-boundary/network files. "
                "Pass the same topology files you pass to SeafloorAgeTracker."
            )
        self.topology_features = load_features(topology_files)

        # Build the topological model once (mirrors SeafloorAgeTracker).
        self._topological_model = pygplates.TopologicalModel(
            self.topology_features, self.rotation_model
        )

        # Load static polygons if provided. Accept a single filename, a list of
        # filenames, or pre-built features — same flexibility as the rotation
        # and topology arguments above. These are only used for the optional
        # static-polygon labelling path in ``assign_plate_ids``.
        if static_polygons is not None:
            self.static_polygons = load_features(static_polygons)
        else:
            self.static_polygons = None

    def _topology_plate_ids(self, xyz: np.ndarray, at_age: float) -> np.ndarray:
        """Assign plate ids by partitioning against the *resolved topologies*.

        This yields the plate/network id that the topological engine actually
        moves each point by, so labels are consistent with motion. Points that
        fall in no resolved topology get 0.
        """
        from .geometry import XYZ2LatLon

        lats, lons = XYZ2LatLon(xyz)
        snapshot = pygplates.TopologicalSnapshot(
            self.topology_features, self.rotation_model, at_age
        )
        resolved = snapshot.get_resolved_topologies()
        partitioner = pygplates.PlatePartitioner(resolved, self.rotation_model)

        plate_ids = np.zeros(len(lats), dtype=int)
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            located = partitioner.partition_point(pygplates.PointOnSphere(lat, lon))
            if located is not None:
                plate_ids[i] = located.get_feature().get_reconstruction_plate_id()
        return plate_ids

    def assign_plate_ids(
        self,
        cloud: PointCloud,
        at_age: float,
        source: str = "topology",
        remove_undefined: bool = False,
        partitioning_features: Optional["pygplates.FeatureCollection"] = None,
        use_static_polygons: Optional[bool] = None,
    ) -> PointCloud:
        """
        Assign plate IDs to points based on their positions.

        Plate IDs are a *labelling convenience only* — an output property. They
        are no longer a motion input: ``rotate`` advects points topologically
        and does not consult ``plate_ids``. Assigning them is therefore optional.

        Parameters
        ----------
        cloud : PointCloud
            Points to assign plate IDs to.
        at_age : float
            Geological age at which to assign plate IDs (Ma).
            Use 0.0 for present-day positions.
        source : {"topology", "static"}, default="topology"
            "topology" partitions against the resolved topologies (rigid plates
            and deforming networks), so the assigned id matches what the engine
            moves the point by. "static" uses the static polygons passed at
            construction (back-compat; requires ``static_polygons``). Ignored if
            ``partitioning_features`` is given.
        remove_undefined : bool, default=False
            If True, remove points with undefined plate IDs (plate_id=0) and emit
            a warning. Default is False: since ids are labels, dropping points
            here would silently shrink the cloud. Left for callers that
            explicitly want the old behaviour.
        partitioning_features : pygplates.FeatureCollection, optional
            Explicit polygon features to use for plate ID assignment, overriding
            ``source``. Partitioned with ``partition_into_plates``.
        use_static_polygons : bool, optional
            Deprecated back-compat alias. If True, equivalent to
            ``source="static"``.

        Returns
        -------
        PointCloud
            Cloud with plate_ids assigned. May have fewer points if
            remove_undefined=True and some points had undefined plates.

        Warns
        -----
        UserWarning
            If any points have undefined plate IDs.
        """
        # Back-compat: use_static_polygons=True maps to source="static".
        if use_static_polygons:
            source = "static"

        # Validate explicit partitioning features
        if partitioning_features is not None:
            if not isinstance(partitioning_features, pygplates.FeatureCollection):
                raise TypeError(
                    f"partitioning_features must be a pygplates.FeatureCollection, "
                    f"got {type(partitioning_features).__name__}"
                )
            plate_ids = _get_plate_ids(
                cloud.xyz, partitioning_features, self.rotation_model, at_age
            )
        elif source == "static":
            if self.static_polygons is None:
                raise ValueError(
                    'assign_plate_ids(source="static") requires static_polygons '
                    "to have been passed to PointRotator."
                )
            plate_ids = _get_plate_ids(
                cloud.xyz, self.static_polygons, self.rotation_model, at_age
            )
        elif source == "topology":
            plate_ids = self._topology_plate_ids(cloud.xyz, at_age)
        else:
            raise ValueError(
                f'source must be "topology" or "static", got {source!r}'
            )

        # Check for undefined plates (plate_id = 0)
        undefined_mask = (plate_ids == 0)
        n_undefined = np.sum(undefined_mask)

        if n_undefined > 0:
            action = "These points will be removed." if remove_undefined else "They will be assigned plate_id=0."
            warnings.warn(
                f"{n_undefined} points ({100*n_undefined/cloud.n_points:.1f}%) "
                f"have undefined plate IDs at {at_age} Ma. {action}",
                UserWarning
            )

        if remove_undefined and n_undefined > 0:
            valid_mask = ~undefined_mask
            result = cloud.subset(valid_mask)
            result.plate_ids = plate_ids[valid_mask]
        else:
            result = cloud.copy()
            result.plate_ids = plate_ids

        return result

    def _advect_topological(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        from_age: float,
        to_age: float,
        time_step: float = 1.0,
        deactivate_points=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advect points from ``from_age`` to ``to_age`` with the topological model.

        Mirrors the SeafloorAgeTracker stepping loop. Returns
        ``(out_lat, out_lon, active)``. Inactive points (only possible when a
        deactivation policy is supplied) come back as NaN with ``active=False``.

        ``reconstruct_geometry`` requires ``(oldest_time - youngest_time)`` to be
        an integer multiple of ``time_increment``; we derive an integer
        ``n_steps`` and an exact increment ``span / n_steps`` to satisfy it.
        """
        if not (time_step > 0):
            raise ValueError(
                f"time_step must be a positive number, got {time_step!r}. "
                "It sets the internal stepping granularity of the topological "
                "reconstruction (Myr)."
            )

        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = len(lats)

        lo, hi = (from_age, to_age) if from_age <= to_age else (to_age, from_age)
        span_len = hi - lo
        if span_len == 0 or n == 0:
            return lats.copy(), lons.copy(), np.ones(n, dtype=bool)

        n_steps = max(1, int(np.ceil(span_len / float(time_step))))
        increment = span_len / n_steps

        points = pygplates.MultiPointOnSphere(list(zip(lats, lons)))
        time_span = self._topological_model.reconstruct_geometry(
            points,
            initial_time=from_age,
            oldest_time=hi,
            youngest_time=lo,
            time_increment=increment,
            deactivate_points=deactivate_points,   # None => keep everything
        )
        recon = time_span.get_geometry_points(to_age, return_inactive_points=True)

        out_lat = np.full(n, np.nan)
        out_lon = np.full(n, np.nan)
        active = np.zeros(n, dtype=bool)
        for i, p in enumerate(recon):
            if p is not None:
                out_lat[i], out_lon[i] = p.to_lat_lon()
                active[i] = True
        return out_lat, out_lon, active

    def rotate(
        self,
        cloud: PointCloud,
        from_age: float,
        to_age: float,
        reassign_plate_ids: bool = False,
        *,
        time_step: float = 1.0,
        deactivate_points=None,
    ) -> PointCloud:
        """
        Rotate points from one geological age to another (deforming-aware).

        Points are advected with ``pygplates.TopologicalModel.reconstruct_geometry``,
        resolving rigid plates and deforming networks. Motion does not depend on
        ``plate_ids``; they are neither required nor consulted.

        No silent drops: with the default ``deactivate_points=None`` every input
        point is returned (``n_out == n_in``), with properties and ``plate_ids``
        passed through unchanged and in input order. Points that fall outside any
        resolved topology simply keep their position (they do not vanish).

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. ``plate_ids`` are optional.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        reassign_plate_ids : bool, default=False
            If True, (re)assign topology-consistent plate IDs at ``to_age`` as an
            output label. Never drops points.
        time_step : float, default=1.0
            Internal stepping granularity (Myr) for the topological reconstruction.
        deactivate_points : optional
            A ``pygplates`` deactivation policy. Default ``None`` keeps every
            point. If supplied, inactive points are removed and the result cloud
            is subset (properties + plate_ids in lockstep).

        Returns
        -------
        PointCloud
            Rotated points with the same properties. Same length as the input
            unless a deactivation policy removed points.

        Notes
        -----
        Direction of rotation (works both ways):
        - from_age=0, to_age=50: rotate present-day positions to 50 Ma
        - from_age=50, to_age=0: rotate 50 Ma positions to present day

        Examples
        --------
        >>> # Rotate present-day continental points to 50 Ma
        >>> rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)
        """
        from .geometry import LatLon2XYZ

        latlon = cloud.latlon
        in_lat = latlon[:, 0]
        in_lon = latlon[:, 1]

        out_lat, out_lon, active = self._advect_topological(
            in_lat, in_lon, from_age, to_age,
            time_step=time_step, deactivate_points=deactivate_points,
        )

        if deactivate_points is None:
            # Contract: keep every point. Any point pygplates could not place
            # (should not happen with no deactivation policy) retains its
            # original position rather than becoming NaN, so n_out == n_in.
            inactive = ~active
            if inactive.any():
                out_lat[inactive] = in_lat[inactive]
                out_lon[inactive] = in_lon[inactive]
            keep = np.ones(cloud.n_points, dtype=bool)
        else:
            keep = active

        new_xyz = LatLon2XYZ(np.column_stack([out_lat[keep], out_lon[keep]]))
        result = PointCloud(
            xyz=new_xyz,
            properties={k: v[keep].copy() for k, v in cloud.properties.items()},
            plate_ids=(cloud.plate_ids[keep].copy()
                       if cloud.plate_ids is not None else None),
        )

        # Optionally (re)assign plate IDs at the new age — label only, no drops.
        if reassign_plate_ids:
            result = self.assign_plate_ids(
                result, at_age=to_age, remove_undefined=False
            )

        return result

    def rotate_incremental(
        self,
        cloud: PointCloud,
        from_age: float,
        to_age: float,
        time_step: float = 1.0,
        reassign_at_each_step: bool = True
    ) -> PointCloud:
        """
        Rotate points through geological time in ``time_step`` increments.

        Retained for back-compat. The topological engine already steps
        internally at ``time_step`` granularity, so this delegates to
        :meth:`rotate` with the same ``time_step``; there is no longer a separate
        rigid per-step path. ``reassign_at_each_step`` no longer changes the
        trajectory (motion is topological, not plate-id driven); it only controls
        whether the *output* label is refreshed at ``to_age``.

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. ``plate_ids`` are optional.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        time_step : float, default=1.0
            Internal stepping granularity (Myr).
        reassign_at_each_step : bool, default=True
            If True, assign topology-consistent plate IDs at ``to_age`` on output.

        Returns
        -------
        PointCloud
            Rotated points.
        """
        return self.rotate(
            cloud,
            from_age=from_age,
            to_age=to_age,
            reassign_plate_ids=reassign_at_each_step,
            time_step=time_step,
        )
