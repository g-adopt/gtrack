"""
Point rotation API for rotating user-provided points through geological time.

This module provides the core API for:
- Managing collections of points with associated properties (PointCloud)
- Rotating points between geological ages using plate reconstructions (PointRotator)

The API uses Cartesian XYZ coordinates internally for compatibility with gadopt,
while interfacing with pygplates using lat/lon coordinates.

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

    # Create point features for partitioning
    point_features = []
    for lat, lon in zip(lats, lons):
        point_feature = pygplates.Feature()
        point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
        point_features.append(point_feature)

    # Partition into plates
    assigned_point_features = pygplates.partition_into_plates(
        partitioning_features,
        rotation_model,
        point_features,
        reconstruction_time=time,
        properties_to_copy=[pygplates.PartitionProperty.reconstruction_plate_id]
    )

    # Extract plate IDs
    plate_ids = np.array([
        f.get_reconstruction_plate_id() for f in assigned_point_features
    ], dtype=int)

    return plate_ids


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

    Key Features:
    - Cartesian XYZ internal representation (matches gadopt)
    - Properties stored separately from positions and preserved during rotation
    - Batched rotation for performance (10-50x speedup by grouping by plate_id)
    - Handles undefined plates with warnings

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot).
    topology_files : list of str, optional
        Paths to topology/plate boundary files (.gpmlz).
        Used for partitioning if static_polygons not provided.
    static_polygons : str, optional
        Path to static polygons for plate ID assignment.
        If None, uses topology_files for partitioning.

    Examples
    --------
    >>> rotator = PointRotator(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     static_polygons='static_polygons.gpmlz'
    ... )
    >>>
    >>> # Load user points
    >>> cloud = PointCloud.from_latlon(my_latlon_array)
    >>>
    >>> # Assign plate IDs at present day
    >>> cloud = rotator.assign_plate_ids(cloud, at_age=0.0)
    >>>
    >>> # Rotate to 50 Ma
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

        # Load topology features if provided
        if topology_files:
            self.topology_features = pygplates.FeatureCollection()
            for file in topology_files:
                features = pygplates.FeatureCollection(file)
                self.topology_features.add(features)
        else:
            self.topology_features = None

        # Load static polygons if provided. Accept a single filename, a list of
        # filenames, or pre-built features — same flexibility as the rotation
        # and topology arguments above (a bare list of filenames would
        # otherwise reach pygplates as a sequence of features and be rejected).
        if static_polygons is not None:
            self.static_polygons = load_features(static_polygons)
        else:
            self.static_polygons = None

        # Ensure we have some features for partitioning
        if self.static_polygons is None and self.topology_features is None:
            raise ValueError(
                "Either topology_files or static_polygons must be provided "
                "for plate ID assignment."
            )

    def assign_plate_ids(
        self,
        cloud: PointCloud,
        at_age: float,
        use_static_polygons: bool = True,
        remove_undefined: bool = True,
        partitioning_features: Optional["pygplates.FeatureCollection"] = None
    ) -> PointCloud:
        """
        Assign plate IDs to points based on their positions.

        Parameters
        ----------
        cloud : PointCloud
            Points to assign plate IDs to.
        at_age : float
            Geological age at which to assign plate IDs (Ma).
            Use 0.0 for present-day positions.
        use_static_polygons : bool, default=True
            If True and static_polygons are loaded, use them for assignment.
            Otherwise use topology_features. Ignored if partitioning_features
            is provided.
        remove_undefined : bool, default=True
            If True, remove points with undefined plate IDs (plate_id=0)
            and emit a warning.
            If False, keep points with plate_id=0.
        partitioning_features : pygplates.FeatureCollection, optional
            Explicit polygon features to use for plate ID assignment.
            When provided, overrides use_static_polygons. Use this to
            assign plate IDs from the same polygon set used for containment
            filtering (e.g., continental polygons), ensuring that rotated
            points stay aligned with the polygon boundaries.

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
        # Validate explicit partitioning features
        if partitioning_features is not None:
            if not isinstance(partitioning_features, pygplates.FeatureCollection):
                raise TypeError(
                    f"partitioning_features must be a pygplates.FeatureCollection, "
                    f"got {type(partitioning_features).__name__}"
                )
        elif use_static_polygons and self.static_polygons is not None:
            partitioning_features = self.static_polygons
        else:
            partitioning_features = self.topology_features

        plate_ids = _get_plate_ids(
            cloud.xyz,
            partitioning_features,
            self.rotation_model,
            at_age
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

    def rotate(
        self,
        cloud: PointCloud,
        from_age: float,
        to_age: float,
        reassign_plate_ids: bool = False
    ) -> PointCloud:
        """
        Rotate points from one geological age to another.

        Uses batched rotation operations for performance (10-50x speedup
        by grouping points by plate_id).

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. Must have plate_ids assigned.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        reassign_plate_ids : bool, default=False
            If True, reassign plate IDs at target age.
            If False, keep original plate IDs.

        Returns
        -------
        PointCloud
            Rotated points with same properties.

        Raises
        ------
        ValueError
            If cloud does not have plate_ids assigned.

        Notes
        -----
        Direction of rotation:
        - from_age=0, to_age=50: Rotate present-day positions to 50 Ma
        - from_age=50, to_age=0: Rotate 50 Ma positions to present day

        Examples
        --------
        >>> # Rotate present-day continental points to 50 Ma
        >>> rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)
        """
        if cloud.plate_ids is None:
            raise ValueError(
                "Cloud must have plate_ids assigned. "
                "Call assign_plate_ids() first."
            )

        # Apply batched rotation (groups by plate_id for efficiency)
        rotated_xyz = _move_points_batched(
            cloud.xyz,
            self.rotation_model,
            cloud.plate_ids,
            from_age,
            to_age
        )

        # Create result cloud with rotated positions
        result = PointCloud(
            xyz=rotated_xyz,
            properties={k: v.copy() for k, v in cloud.properties.items()},
            plate_ids=cloud.plate_ids.copy()
        )

        # Optionally reassign plate IDs at new age
        if reassign_plate_ids:
            result = self.assign_plate_ids(result, at_age=to_age)

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
        Rotate points incrementally through geological time.

        Useful when plate IDs may change during rotation (e.g.,
        points crossing plate boundaries over long time spans).

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. Must have plate_ids assigned.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        time_step : float, default=1.0
            Time step for incremental rotation (Myr).
        reassign_at_each_step : bool, default=True
            If True, reassign plate IDs after each time step.
            This handles points that cross plate boundaries.

        Returns
        -------
        PointCloud
            Rotated points.

        Examples
        --------
        >>> # Rotate with 1 Myr steps, reassigning plates
        >>> rotated = rotator.rotate_incremental(
        ...     cloud, from_age=0.0, to_age=100.0, time_step=1.0
        ... )
        """
        if cloud.plate_ids is None:
            raise ValueError("Cloud must have plate_ids assigned.")

        result = cloud.copy()
        current_age = from_age
        direction = 1 if to_age > from_age else -1

        while (direction > 0 and current_age < to_age) or \
              (direction < 0 and current_age > to_age):

            # Calculate step size (don't overshoot)
            remaining = abs(to_age - current_age)
            actual_step = min(time_step, remaining) * direction
            next_age = current_age + actual_step

            # Rotate one step
            result = self.rotate(
                result,
                from_age=current_age,
                to_age=next_age,
                reassign_plate_ids=reassign_at_each_step
            )

            current_age = next_age

        return result
