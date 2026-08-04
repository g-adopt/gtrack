"""
Initial age calculation for icosahedral mesh points.

This module provides functions to compute initial seafloor ages
from distance to the nearest ridge, matching GPlately's approach.

Sub-segments are visited through ``ordered_sub_segments`` so the ridge list
built for each plate does not depend on the order pygplates happens to return
them in. Unlike the other callers of that helper, this one reads more than the
geometry — it reads ``get_sharing_resolved_topologies()`` — so the argument
that a tied sort key is harmless has to be made separately here, and it has two
parts. The sharing topologies decide *whether* a sub-segment is appended, never
*what* is appended, and what is appended is the resolved geometry; so two
sub-segments tied on geometry contribute the same value, and if they bound
different plates they contribute to different lists entirely. Measured at
Matthews 2016, 0 Ma, where exactly this occurs: two coincident sub-segments,
one bounding plate 911 and the other 923. The list is then consumed as a
minimum over distances, which does not depend on the order it is taken in.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple

import pygplates

from .topology_order import ordered_sub_segments


def compute_initial_ages(
    ocean_points: pygplates.MultiPointOnSphere,
    resolved_topologies: List,
    shared_boundary_sections: List,
    initial_ocean_mean_spreading_rate: float = 75.0,
    fill_value: float = 5000.0,
    age_distance_law: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute initial seafloor ages from distance to nearest ridge.

    For each ocean point, finds the nearest mid-ocean ridge within the same
    plate polygon and calculates age using the distance/spreading-rate formula.

    Parameters
    ----------
    ocean_points : pygplates.MultiPointOnSphere
        Ocean basin points (after filtering out continental points).
    resolved_topologies : list
        Resolved topologies from pygplates.resolve_topologies().
    shared_boundary_sections : list
        Shared boundary sections from pygplates.resolve_topologies().
    initial_ocean_mean_spreading_rate : float, default=75.0
        Mean spreading rate in mm/yr (numerically equal to km/Myr).
        Default is GPlately's value of 75 mm/yr.
    fill_value : float, default=5000.0
        Distance value (km) for points in plates without ridges.
        Corresponds to ~133 Myr at 75 mm/yr spreading rate.
    age_distance_law : callable, optional
        Custom function to convert distances to ages.
        Signature: (distances_km, spreading_rate_mm_yr) -> ages_myr
        If None, uses default: age = distance / (rate / 2)

    Returns
    -------
    lons : np.ndarray
        Longitudes of ocean points in degrees.
    lats : np.ndarray
        Latitudes of ocean points in degrees.
    ages : np.ndarray
        Computed ages in Myr.

    Notes
    -----
    The default age formula is:
        age_myr = distance_km / (spreading_rate_mm_yr / 2)

    The division by 2 accounts for half-spreading rate (each plate moves
    at half the full spreading rate). The units work because:
        75 mm/yr = 75 km/Myr (numerically equal)

    Examples
    --------
    >>> # Resolve topologies at starting time
    >>> resolved_topologies = []
    >>> shared_boundary_sections = []
    >>> pygplates.resolve_topologies(
    ...     topology_features, rotation_model, resolved_topologies,
    ...     starting_age, shared_boundary_sections
    ... )
    >>> lons, lats, ages = compute_initial_ages(
    ...     ocean_points, resolved_topologies, shared_boundary_sections
    ... )
    """
    all_lons = []
    all_lats = []
    all_distances = []

    # Create point features from MultiPointOnSphere for the query
    point_feature = pygplates.Feature()
    point_feature.set_geometry(ocean_points)
    point_features = [point_feature]

    # Process each plate topology
    for topology in resolved_topologies:
        plate_id = topology.get_resolved_feature().get_reconstruction_plate_id()

        # Find mid-ocean ridges that bound this plate
        mid_ocean_ridges_on_plate = []
        for shared_boundary_section in shared_boundary_sections:
            if (
                shared_boundary_section.get_feature().get_feature_type()
                == pygplates.FeatureType.create_gpml("MidOceanRidge")
            ):
                for shared_subsegment in ordered_sub_segments(shared_boundary_section):
                    sharing_resolved_topologies = (
                        shared_subsegment.get_sharing_resolved_topologies()
                    )
                    for resolved_polygon in sharing_resolved_topologies:
                        if (
                            resolved_polygon.get_feature().get_reconstruction_plate_id()
                            == plate_id
                        ):
                            mid_ocean_ridges_on_plate.append(
                                shared_subsegment.get_resolved_geometry()
                            )

        # Process points within this plate
        for pf in point_features:
            for points in pf.get_geometries():
                for point in points:
                    if topology.get_resolved_geometry().is_point_in_polygon(point):
                        lat, lon = point.to_lat_lon()

                        if len(mid_ocean_ridges_on_plate) > 0:
                            # Find minimum distance to any ridge
                            min_distance = None
                            for ridge in mid_ocean_ridges_on_plate:
                                distance = pygplates.GeometryOnSphere.distance(
                                    point, ridge, min_distance
                                )
                                if distance is not None:
                                    min_distance = distance

                            # Convert to km
                            distance_km = min_distance * pygplates.Earth.mean_radius_in_kms
                        else:
                            # No ridges bounding this plate - use fill value
                            distance_km = fill_value

                        all_lons.append(lon)
                        all_lats.append(lat)
                        all_distances.append(distance_km)

    # Convert to arrays
    lons = np.array(all_lons)
    lats = np.array(all_lats)
    distances = np.array(all_distances)

    # Compute ages using provided or default formula
    if age_distance_law is not None:
        ages = age_distance_law(distances, initial_ocean_mean_spreading_rate)
    else:
        ages = default_age_distance_law(distances, initial_ocean_mean_spreading_rate)

    return lons, lats, ages


def default_age_distance_law(
    distances_km: np.ndarray,
    spreading_rate_mm_yr: float
) -> np.ndarray:
    """
    Default age-distance law: age = distance / (spreading_rate / 2).

    This is GPlately's formula, where the half-spreading rate is used
    because each plate moves at half the full spreading rate.

    Parameters
    ----------
    distances_km : np.ndarray
        Distances to nearest ridge in km.
    spreading_rate_mm_yr : float
        Full spreading rate in mm/yr (numerically equal to km/Myr).

    Returns
    -------
    np.ndarray
        Ages in Myr.

    Notes
    -----
    The formula works because 1 mm/yr = 1 km/Myr numerically:
        75 mm/yr × 1e6 yr/Myr × 1e-6 km/mm = 75 km/Myr
    """
    half_spreading_rate = spreading_rate_mm_yr / 2.0
    return distances_km / half_spreading_rate


def compute_initial_ages_kdtree(
    ocean_lats: np.ndarray,
    ocean_lons: np.ndarray,
    ridge_lats: np.ndarray,
    ridge_lons: np.ndarray,
    initial_ocean_mean_spreading_rate: float = 75.0,
    earth_radius_km: float = 6371.0,
    age_distance_law: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute initial ages using a fast KDTree approach.

    This is a simpler alternative that doesn't use plate polygon queries.
    It finds the globally nearest ridge point for each ocean point.

    Parameters
    ----------
    ocean_lats : np.ndarray
        Ocean point latitudes in degrees.
    ocean_lons : np.ndarray
        Ocean point longitudes in degrees.
    ridge_lats : np.ndarray
        Ridge point latitudes in degrees.
    ridge_lons : np.ndarray
        Ridge point longitudes in degrees.
    initial_ocean_mean_spreading_rate : float, default=75.0
        Mean spreading rate in mm/yr.
    earth_radius_km : float, default=6371.0
        Earth radius in km.
    age_distance_law : callable, optional
        Custom function to convert distances to ages.

    Returns
    -------
    ages : np.ndarray
        Computed ages in Myr.

    Notes
    -----
    This method is faster but less accurate than the plate-based approach
    because it doesn't respect plate boundaries. Use for quick estimates
    or when plate topology information is not available.
    """
    from scipy.spatial import cKDTree

    # Convert to Cartesian XYZ on unit sphere
    def latlon_to_xyz(lats, lons):
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        x = np.cos(lats_rad) * np.cos(lons_rad)
        y = np.cos(lats_rad) * np.sin(lons_rad)
        z = np.sin(lats_rad)
        return np.column_stack([x, y, z])

    ocean_xyz = latlon_to_xyz(ocean_lats, ocean_lons)
    ridge_xyz = latlon_to_xyz(ridge_lats, ridge_lons)

    # Build KDTree and query
    tree = cKDTree(ridge_xyz)
    chord_distances, _ = tree.query(ocean_xyz, k=1)

    # Convert chord distance to great circle distance
    # chord = 2 * sin(angle/2), so angle = 2 * arcsin(chord/2)
    angles_rad = 2 * np.arcsin(np.clip(chord_distances / 2, 0, 1))
    distances_km = angles_rad * earth_radius_km

    # Compute ages
    if age_distance_law is not None:
        ages = age_distance_law(distances_km, initial_ocean_mean_spreading_rate)
    else:
        ages = default_age_distance_law(distances_km, initial_ocean_mean_spreading_rate)

    return ages
