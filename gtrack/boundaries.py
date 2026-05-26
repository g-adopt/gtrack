"""
Plate boundary caching and extraction utilities.

This module provides caching for resolved topologies and utility functions
for extracting plate boundary information. The main reconstruction is handled
by pygplates' C++ backend (TopologicalModel.reconstruct_geometry).
"""

import numpy as np
import pygplates
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .geometry import load_features as _load_features
from .spatial import find_polygons


class ResolvedTopologyCache:
    """
    LRU cache for resolved topologies at different timesteps.

    Parameters
    ----------
    topology_features : pygplates.FeatureCollection or list
        Topology features for resolving.
    rotation_model : pygplates.RotationModel
        Rotation model for reconstruction.
    max_cache_size : int, default=10
        Maximum number of timesteps to cache.
    """

    def __init__(
        self,
        topology_features,
        rotation_model: pygplates.RotationModel,
        max_cache_size: int = 10,
    ):
        self.topology_features = topology_features
        self.rotation_model = rotation_model
        self.max_cache_size = max_cache_size

        # LRU cache: {time: (resolved_topologies, shared_boundary_sections)}
        self._cache: OrderedDict = OrderedDict()

    def get(self, time: float) -> Tuple[List, List]:
        """
        Get resolved topologies and shared boundary sections for a time.

        Parameters
        ----------
        time : float
            Geological time (Ma).

        Returns
        -------
        resolved_topologies : list
            List of resolved topology polygons.
        shared_boundary_sections : list
            List of shared boundary sections.
        """
        if time in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(time)
            return self._cache[time]

        # Resolve topologies
        resolved_topologies = []
        shared_boundary_sections = []
        pygplates.resolve_topologies(
            self.topology_features,
            self.rotation_model,
            resolved_topologies,
            time,
            shared_boundary_sections,
        )

        # Add to cache
        self._cache[time] = (resolved_topologies, shared_boundary_sections)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        return resolved_topologies, shared_boundary_sections

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get approximate memory usage information."""
        return {
            'cached_timesteps': len(self._cache),
            'max_cache_size': self.max_cache_size,
        }


class ContinentalPolygonCache:
    """
    Cache for reconstructed continental polygons with efficient point-in-polygon queries.

    Parameters
    ----------
    continental_polygons : str or pygplates.FeatureCollection
        Continental polygon features.
    rotation_model : pygplates.RotationModel
        Rotation model for reconstruction.
    max_cache_size : int, default=10
        Maximum number of timesteps to cache.
    """

    def __init__(
        self,
        continental_polygons,
        rotation_model: pygplates.RotationModel,
        max_cache_size: int = 10,
    ):
        # Load continental polygons. Accept a single filename, a list of
        # filenames, a pre-built FeatureCollection, or a sequence of features —
        # the same flexibility rotation_files/topology_files already have via
        # ensure_list. A bare list of filename strings must NOT be handed
        # straight to pygplates, which reads it as a sequence of features and
        # raises "Expected an optional filename, or sequence of features, or a
        # single feature".
        self._polygon_features = _load_features(continental_polygons)

        self.rotation_model = rotation_model
        self.max_cache_size = max_cache_size

        # LRU cache: {time: list of reconstructed polygon geometries}
        self._cache: OrderedDict = OrderedDict()

    def get_polygons(self, time: float) -> List[pygplates.PolygonOnSphere]:
        """
        Get reconstructed continental polygons at a time.

        Parameters
        ----------
        time : float
            Geological time (Ma).

        Returns
        -------
        list of pygplates.PolygonOnSphere
            Reconstructed continental polygon geometries.
        """
        if time in self._cache:
            self._cache.move_to_end(time)
            return self._cache[time]

        # Reconstruct polygons
        reconstructed = []
        pygplates.reconstruct(
            self._polygon_features,
            self.rotation_model,
            reconstructed,
            time,
        )

        # Extract polygon geometries
        polygon_geometries = []
        for rfg in reconstructed:
            geom = rfg.get_reconstructed_geometry()
            if isinstance(geom, pygplates.PolygonOnSphere):
                polygon_geometries.append(geom)

        # Cache
        self._cache[time] = polygon_geometries

        # Evict oldest if over capacity
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        return polygon_geometries

    def get_continental_mask(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """
        Get boolean mask indicating which points are inside continental polygons.

        Uses ptt.utils.points_in_polygons.find_polygons for efficient vectorized
        point-in-polygon testing with spatial tree acceleration.

        Parameters
        ----------
        lats : np.ndarray
            Point latitudes in degrees.
        lons : np.ndarray
            Point longitudes in degrees.
        time : float
            Geological time (Ma).

        Returns
        -------
        np.ndarray
            Boolean array, True for points inside continental polygons.
        """
        polygons = self.get_polygons(time)

        if len(polygons) == 0 or len(lats) == 0:
            return np.zeros(len(lats), dtype=bool)

        # Create all points in single C++ call using MultiPointOnSphere
        points = pygplates.MultiPointOnSphere(zip(lats, lons)).get_points()

        # Use vectorized point-in-polygon with spatial tree
        containing_polygons = find_polygons(
            points, polygons, all_polygons=False
        )

        # Convert to boolean mask: True if point is in any polygon
        mask = np.array([p is not None for p in containing_polygons], dtype=bool)

        return mask

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


def extract_ridge_geometries(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
) -> List[pygplates.PolylineOnSphere]:
    """
    Extract mid-ocean ridge geometries at a given time.

    Parameters
    ----------
    time : float
        Geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology features.
    rotation_model : pygplates.RotationModel
        Rotation model.

    Returns
    -------
    list of pygplates.PolylineOnSphere
        Ridge geometries at the specified time.
    """
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )

    ridge_geometries = []
    for shared_boundary_section in shared_boundary_sections:
        if (
            shared_boundary_section.get_feature().get_feature_type()
            == pygplates.FeatureType.create_gpml("MidOceanRidge")
        ):
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                ridge_geometries.append(shared_sub_segment.get_resolved_geometry())

    return ridge_geometries


def extract_ridge_points_latlon(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
    tessellate_degrees: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mid-ocean ridge points as lat/lon arrays.

    Parameters
    ----------
    time : float
        Geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology features.
    rotation_model : pygplates.RotationModel
        Rotation model.
    tessellate_degrees : float, default=0.5
        Tessellation resolution in degrees.

    Returns
    -------
    lats : np.ndarray
        Ridge point latitudes in degrees.
    lons : np.ndarray
        Ridge point longitudes in degrees.
    """
    ridge_geometries = extract_ridge_geometries(time, topology_features, rotation_model)

    if len(ridge_geometries) == 0:
        return np.array([]), np.array([])

    all_lats = []
    all_lons = []

    for geom in ridge_geometries:
        tessellated = geom.to_tessellated(np.radians(tessellate_degrees))
        for point in tessellated:
            lat, lon = point.to_lat_lon()
            all_lats.append(lat)
            all_lons.append(lon)

    return np.array(all_lats), np.array(all_lons)


def extract_subduction_geometries(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
) -> List[pygplates.PolylineOnSphere]:
    """
    Extract subduction zone geometries at a given time.

    Parameters
    ----------
    time : float
        Geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology features.
    rotation_model : pygplates.RotationModel
        Rotation model.

    Returns
    -------
    list of pygplates.PolylineOnSphere
        Subduction zone geometries at the specified time.
    """
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )

    subduction_geometries = []
    for shared_boundary_section in shared_boundary_sections:
        feature_type = shared_boundary_section.get_feature().get_feature_type()
        if feature_type == pygplates.FeatureType.create_gpml("SubductionZone"):
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                subduction_geometries.append(shared_sub_segment.get_resolved_geometry())

    return subduction_geometries


def extract_subduction_points_latlon(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
    tessellate_degrees: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract subduction zone points as lat/lon arrays.

    Parameters
    ----------
    time : float
        Geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology features.
    rotation_model : pygplates.RotationModel
        Rotation model.
    tessellate_degrees : float, default=0.5
        Tessellation resolution in degrees.

    Returns
    -------
    lats : np.ndarray
        Subduction zone point latitudes in degrees.
    lons : np.ndarray
        Subduction zone point longitudes in degrees.
    """
    subduction_geometries = extract_subduction_geometries(
        time, topology_features, rotation_model
    )

    if len(subduction_geometries) == 0:
        return np.array([]), np.array([])

    all_lats = []
    all_lons = []

    for geom in subduction_geometries:
        tessellated = geom.to_tessellated(np.radians(tessellate_degrees))
        for point in tessellated:
            lat, lon = point.to_lat_lon()
            all_lats.append(lat)
            all_lons.append(lon)

    return np.array(all_lats), np.array(all_lons)
