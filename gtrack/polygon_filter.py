"""
Polygon filtering operations for point clouds.

This module provides functionality to filter points based on polygon containment,
useful for selecting points inside continental polygons, exclusion zones, etc.
"""

from typing import List, Union
import warnings

import numpy as np


class PolygonFilter:
    """
    Filter points by polygon containment.

    Supports filtering by:
    - Continental polygons (keep only continental points)
    - Custom polygons (user-provided)
    - Exclusion zones (remove points inside polygons)

    Parameters
    ----------
    polygon_files : str or list of str
        Path(s) to polygon files (.gpml, .gpmlz).
    rotation_files : str or list of str
        Paths to rotation model files (.rot).

    Examples
    --------
    >>> filter = PolygonFilter(
    ...     polygon_files='continental_polygons.gpmlz',
    ...     rotation_files=['rotations.rot']
    ... )
    >>>
    >>> # Keep only continental points
    >>> continental_cloud = filter.filter_inside(cloud, at_age=0.0)
    >>>
    >>> # Remove continental points (keep oceanic)
    >>> oceanic_cloud = filter.filter_outside(cloud, at_age=0.0)
    """

    def __init__(
        self,
        polygon_files: Union[str, List[str]],
        rotation_files: Union[str, List[str]]
    ):
        import pygplates
        from .geometry import ensure_list, load_features
        from .boundaries import ContinentalPolygonCache

        # Handle single file or Path as list
        rotation_files = ensure_list(rotation_files)

        self.rotation_model = pygplates.RotationModel(rotation_files)

        # Load polygon features (accepts a single filename, a list of
        # filenames, prebuilt features, or a FeatureCollection).
        self.polygon_features = load_features(polygon_files)

        # Use the same containment test as ContinentalPolygonCache to ensure
        # consistent results between ocean excision and continental filtering.
        self._continental_cache = ContinentalPolygonCache(
            self.polygon_features, self.rotation_model
        )

    def get_containment_mask(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> np.ndarray:
        """
        Get boolean mask of points inside polygons.

        Uses the same reconstruct-then-test approach as
        ContinentalPolygonCache.get_continental_mask to ensure consistent
        containment results across gtrack.

        Parameters
        ----------
        cloud : PointCloud
            Points to check.
        at_age : float
            Geological age at which to check containment (Ma).
            Use 0.0 for present-day polygons.

        Returns
        -------
        np.ndarray
            Boolean mask, shape (N,), True for points inside polygons.
        """
        from .geometry import XYZ2LatLon

        lats, lons = XYZ2LatLon(cloud.xyz)
        return self._continental_cache.get_continental_mask(lats, lons, at_age)

    def filter_inside(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> "PointCloud":
        """
        Keep only points inside polygons.

        Parameters
        ----------
        cloud : PointCloud
            Points to filter.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        PointCloud
            Points inside polygons.

        Examples
        --------
        >>> # Keep only continental points at present day
        >>> continental = filter.filter_inside(cloud, at_age=0.0)
        """
        mask = self.get_containment_mask(cloud, at_age)

        if not mask.any():
            warnings.warn(
                f"No points found inside polygons at {at_age} Ma. "
                f"Returning empty PointCloud.",
                UserWarning
            )

        return cloud.subset(mask)

    def filter_outside(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> "PointCloud":
        """
        Keep only points outside polygons.

        Parameters
        ----------
        cloud : PointCloud
            Points to filter.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        PointCloud
            Points outside polygons.

        Examples
        --------
        >>> # Remove continental points (keep oceanic) at present day
        >>> oceanic = filter.filter_outside(cloud, at_age=0.0)
        """
        mask = self.get_containment_mask(cloud, at_age)
        n_outside = np.sum(~mask)

        if n_outside == 0:
            warnings.warn(
                f"All points are inside polygons at {at_age} Ma. "
                f"Returning empty PointCloud.",
                UserWarning
            )

        return cloud.subset(~mask)

    def get_statistics(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> dict:
        """
        Get statistics about polygon containment.

        Parameters
        ----------
        cloud : PointCloud
            Points to analyze.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        dict
            Statistics including:
            - total: Total number of points
            - inside: Number of points inside polygons
            - outside: Number of points outside polygons
            - inside_fraction: Fraction of points inside
        """
        mask = self.get_containment_mask(cloud, at_age)
        n_inside = int(np.sum(mask))
        n_total = cloud.n_points

        return {
            'total': n_total,
            'inside': n_inside,
            'outside': n_total - n_inside,
            'inside_fraction': n_inside / n_total if n_total > 0 else 0.0,
            'at_age': at_age
        }
