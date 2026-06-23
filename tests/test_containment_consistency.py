"""Tests for consistent polygon containment across gtrack.

These tests verify that PolygonFilter and ContinentalPolygonCache produce
identical containment results, addressing the API mismatch described in
ISSUES.md section 1.1.
"""

import numpy as np
import pytest
import pygplates
import tempfile
import os

from gtrack.boundaries import ContinentalPolygonCache
from gtrack.polygon_filter import PolygonFilter
from gtrack.point_rotation import PointCloud
from gtrack.geometry import LatLon2XYZ, EARTH_RADIUS


def _create_polygon_feature(vertices_latlon, plate_id=0):
    """Create a pygplates polygon feature from (lat, lon) vertices."""
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in vertices_latlon]
    polygon = pygplates.PolygonOnSphere(points)
    feature = pygplates.Feature()
    feature.set_geometry(polygon)
    feature.set_reconstruction_plate_id(plate_id)
    return feature


def _write_features_to_gpml(features, path):
    """Write a list of features to a .gpml file."""
    fc = pygplates.FeatureCollection(features)
    fc.write(path)


def _create_empty_rotation_file(path):
    """Write a minimal .rot file with an identity rotation for plate 0."""
    with open(path, "w") as f:
        f.write("0  0.0  0.0  0.0  0.0  999  !identity\n")


def _create_rotation_file_with_motion(path):
    """Write a .rot file where plate 1 rotates 10 degrees around the north pole by 50 Ma."""
    with open(path, "w") as f:
        # Plate 0 (anchor) — identity at all times
        f.write("0  0.0  0.0  0.0  0.0  999  !anchor identity\n")
        # Plate 1 — 10 degree rotation around north pole (lat=90, lon=0) at 50 Ma
        f.write("1  0.0  0.0  0.0  0.0  0  !plate1 at 0 Ma\n")
        f.write("1  50.0  90.0  0.0  10.0  0  !plate1 at 50 Ma\n")


@pytest.fixture
def simple_polygon_env(tmp_path):
    """Create a test environment with a single rectangular continental polygon.

    The polygon covers roughly lat [-10, 10], lon [-10, 10], centred on (0, 0).
    """
    vertices = [
        (10.0, -10.0),
        (10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, -10.0),
    ]
    feature = _create_polygon_feature(vertices, plate_id=0)

    gpml_path = str(tmp_path / "continent.gpml")
    rot_path = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature], gpml_path)
    _create_empty_rotation_file(rot_path)

    return gpml_path, rot_path


@pytest.fixture
def rotated_polygon_env(tmp_path):
    """Create a test environment where plate 1 rotates 10 degrees by 50 Ma.

    The polygon (on plate 1) covers roughly lat [-10, 10], lon [-10, 10] at
    present day (0 Ma). At 50 Ma it is shifted ~10 degrees in longitude.
    """
    vertices = [
        (10.0, -10.0),
        (10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, -10.0),
    ]
    feature = _create_polygon_feature(vertices, plate_id=1)

    gpml_path = str(tmp_path / "continent.gpml")
    rot_path = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature], gpml_path)
    _create_rotation_file_with_motion(rot_path)

    return gpml_path, rot_path


@pytest.fixture
def overlapping_polygon_env(tmp_path):
    """Create a test environment with two overlapping continental polygons.

    Polygon A (plate 0): lat [-10, 10], lon [-10, 10]
    Polygon B (plate 0): lat [-5, 15], lon [-5, 15]
    Overlap region: lat [-5, 10], lon [-5, 10]
    """
    vertices_a = [
        (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0), (-10.0, -10.0),
    ]
    vertices_b = [
        (15.0, -5.0), (15.0, 15.0), (-5.0, 15.0), (-5.0, -5.0),
    ]
    feature_a = _create_polygon_feature(vertices_a, plate_id=0)
    feature_b = _create_polygon_feature(vertices_b, plate_id=0)

    gpml_path = str(tmp_path / "continents.gpml")
    rot_path = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature_a, feature_b], gpml_path)
    _create_empty_rotation_file(rot_path)

    return gpml_path, rot_path


class TestContainmentConsistency:
    """Verify PolygonFilter and ContinentalPolygonCache agree."""

    def test_clearly_inside_points(self, simple_polygon_env):
        """Points well inside the polygon are detected by both APIs."""
        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # Points clearly inside the [-10,10] x [-10,10] polygon
        lats = np.array([0.0, 5.0, -5.0, 3.0])
        lons = np.array([0.0, 5.0, -5.0, -3.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        # ContinentalPolygonCache path
        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        # PolygonFilter path
        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        np.testing.assert_array_equal(mask_cache, mask_filter)
        assert mask_cache.all(), "All interior points should be inside"

    def test_clearly_outside_points(self, simple_polygon_env):
        """Points well outside the polygon are rejected by both APIs."""
        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # Points clearly outside the [-10,10] x [-10,10] polygon
        lats = np.array([45.0, -45.0, 0.0, 80.0])
        lons = np.array([0.0, 0.0, 90.0, 170.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        np.testing.assert_array_equal(mask_cache, mask_filter)
        assert not mask_cache.any(), "All exterior points should be outside"

    def test_near_boundary_points_agree(self, simple_polygon_env):
        """Points near the boundary produce the same result from both APIs."""
        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # Points near the boundary (but not exactly on it)
        # Slightly inside
        offset = 0.5  # degrees inside the boundary
        lats_in = np.array([0.0, 9.5, -9.5, 0.0])
        lons_in = np.array([9.5, 0.0, 0.0, -9.5])
        # Slightly outside
        lats_out = np.array([0.0, 10.5, -10.5, 0.0])
        lons_out = np.array([10.5, 0.0, 0.0, -10.5])

        lats = np.concatenate([lats_in, lats_out])
        lons = np.concatenate([lons_in, lons_out])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        np.testing.assert_array_equal(
            mask_cache, mask_filter,
            err_msg="Near-boundary containment must be identical between APIs"
        )

    def test_overlapping_polygons_agree(self, overlapping_polygon_env):
        """Containment is consistent when polygons overlap."""
        gpml_path, rot_path = overlapping_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # Points in various regions:
        # In polygon A only (inside A, outside B)
        # In polygon B only (inside B, outside A)
        # In both (overlap region)
        # In neither
        lats = np.array([0.0, -8.0, 12.0, 3.0, 45.0])
        lons = np.array([0.0, -8.0, 12.0, 3.0, 90.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        np.testing.assert_array_equal(
            mask_cache, mask_filter,
            err_msg="Overlapping polygon containment must be identical between APIs"
        )

    def test_empty_cloud(self, simple_polygon_env):
        """Empty PointCloud produces empty mask from both APIs."""
        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        lats = np.array([])
        lons = np.array([])
        xyz = np.empty((0, 3))
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        assert len(mask_cache) == 0
        assert len(mask_filter) == 0

    def test_filter_inside_uses_consistent_api(self, simple_polygon_env):
        """PolygonFilter.filter_inside returns the same subset as manual cache masking."""
        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # Mix of inside and outside points
        lats = np.array([0.0, 5.0, 45.0, -5.0, 80.0])
        lons = np.array([0.0, 5.0, 90.0, -5.0, 170.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)
        cloud.add_property("thickness", np.array([100.0, 120.0, 50.0, 130.0, 20.0]))

        # Manual cache path
        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)
        expected_xyz = cloud.xyz[mask_cache]

        # PolygonFilter.filter_inside path
        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        filtered = pf.filter_inside(cloud, at_age=0.0)

        np.testing.assert_array_equal(filtered.xyz, expected_xyz)
        np.testing.assert_array_equal(
            filtered.get_property("thickness"),
            cloud.get_property("thickness")[mask_cache],
        )

    def test_nonzero_age_consistency(self, rotated_polygon_env):
        """Both APIs agree at a non-zero reconstruction time where the polygon has moved."""
        gpml_path, rot_path = rotated_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        # At 50 Ma, plate 1 is rotated ~10 degrees in longitude.
        # Points that are inside at 0 Ma may be outside at 50 Ma and vice versa.
        # We test a grid of points spanning the relevant region.
        lats = np.linspace(-15, 15, 10)
        lons = np.linspace(-20, 25, 15)
        grid_lat, grid_lon = np.meshgrid(lats, lons)
        flat_lats = grid_lat.ravel()
        flat_lons = grid_lon.ravel()

        latlon = np.column_stack([flat_lats, flat_lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(flat_lats, flat_lons, 50.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=50.0)

        np.testing.assert_array_equal(
            mask_cache, mask_filter,
            err_msg="Containment must agree at non-zero reconstruction time"
        )
        # Sanity: the polygon has moved, so the mask at 50 Ma should differ from 0 Ma
        mask_0 = cache.get_continental_mask(flat_lats, flat_lons, 0.0)
        assert not np.array_equal(mask_cache, mask_0), (
            "Mask at 50 Ma should differ from 0 Ma (the polygon moved)"
        )

    def test_filter_outside_uses_consistent_api(self, simple_polygon_env):
        """PolygonFilter.filter_outside returns the complement of filter_inside."""
        gpml_path, rot_path = simple_polygon_env

        lats = np.array([0.0, 5.0, 45.0, -5.0, 80.0])
        lons = np.array([0.0, 5.0, 90.0, -5.0, 170.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        inside = pf.filter_inside(cloud, at_age=0.0)
        outside = pf.filter_outside(cloud, at_age=0.0)

        assert inside.n_points + outside.n_points == cloud.n_points


class TestContainmentLargeScale:
    """Test containment consistency with many uniformly distributed points."""

    def test_fibonacci_mesh_consistency(self, simple_polygon_env):
        """A full Fibonacci mesh produces identical masks from both APIs."""
        from gtrack.mesh import create_sphere_mesh_latlon

        gpml_path, rot_path = simple_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)

        n_points = 5000
        lats, lons = create_sphere_mesh_latlon(n_points)
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        cache = ContinentalPolygonCache(features, rotation_model)
        mask_cache = cache.get_continental_mask(lats, lons, 0.0)

        pf = PolygonFilter(polygon_files=gpml_path, rotation_files=rot_path)
        mask_filter = pf.get_containment_mask(cloud, at_age=0.0)

        np.testing.assert_array_equal(
            mask_cache, mask_filter,
            err_msg=(
                f"Masks disagree on {np.sum(mask_cache != mask_filter)} of "
                f"{n_points} points"
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
