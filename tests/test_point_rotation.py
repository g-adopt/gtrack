"""Tests for point rotation API."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from gtrack.point_rotation import PointCloud, PointRotator
from gtrack.polygon_filter import PolygonFilter
from gtrack.io_formats import (
    load_points_numpy,
    save_points_numpy,
    load_points_latlon,
    save_points_latlon,
    PointCloudCheckpoint,
)
from gtrack.geometry import LatLon2XYZ, XYZ2LatLon, normalize_to_sphere, EARTH_RADIUS


class TestPointCloud:
    """Test PointCloud class."""

    def test_create_from_xyz(self):
        """Test creating PointCloud from XYZ array."""
        xyz = np.random.randn(100, 3)
        xyz = normalize_to_sphere(xyz)

        cloud = PointCloud(xyz=xyz)

        assert cloud.n_points == 100
        assert cloud.xyz.shape == (100, 3)
        assert cloud.plate_ids is None
        assert len(cloud.properties) == 0

    def test_create_from_latlon(self):
        """Test creating PointCloud from lat/lon."""
        latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])

        cloud = PointCloud.from_latlon(latlon)

        assert cloud.n_points == 3
        assert cloud.xyz.shape == (3, 3)
        # Verify points are on Earth's surface
        distances = np.linalg.norm(cloud.xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)

    def test_from_xyz_basic(self):
        """Test creating PointCloud via from_xyz classmethod."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud.from_xyz(xyz)

        assert cloud.n_points == 50
        np.testing.assert_allclose(cloud.xyz, xyz)
        assert cloud.plate_ids is None
        assert len(cloud.properties) == 0

    def test_from_xyz_with_properties(self):
        """Test from_xyz passes properties correctly."""
        xyz = normalize_to_sphere(np.random.randn(30, 3))
        props = {'depth': np.random.rand(30), 'temp': np.random.rand(30)}
        cloud = PointCloud.from_xyz(xyz, properties=props)

        assert cloud.n_points == 30
        assert 'depth' in cloud.properties
        assert 'temp' in cloud.properties

    def test_from_xyz_normalize_to_earth(self):
        """Test from_xyz with normalize_to_earth scales to Earth's radius."""
        # Points on unit sphere
        xyz_unit = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cloud = PointCloud.from_xyz(xyz_unit, normalize_to_earth=True)

        distances = np.linalg.norm(cloud.xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)

    def test_from_xyz_normalize_arbitrary_radius(self):
        """Test from_xyz normalizes points at arbitrary radii to Earth's radius."""
        # Points at radius 100
        xyz = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 50.0]])
        cloud = PointCloud.from_xyz(xyz, normalize_to_earth=True)

        distances = np.linalg.norm(cloud.xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)

        # Direction should be preserved
        direction_original = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        direction_result = cloud.xyz / np.linalg.norm(cloud.xyz, axis=1, keepdims=True)
        np.testing.assert_allclose(direction_result, direction_original, atol=1e-10)

    def test_from_xyz_no_normalize(self):
        """Test from_xyz without normalization preserves coordinates exactly."""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cloud = PointCloud.from_xyz(xyz, normalize_to_earth=False)

        np.testing.assert_array_equal(cloud.xyz, xyz)

    def test_from_xyz_near_zero_vector(self):
        """Test from_xyz handles near-zero vectors without crashing."""
        xyz = np.array([[1e-15, 1e-15, 1e-15], [1.0, 0.0, 0.0]])
        cloud = PointCloud.from_xyz(xyz, normalize_to_earth=True)

        # Should not raise; second point should be at Earth's radius
        distances = np.linalg.norm(cloud.xyz, axis=1)
        np.testing.assert_allclose(distances[1], EARTH_RADIUS, rtol=1e-10)
        # First point (near-zero) should still produce a finite result
        assert np.all(np.isfinite(cloud.xyz[0]))

    def test_latlon_property(self):
        """Test that latlon property correctly converts from XYZ."""
        original_latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])

        cloud = PointCloud.from_latlon(original_latlon)
        recovered_latlon = cloud.latlon

        np.testing.assert_allclose(recovered_latlon, original_latlon, atol=1e-9)

    def test_add_property(self):
        """Test adding properties to PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        depths = np.random.rand(50) * 100e3
        cloud.add_property('depth', depths)

        assert 'depth' in cloud.properties
        np.testing.assert_array_equal(cloud.get_property('depth'), depths)

    def test_add_property_wrong_length(self):
        """Test that adding property with wrong length raises error."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        with pytest.raises(ValueError, match="has 30 values, expected 50"):
            cloud.add_property('depth', np.random.rand(30))

    def test_remove_property(self):
        """Test removing properties."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))

        cloud.remove_property('depth')

        assert 'depth' not in cloud.properties

    def test_get_property_not_found(self):
        """Test that getting non-existent property raises KeyError."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        with pytest.raises(KeyError, match="'missing'"):
            cloud.get_property('missing')

    def test_subset(self):
        """Test subsetting PointCloud with boolean mask."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.arange(100))
        cloud.plate_ids = np.arange(100)

        mask = np.zeros(100, dtype=bool)
        mask[:25] = True

        subset = cloud.subset(mask)

        assert subset.n_points == 25
        np.testing.assert_array_equal(subset.get_property('depth'), np.arange(25))
        np.testing.assert_array_equal(subset.plate_ids, np.arange(25))

    def test_copy(self):
        """Test deep copying PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        copied = cloud.copy()

        # Modify original
        cloud.xyz[0] = [0, 0, 0]
        cloud.properties['depth'][0] = -999
        cloud.plate_ids[0] = -999

        # Copy should be unchanged
        assert copied.xyz[0, 0] != 0
        assert copied.properties['depth'][0] != -999
        assert copied.plate_ids[0] != -999

    def test_len(self):
        """Test len() on PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(42, 3))
        cloud = PointCloud(xyz=xyz)

        assert len(cloud) == 42

    def test_repr(self):
        """Test string representation."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(100))
        cloud.plate_ids = np.arange(100)

        repr_str = repr(cloud)

        assert "n_points=100" in repr_str
        assert "depth" in repr_str
        assert "has_plate_ids=True" in repr_str

    def test_invalid_xyz_shape(self):
        """Test that invalid XYZ shape raises error."""
        with pytest.raises(ValueError, match="must have shape"):
            PointCloud(xyz=np.random.randn(100, 2))

    def test_create_with_properties(self):
        """Test creating PointCloud with properties in constructor."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        props = {'depth': np.random.rand(50), 'temp': np.random.rand(50)}

        cloud = PointCloud(xyz=xyz, properties=props)

        assert 'depth' in cloud.properties
        assert 'temp' in cloud.properties

    def test_concatenate_two_clouds(self):
        """Test concatenating two PointClouds with same properties."""
        xyz1 = normalize_to_sphere(np.random.randn(50, 3))
        xyz2 = normalize_to_sphere(np.random.randn(30, 3))

        cloud1 = PointCloud(xyz=xyz1)
        cloud1.add_property('temperature', np.arange(50, dtype=float))

        cloud2 = PointCloud(xyz=xyz2)
        cloud2.add_property('temperature', np.arange(50, 80, dtype=float))

        merged = PointCloud.concatenate([cloud1, cloud2])

        assert merged.n_points == 80
        assert merged.xyz.shape == (80, 3)
        np.testing.assert_allclose(merged.xyz[:50], xyz1)
        np.testing.assert_allclose(merged.xyz[50:], xyz2)
        np.testing.assert_array_equal(
            merged.get_property('temperature'),
            np.arange(80, dtype=float)
        )

    def test_concatenate_with_plate_ids(self):
        """Test concatenating clouds that all have plate_ids."""
        xyz1 = normalize_to_sphere(np.random.randn(30, 3))
        xyz2 = normalize_to_sphere(np.random.randn(20, 3))

        cloud1 = PointCloud(xyz=xyz1)
        cloud1.plate_ids = np.ones(30, dtype=int) * 101

        cloud2 = PointCloud(xyz=xyz2)
        cloud2.plate_ids = np.ones(20, dtype=int) * 201

        merged = PointCloud.concatenate([cloud1, cloud2])

        assert merged.plate_ids is not None
        assert len(merged.plate_ids) == 50
        np.testing.assert_array_equal(merged.plate_ids[:30], 101)
        np.testing.assert_array_equal(merged.plate_ids[30:], 201)

    def test_concatenate_mixed_plate_ids(self):
        """Test that mixed plate_ids (some None) results in None with warning."""
        xyz1 = normalize_to_sphere(np.random.randn(30, 3))
        xyz2 = normalize_to_sphere(np.random.randn(20, 3))

        cloud1 = PointCloud(xyz=xyz1)
        cloud1.plate_ids = np.ones(30, dtype=int)

        cloud2 = PointCloud(xyz=xyz2)
        # cloud2 has no plate_ids

        with pytest.warns(UserWarning, match="Some clouds have plate_ids"):
            merged = PointCloud.concatenate([cloud1, cloud2])

        assert merged.plate_ids is None

    def test_concatenate_different_properties(self):
        """Test that different properties result in common subset with warning."""
        xyz1 = normalize_to_sphere(np.random.randn(30, 3))
        xyz2 = normalize_to_sphere(np.random.randn(20, 3))

        cloud1 = PointCloud(xyz=xyz1)
        cloud1.add_property('common', np.ones(30))
        cloud1.add_property('only_in_1', np.zeros(30))

        cloud2 = PointCloud(xyz=xyz2)
        cloud2.add_property('common', np.ones(20) * 2)
        cloud2.add_property('only_in_2', np.zeros(20))

        with pytest.warns(UserWarning, match="not present in all clouds"):
            merged = PointCloud.concatenate([cloud1, cloud2])

        # Only common property should be present
        assert 'common' in merged.properties
        assert 'only_in_1' not in merged.properties
        assert 'only_in_2' not in merged.properties

        # Check common property values
        np.testing.assert_array_equal(merged.get_property('common')[:30], 1.0)
        np.testing.assert_array_equal(merged.get_property('common')[30:], 2.0)

    def test_concatenate_single_cloud(self):
        """Test that concatenating single cloud returns a copy."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))

        merged = PointCloud.concatenate([cloud])

        assert merged.n_points == cloud.n_points
        np.testing.assert_allclose(merged.xyz, cloud.xyz)

        # Verify it's a copy (modifying original doesn't affect merged)
        cloud.xyz[0] = [0, 0, 0]
        assert merged.xyz[0, 0] != 0

    def test_concatenate_empty_list(self):
        """Test that concatenating empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concatenate empty list"):
            PointCloud.concatenate([])

    def test_concatenate_type_error(self):
        """Test that non-PointCloud elements raise TypeError."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        with pytest.raises(TypeError, match="expected PointCloud"):
            PointCloud.concatenate([cloud, "not a cloud"])

    def test_concatenate_three_clouds(self):
        """Test concatenating three or more clouds."""
        clouds = []
        for i in range(3):
            xyz = normalize_to_sphere(np.random.randn(20, 3))
            cloud = PointCloud(xyz=xyz)
            cloud.add_property('index', np.ones(20) * i)
            clouds.append(cloud)

        merged = PointCloud.concatenate(clouds)

        assert merged.n_points == 60
        assert merged.xyz.shape == (60, 3)
        np.testing.assert_array_equal(merged.get_property('index')[:20], 0)
        np.testing.assert_array_equal(merged.get_property('index')[20:40], 1)
        np.testing.assert_array_equal(merged.get_property('index')[40:], 2)

    def test_concatenate_no_properties(self):
        """Test concatenating clouds with no properties."""
        xyz1 = normalize_to_sphere(np.random.randn(30, 3))
        xyz2 = normalize_to_sphere(np.random.randn(20, 3))

        cloud1 = PointCloud(xyz=xyz1)
        cloud2 = PointCloud(xyz=xyz2)

        merged = PointCloud.concatenate([cloud1, cloud2])

        assert merged.n_points == 50
        assert len(merged.properties) == 0


class TestPointCloudIO:
    """Test PointCloud IO functions."""

    def test_save_load_numpy_npz(self):
        """Test saving and loading to/from numpy .npz format."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            save_points_numpy(cloud, filepath)
            loaded = load_points_numpy(filepath)

        np.testing.assert_allclose(loaded.xyz, cloud.xyz)
        np.testing.assert_allclose(
            loaded.get_property('depth'),
            cloud.get_property('depth')
        )
        np.testing.assert_array_equal(loaded.plate_ids, cloud.plate_ids)

    def test_save_load_latlon_csv(self):
        """Test saving and loading lat/lon CSV."""
        latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])
        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property('depth', np.array([100.0, 200.0, 300.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_points_latlon(cloud, filepath)
            loaded = load_points_latlon(
                filepath,
                latlon_columns=(0, 1),
                property_columns={'depth': 2}
            )

        np.testing.assert_allclose(loaded.latlon, cloud.latlon, atol=1e-6)
        np.testing.assert_allclose(
            loaded.get_property('depth'),
            cloud.get_property('depth'),
            atol=1e-6
        )


class TestPointCloudCheckpoint:
    """Test checkpoint functionality."""

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint with metadata."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        checkpoint = PointCloudCheckpoint()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "checkpoint.npz"
            checkpoint.save(
                cloud, filepath,
                geological_age=50.0,
                metadata={'step': 100}
            )
            loaded, metadata = checkpoint.load(filepath)

        np.testing.assert_allclose(loaded.xyz, cloud.xyz)
        assert metadata['geological_age'] == 50.0
        assert metadata['step'] == 100

    def test_list_checkpoints(self):
        """Test listing checkpoint files."""
        checkpoint = PointCloudCheckpoint()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some checkpoint files
            for i in range(3):
                xyz = normalize_to_sphere(np.random.randn(10, 3))
                cloud = PointCloud(xyz=xyz)
                checkpoint.save(cloud, Path(tmpdir) / f"ckpt_{i:02d}.npz")

            files = checkpoint.list_checkpoints(tmpdir)

        assert len(files) == 3


class TestPointRotatorBasic:
    """Basic contract tests for the topological PointRotator.

    The rotator now advects points topologically; motion no longer depends on
    plate_ids and the old "rotate requires plate_ids" ValueError is gone. Those
    behaviours are covered in detail in test_topological_rotation.py against the
    synthetic model fixture. Here we only pin the two facts that do not need the
    full topological engine to state.
    """

    def test_rotate_no_longer_requires_plate_ids(self, synthetic_model):
        """rotate() works on a cloud with no plate_ids (motion is topological)."""
        cloud = PointCloud.from_latlon(np.array([[10.0, 10.0]]))
        assert cloud.plate_ids is None
        rotator = PointRotator(
            rotation_files=synthetic_model["rotation_files"],
            topology_files=synthetic_model["topology_files"],
        )
        out = rotator.rotate(cloud, from_age=0.0, to_age=50.0)  # must not raise
        assert out.n_points == 1

    def test_topology_files_required(self, synthetic_model):
        """Constructing without topology_files raises a clear error."""
        with pytest.raises(ValueError, match="topology_files is required"):
            PointRotator(rotation_files=synthetic_model["rotation_files"])


class TestPolygonFilterBasic:
    """Basic tests for PolygonFilter that don't require pygplates data."""

    def test_filter_preserves_properties(self):
        """Test that filtering preserves properties."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.arange(100, dtype=float))

        # Create a manual mask and use subset
        mask = np.zeros(100, dtype=bool)
        mask[::2] = True  # Keep every other point

        filtered = cloud.subset(mask)

        assert filtered.n_points == 50
        # Check properties are preserved for correct indices
        np.testing.assert_array_equal(
            filtered.get_property('depth'),
            np.arange(0, 100, 2, dtype=float)
        )


class TestCoordinateRoundTrip:
    """Test coordinate transformations between PointCloud and lat/lon."""

    def test_xyz_latlon_roundtrip(self):
        """Test XYZ -> PointCloud -> latlon -> PointCloud -> XYZ."""
        # Start with random points on sphere
        xyz_original = normalize_to_sphere(np.random.randn(100, 3))

        # Create PointCloud
        cloud1 = PointCloud(xyz=xyz_original)

        # Get latlon
        latlon = cloud1.latlon

        # Create new PointCloud from latlon
        cloud2 = PointCloud.from_latlon(latlon)

        # Compare XYZ coordinates
        np.testing.assert_allclose(cloud2.xyz, xyz_original, rtol=1e-10)

    def test_latlon_xyz_roundtrip(self):
        """Test latlon -> PointCloud -> XYZ -> latlon."""
        latlon_original = np.random.rand(100, 2) * np.array([[180, 360]]) - np.array([[90, 180]])

        # Create PointCloud from latlon
        cloud = PointCloud.from_latlon(latlon_original)

        # Get latlon back
        latlon_recovered = cloud.latlon

        np.testing.assert_allclose(latlon_recovered, latlon_original, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
