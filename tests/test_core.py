"""Tests for core functionality."""

import numpy as np
import pytest
from gtrack import TracerConfig
from gtrack.geometry import LatLon2XYZ, XYZ2LatLon


class TestTracerConfig:
    """Test TracerConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = TracerConfig()

        assert config.tracker_step_myr == 1.0
        assert config.earth_radius_m == 6.3781e6
        assert config.collision_velocity_difference_km_per_myr == 7.0
        assert config.collision_distance_rate_km_per_myr == 10.0
        assert config.tracker_point_count == 10000
        assert config.initial_spreading_rate_mm_per_yr == 75.0
        assert config.ridge_sampling_angle_deg == 0.5
        assert config.ridge_offset_angle_deg == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracerConfig(
            tracker_step_myr=0.5,
            tracker_point_count=40000,
            ridge_sampling_angle_deg=0.25
        )

        assert config.tracker_step_myr == 0.5
        assert config.tracker_point_count == 40000
        assert config.ridge_sampling_angle_deg == 0.25
        # Other values should be defaults
        assert config.collision_velocity_difference_km_per_myr == 7.0

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="tracker_step_myr"):
            TracerConfig(tracker_step_myr=-1)

        # Zero mesh points should raise error
        with pytest.raises(ValueError, match="tracker_point_count"):
            TracerConfig(tracker_point_count=0)

        # Non-positive spreading rate should raise error
        with pytest.raises(ValueError, match="initial_spreading_rate_mm_per_yr"):
            TracerConfig(initial_spreading_rate_mm_per_yr=0)

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = TracerConfig(tracker_step_myr=0.5)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['time_step'] == 0.5
        assert 'velocity_delta_threshold' in config_dict
        assert 'default_mesh_points' in config_dict

    def test_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'time_step': 0.5,
            'default_mesh_points': 40000,
        }
        config = TracerConfig.from_dict(config_dict)

        assert config.tracker_step_myr == 0.5
        assert config.tracker_point_count == 40000

    def test_velocity_threshold_conversion(self):
        """Test velocity threshold unit conversion property."""
        config = TracerConfig(collision_velocity_difference_km_per_myr=7.0)

        # 7.0 km/Myr should convert to 0.7 cm/yr
        assert config.collision_velocity_difference_cm_per_yr == 0.7


class TestMeshGeneration:
    """Test Fibonacci sphere mesh generation."""

    def test_create_sphere_mesh_latlon(self):
        """Test sphere mesh creation returning lat/lon."""
        from gtrack.mesh import create_sphere_mesh_latlon

        n_points = 1000
        lats, lons = create_sphere_mesh_latlon(n_points)

        # Check shapes match
        assert len(lats) == len(lons)

        # Check we get the expected number of points
        assert len(lats) == n_points

        # Check latitude range
        assert np.all(lats >= -90)
        assert np.all(lats <= 90)

        # Check longitude range
        assert np.all(lons >= -180)
        assert np.all(lons <= 180)

    def test_create_sphere_mesh_xyz(self):
        """Test sphere mesh creation returning XYZ."""
        from gtrack.mesh import create_sphere_mesh_xyz

        n_points = 1000
        xyz = create_sphere_mesh_xyz(n_points)

        # Check shape
        assert xyz.shape == (n_points, 3)

        # Check points are on unit sphere (default radius=1)
        radii = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(radii, 1.0, rtol=1e-10)

    def test_create_sphere_mesh_with_radius(self):
        """Test sphere mesh with custom radius."""
        from gtrack.mesh import create_sphere_mesh_xyz

        n_points = 500
        radius = 6.3781e6  # Earth radius
        xyz = create_sphere_mesh_xyz(n_points, radius=radius)

        # Check points are on sphere with given radius
        radii = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(radii, radius, rtol=1e-10)

    def test_create_sphere_mesh_arbitrary_count(self):
        """Test that we can create meshes with any point count."""
        from gtrack.mesh import create_sphere_mesh_xyz

        for n_points in [1, 10, 100, 12345, 40000]:
            xyz = create_sphere_mesh_xyz(n_points)
            assert len(xyz) == n_points


class TestMORSeeds:
    """Test MOR seed point generation functions."""

    def test_get_stage_rotation_returns_none_without_plates(self):
        """Test that get_stage_rotation returns None for features without plate IDs."""
        from gtrack.mor_seeds import get_stage_rotation_for_reconstructed_geometry
        import pygplates

        # Create a mock feature without plate IDs
        feature = pygplates.Feature()
        feature.set_geometry(pygplates.PointOnSphere(0, 0))

        # Create a simple rotation model (won't be used since we lack plate IDs)
        rotation_model = pygplates.RotationModel([])

        result = get_stage_rotation_for_reconstructed_geometry(
            feature, rotation_model, spreading_time=100.0
        )

        assert result is None


class TestInitialConditions:
    """Test initial age calculation functions."""

    def test_default_age_distance_law(self):
        """Test the default age-distance formula."""
        from gtrack.initial_conditions import default_age_distance_law

        distances_km = np.array([75.0, 150.0, 300.0])
        spreading_rate_mm_yr = 75.0  # = 75 km/Myr

        ages = default_age_distance_law(distances_km, spreading_rate_mm_yr)

        # age = distance / (rate / 2)
        # For 75 km at 75 mm/yr half-rate (37.5 km/Myr): 75 / 37.5 = 2 Myr
        expected = np.array([2.0, 4.0, 8.0])
        np.testing.assert_allclose(ages, expected)


class TestBoundaries:
    """Test boundary caching and extraction."""

    def test_continental_polygon_cache_empty(self):
        """Test continental cache behavior with no polygons."""
        from gtrack.boundaries import ContinentalPolygonCache
        import pygplates

        # Create empty feature collection
        empty_features = pygplates.FeatureCollection()
        rotation_model = pygplates.RotationModel([])

        cache = ContinentalPolygonCache(empty_features, rotation_model)

        # Query should return empty polygons
        polygons = cache.get_polygons(100.0)
        assert len(polygons) == 0

        # Mask should be all False (no continental points)
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 90.0, -90.0])
        mask = cache.get_continental_mask(lats, lons, 100.0)

        assert mask.shape == (3,)
        assert not mask.any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
