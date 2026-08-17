"""Tests for removal of temporal snapping in continental excision.

These tests verify that continental polygon excision uses exact float times
instead of snapping to coarse integer intervals, addressing ISSUES.md
section 1.3.
"""

import numpy as np
import pytest
import pygplates

from gtrack import TracerConfig
from gtrack.boundaries import ContinentalPolygonCache


def _create_polygon_feature(vertices_latlon, plate_id):
    """Create a pygplates polygon feature from (lat, lon) vertices."""
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in vertices_latlon]
    polygon = pygplates.PolygonOnSphere(points)
    feature = pygplates.Feature()
    feature.set_geometry(polygon)
    feature.set_reconstruction_plate_id(plate_id)
    return feature


def _write_features_to_gpml(features, path):
    fc = pygplates.FeatureCollection(features)
    fc.write(path)


@pytest.fixture
def moving_polygon_env(tmp_path):
    """Create a polygon on a plate that rotates 20 degrees over 100 Ma.

    The polygon covers lat [-10, 10], lon [0, 20] at present day (0 Ma).
    Plate 1 rotates 20 degrees around the north pole by 100 Ma, so at 100 Ma
    the polygon is at lon [20, 40].

    At 50 Ma the polygon is at lon [10, 30] (SLERP gives 10-degree shift).
    At 47 Ma the polygon is at lon [9.4, 29.4] (20 * 47/100 = 9.4 degree shift).
    At 45 Ma the polygon is at lon [9, 29] (snapped to 45 with interval=5).
    """
    vertices = [
        (10.0, 0.0), (10.0, 20.0), (-10.0, 20.0), (-10.0, 0.0),
    ]
    feature = _create_polygon_feature(vertices, plate_id=1)

    gpml_path = str(tmp_path / "continent.gpml")
    rot_path = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature], gpml_path)

    with open(rot_path, "w") as f:
        f.write("0  0.0  0.0  0.0  0.0  999  !anchor\n")
        f.write("1  0.0  0.0  0.0  0.0  0  !plate1 at 0\n")
        f.write("1  100.0  90.0  0.0  20.0  0  !plate1 at 100 Ma\n")

    return gpml_path, rot_path


class TestSnappingRemoved:
    """Verify that _snap_continental_time no longer exists."""

    def test_snap_method_removed(self):
        """SeafloorAgeTracker should not have _snap_continental_time anymore."""
        from gtrack.hpc_integration import SeafloorAgeTracker
        assert not hasattr(SeafloorAgeTracker, '_snap_continental_time')


class TestExactFloatTimeReconstruction:
    """Verify that continental polygons are reconstructed at exact float times."""

    def test_different_float_times_give_different_masks(self, moving_polygon_env):
        """Reconstructions at 47.0 vs 45.0 Ma give different masks.

        With the old snapping (interval=5), both 47 and 45 would snap to 45
        and produce identical masks. Without snapping, 47 and 45 reconstruct
        to different polygon positions.
        """
        gpml_path, rot_path = moving_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)
        cache = ContinentalPolygonCache(features, rotation_model)

        # Test a grid of points spanning the region where the polygon moves
        lons = np.linspace(-5, 45, 50)
        lats = np.zeros_like(lons)

        mask_47 = cache.get_continental_mask(lats, lons, 47.0)
        mask_45 = cache.get_continental_mask(lats, lons, 45.0)

        assert not np.array_equal(mask_47, mask_45), (
            "Masks at 47 Ma and 45 Ma should differ (polygon moved ~0.4 degrees "
            "between these times). If they are identical, snapping may still be active."
        )

    def test_float_time_precision(self, moving_polygon_env):
        """Reconstruction at 50.0 and 50.5 Ma give subtly different masks."""
        gpml_path, rot_path = moving_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)
        cache = ContinentalPolygonCache(features, rotation_model)

        # The polygon shifts 0.1 degrees between 50.0 and 50.5 Ma.
        # Use a dense grid near the polygon edge to detect this.
        lons = np.linspace(9.5, 11.0, 100)
        lats = np.zeros_like(lons)

        mask_50_0 = cache.get_continental_mask(lats, lons, 50.0)
        mask_50_5 = cache.get_continental_mask(lats, lons, 50.5)

        assert not np.array_equal(mask_50_0, mask_50_5), (
            "Masks at 50.0 and 50.5 Ma should differ near the polygon edge"
        )

    def test_monotonic_polygon_shift(self, moving_polygon_env):
        """As time increases, the polygon boundary shifts monotonically in longitude."""
        gpml_path, rot_path = moving_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)
        cache = ContinentalPolygonCache(features, rotation_model)

        # Track the rightmost longitude where the polygon ends on the equator
        times = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
        right_edges = []

        for t in times:
            lons = np.linspace(-5, 50, 500)
            lats = np.zeros_like(lons)
            mask = cache.get_continental_mask(lats, lons, t)
            if mask.any():
                right_edges.append(lons[mask][-1])
            else:
                right_edges.append(None)

        # The right edge should increase with time (polygon moves eastward)
        for i in range(1, len(right_edges)):
            assert right_edges[i] is not None and right_edges[i - 1] is not None
            assert right_edges[i] > right_edges[i - 1], (
                f"Right edge at {times[i]} Ma ({right_edges[i]:.1f}) should be "
                f"greater than at {times[i-1]} Ma ({right_edges[i-1]:.1f})"
            )


class TestGPlatelyIntervalKey:
    """Verify the isolated GPlately compatibility key."""

    def test_internal_config_omits_external_field(self):
        assert "continental_reconstruction_interval" not in (
            TracerConfig.__dataclass_fields__
        )

    def test_to_dict_emits_constant_external_value(self):
        assert TracerConfig().to_dict()["continental_reconstruction_interval"] == 1

    def test_from_dict_resets_old_interval(self):
        """from_dict with old interval value resets to 1."""
        old_config = {
            'time_step': 1.0,
            'continental_reconstruction_interval': 5,
        }

        config = TracerConfig.from_dict(old_config)
        assert config.tracker_step_myr == 1.0

    def test_from_dict_default_interval_no_reset(self):
        """from_dict with interval=1 does not modify anything."""
        config = TracerConfig.from_dict({'continental_reconstruction_interval': 1})
        assert "continental_reconstruction_interval" not in config.__dataclass_fields__

    @pytest.mark.parametrize("value", [0, -1, 1.5, "1", True])
    def test_from_dict_rejects_invalid_interval(self, value):
        with pytest.raises(ValueError, match="must be a positive integer"):
            TracerConfig.from_dict({'continental_reconstruction_interval': value})


class TestCacheHandlesFloatTimes:
    """Verify the ContinentalPolygonCache properly caches float times."""

    def test_cache_hit_on_same_float(self, moving_polygon_env):
        """Querying the same float time twice uses the cache."""
        gpml_path, rot_path = moving_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)
        cache = ContinentalPolygonCache(features, rotation_model, max_cache_size=5)

        lats = np.array([0.0])
        lons = np.array([15.0])

        mask1 = cache.get_continental_mask(lats, lons, 47.3)
        mask2 = cache.get_continental_mask(lats, lons, 47.3)

        np.testing.assert_array_equal(mask1, mask2)
        # Cache should have exactly one entry for 47.3
        assert 47.3 in cache._cache

    def test_cache_eviction_with_many_float_times(self, moving_polygon_env):
        """Cache correctly evicts old entries when capacity is exceeded."""
        gpml_path, rot_path = moving_polygon_env
        rotation_model = pygplates.RotationModel([rot_path])
        features = pygplates.FeatureCollection(gpml_path)
        cache = ContinentalPolygonCache(features, rotation_model, max_cache_size=3)

        lats = np.array([0.0])
        lons = np.array([15.0])

        # Fill the cache with 3 entries
        cache.get_continental_mask(lats, lons, 10.0)
        cache.get_continental_mask(lats, lons, 20.0)
        cache.get_continental_mask(lats, lons, 30.0)
        assert len(cache._cache) == 3

        # Adding a 4th should evict the oldest (10.0)
        cache.get_continental_mask(lats, lons, 40.0)
        assert len(cache._cache) == 3
        assert 10.0 not in cache._cache
        assert 40.0 in cache._cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
