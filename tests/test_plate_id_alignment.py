"""Tests for plate ID assignment under the topological rotation engine.

History: this file used to assert that assigning plate IDs from the "right"
polygon set kept RIGID rotation aligned with a containment polygon. Under the
single topological engine that premise is gone — motion no longer depends on
plate IDs at all, so the static-vs-continental id mismatch can no longer bend a
point's trajectory. The mismatch is now the *documented regression*: the engine
advects a point by the topology it sits in, whatever (label-only) plate id is
attached. These tests pin that new contract on the synthetic model fixture (see
tests/data/make_synthetic_model.py), plus the assign_plate_ids input validation.
"""

import numpy as np
import pygplates
import pytest

from gtrack.point_rotation import PointCloud, PointRotator
from gtrack.geometry import LatLon2XYZ


def _rotator(sm):
    return PointRotator(
        rotation_files=sm["rotation_files"],
        topology_files=sm["topology_files"],
        static_polygons=sm["static_polygons"],
    )


class TestPlateIdSourceModes:
    """source="static" vs source="topology" (the label mismatch, made explicit)."""

    def test_static_and_topology_ids_differ(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["out_of_circuit_point"]]))
        static = r.assign_plate_ids(cloud, at_age=0.0, source="static")
        topo = r.assign_plate_ids(cloud, at_age=0.0, source="topology")
        # Static polygon labels the point plate 999; topology labels it plate 1.
        assert static.plate_ids[0] == sm["out_of_circuit_plate"]
        assert topo.plate_ids[0] == 1

    def test_topology_is_the_default(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["out_of_circuit_point"]]))
        default = r.assign_plate_ids(cloud, at_age=0.0)
        assert default.plate_ids[0] == 1

    def test_use_static_polygons_backcompat_alias(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["out_of_circuit_point"]]))
        result = r.assign_plate_ids(cloud, at_age=0.0, use_static_polygons=True)
        assert result.plate_ids[0] == sm["out_of_circuit_plate"]


class TestMotionIndependentOfLabel:
    """The core fix: trajectory is set by topology, not by the attached id."""

    def test_rotation_ignores_attached_plate_ids(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        pts = np.array([sm["plate_a_point"], sm["out_of_circuit_point"]])

        # Same points, two very different (label-only) plate-id assignments.
        cloud_static = r.assign_plate_ids(
            PointCloud.from_latlon(pts), at_age=0.0, source="static")
        cloud_topo = r.assign_plate_ids(
            PointCloud.from_latlon(pts), at_age=0.0, source="topology")
        assert not np.array_equal(cloud_static.plate_ids, cloud_topo.plate_ids)

        moved_static = r.rotate(cloud_static, from_age=0.0, to_age=100.0)
        moved_topo = r.rotate(cloud_topo, from_age=0.0, to_age=100.0)

        # Identical trajectories regardless of the attached label.
        np.testing.assert_allclose(moved_static.xyz, moved_topo.xyz, atol=1e-6)
        # ... and both keep all points (no out-of-circuit drop).
        assert moved_static.n_points == moved_topo.n_points == 2


class TestPartitioningFeaturesOverride:
    def test_partitioning_features_overrides_source(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["out_of_circuit_point"]]))
        # An explicit empty collection => everything partitions to 0, overriding
        # both source="topology" and the static polygons.
        empty = pygplates.FeatureCollection()
        with pytest.warns(UserWarning, match="undefined plate IDs"):
            result = r.assign_plate_ids(
                cloud, at_age=0.0, partitioning_features=empty, remove_undefined=False)
        assert result.plate_ids[0] == 0


class TestAssignPlateIdsValidation:
    def test_wrong_type_raises_type_error(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["plate_a_point"]]))
        with pytest.raises(TypeError, match="pygplates.FeatureCollection"):
            r.assign_plate_ids(
                cloud, at_age=0.0, partitioning_features=sm["static_polygons"])

    def test_static_source_without_static_polygons_raises(self, synthetic_model):
        sm = synthetic_model
        r = PointRotator(
            rotation_files=sm["rotation_files"],
            topology_files=sm["topology_files"],
        )  # no static polygons
        cloud = PointCloud.from_latlon(np.array([sm["plate_a_point"]]))
        with pytest.raises(ValueError, match="requires static_polygons"):
            r.assign_plate_ids(cloud, at_age=0.0, source="static")

    def test_invalid_source_raises(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = PointCloud.from_latlon(np.array([sm["plate_a_point"]]))
        with pytest.raises(ValueError, match="topology.*static"):
            r.assign_plate_ids(cloud, at_age=0.0, source="bogus")


class TestNonzeroAge:
    def test_assign_at_nonzero_age(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        # Plate A point, reconstructed forward, still lands on plate A at 50 Ma.
        cloud = PointCloud.from_latlon(np.array([sm["plate_a_point"]]))
        moved = r.rotate(cloud, from_age=0.0, to_age=50.0)
        ids = r.assign_plate_ids(moved, at_age=50.0, source="topology").plate_ids
        assert ids[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
