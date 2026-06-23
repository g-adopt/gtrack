"""Tests for plate ID assignment alignment with containment polygons.

These tests verify that plate IDs can be assigned from the same polygon set
used for containment filtering, addressing the polygon file mismatch
described in ISSUES.md section 1.2.
"""

import numpy as np
import pytest
import pygplates

from gtrack.point_rotation import PointCloud, PointRotator
from gtrack.polygon_filter import PolygonFilter
from gtrack.geometry import LatLon2XYZ


def _create_polygon_feature(vertices_latlon, plate_id):
    """Create a pygplates polygon feature from (lat, lon) vertices."""
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in vertices_latlon]
    polygon = pygplates.PolygonOnSphere(points)
    feature = pygplates.Feature()
    feature.set_geometry(polygon)
    feature.set_reconstruction_plate_id(plate_id)
    return feature


def _write_features_to_gpml(features, path):
    """Write features to a .gpml file."""
    fc = pygplates.FeatureCollection(features)
    fc.write(path)


@pytest.fixture
def mismatched_polygon_env(tmp_path):
    """Create an environment where continental and static polygons assign different plate IDs.

    Continental polygon: lat [-10, 10], lon [-10, 10], plate_id=1
    Static polygon A: lat [-20, 0], lon [-20, 20], plate_id=2  (covers southern half + extra)
    Static polygon B: lat [0, 20], lon [-20, 20], plate_id=3   (covers northern half + extra)

    A point at (5, 5) is inside the continental polygon (plate 1) but the static
    polygons assign it to plate 3.
    A point at (-5, -5) is inside the continental polygon (plate 1) but the static
    polygons assign it to plate 2.

    Rotation model:
    - Plate 1: no motion (identity)
    - Plate 2: rotates 15 degrees around north pole by 100 Ma
    - Plate 3: rotates -15 degrees around north pole by 100 Ma
    """
    # Continental polygon
    continental_feature = _create_polygon_feature(
        [(10.0, -10.0), (10.0, 10.0), (-10.0, 10.0), (-10.0, -10.0)],
        plate_id=1,
    )
    continental_gpml = str(tmp_path / "continental.gpml")
    _write_features_to_gpml([continental_feature], continental_gpml)

    # Static polygons — split the globe into two hemispheres with different plate IDs
    static_a = _create_polygon_feature(
        [(0.0, -20.0), (0.0, 20.0), (-20.0, 20.0), (-20.0, -20.0)],
        plate_id=2,
    )
    static_b = _create_polygon_feature(
        [(20.0, -20.0), (20.0, 20.0), (0.0, 20.0), (0.0, -20.0)],
        plate_id=3,
    )
    static_gpml = str(tmp_path / "static.gpml")
    _write_features_to_gpml([static_a, static_b], static_gpml)

    # Rotation file
    rot_path = str(tmp_path / "rotations.rot")
    with open(rot_path, "w") as f:
        # Anchor plate
        f.write("0  0.0  0.0  0.0  0.0  999  !anchor\n")
        # Plate 1 — no motion
        f.write("1  0.0  0.0  0.0  0.0  0  !plate1 at 0\n")
        f.write("1  100.0  0.0  0.0  0.0  0  !plate1 at 100 Ma\n")
        # Plate 2 — rotates 15 deg around north pole
        f.write("2  0.0  0.0  0.0  0.0  0  !plate2 at 0\n")
        f.write("2  100.0  90.0  0.0  15.0  0  !plate2 at 100 Ma\n")
        # Plate 3 — rotates -15 deg around north pole
        f.write("3  0.0  0.0  0.0  0.0  0  !plate3 at 0\n")
        f.write("3  100.0  90.0  0.0  -15.0  0  !plate3 at 100 Ma\n")

    return continental_gpml, static_gpml, rot_path


class TestPlateIdAssignmentSource:
    """Test that plate IDs can be assigned from different polygon sources."""

    def test_static_vs_continental_plate_ids_differ(self, mismatched_polygon_env):
        """Points get different plate IDs from static polygons vs continental polygons."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        # Point at (5, 5) — inside continental polygon (plate 1), inside static B (plate 3)
        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        continental_features = pygplates.FeatureCollection(continental_gpml)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        # Default: uses static polygons
        cloud_static = rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=False
        )
        # Explicit: uses continental polygons
        cloud_continental = rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=False,
            partitioning_features=continental_features,
        )

        assert cloud_static.plate_ids[0] == 3, "Static polygon should assign plate 3"
        assert cloud_continental.plate_ids[0] == 1, "Continental polygon should assign plate 1"

    def test_continental_plate_ids_preserve_boundary_alignment(self, mismatched_polygon_env):
        """Points rotated with continental plate IDs stay closer to the continental boundary.

        When plate IDs come from the continental polygon, the point and the polygon
        move with the same rotation (plate 1 = identity), so they stay aligned.
        When plate IDs come from the static polygon (plate 2 or 3), the point moves
        with a different rotation than the continental polygon, causing drift.
        """
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        # Points inside the continental polygon
        lats = np.array([5.0, -5.0, 3.0, -3.0])
        lons = np.array([5.0, -5.0, 8.0, -8.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        continental_features = pygplates.FeatureCollection(continental_gpml)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        # Assign from continental polygons and rotate
        cloud_cont = rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=False,
            partitioning_features=continental_features,
        )
        rotated_cont = rotator.rotate(cloud_cont, from_age=0.0, to_age=100.0)

        # Assign from static polygons and rotate
        cloud_static = rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=False,
        )
        rotated_static = rotator.rotate(cloud_static, from_age=0.0, to_age=100.0)

        # Continental polygon has plate_id=1 which has identity rotation,
        # so continental-assigned points should not move
        np.testing.assert_allclose(
            rotated_cont.xyz, cloud.xyz, rtol=1e-6,
            err_msg="Continental plate ID points should stay put (plate 1 = identity)"
        )

        # Static-assigned points should have moved (plates 2 and 3 rotate)
        assert not np.allclose(rotated_static.xyz, cloud.xyz, rtol=1e-3), (
            "Static plate ID points should have moved (plates 2/3 rotate)"
        )

    def test_partitioning_features_overrides_static(self, mismatched_polygon_env):
        """The partitioning_features parameter takes precedence over use_static_polygons."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[-5.0, -5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        continental_features = pygplates.FeatureCollection(continental_gpml)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        # Even with use_static_polygons=True, partitioning_features overrides
        result = rotator.assign_plate_ids(
            cloud, at_age=0.0, remove_undefined=False,
            use_static_polygons=True,
            partitioning_features=continental_features,
        )
        assert result.plate_ids[0] == 1, "partitioning_features should override static"

    def test_backward_compatible_default(self, mismatched_polygon_env):
        """Without partitioning_features, behavior is unchanged (uses static polygons)."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        result = rotator.assign_plate_ids(cloud, at_age=0.0, remove_undefined=False)
        assert result.plate_ids[0] == 3, "Default should still use static polygons"

    def test_full_pipeline_continental_alignment(self, mismatched_polygon_env):
        """End-to-end test: filter -> assign from continental -> rotate stays aligned."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        # Create a grid of points, some inside and some outside the continental polygon
        lats = np.array([0.0, 5.0, -5.0, 8.0, -8.0, 20.0, -20.0])
        lons = np.array([0.0, 5.0, -5.0, 8.0, -8.0, 0.0, 0.0])
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)
        cloud.add_property("thickness", np.arange(7, dtype=float))

        continental_features = pygplates.FeatureCollection(continental_gpml)

        # Step 1: filter to continental points
        pf = PolygonFilter(
            polygon_files=continental_gpml, rotation_files=rot_path,
        )
        cont_cloud = pf.filter_inside(cloud, at_age=0.0)

        # The last two points (lat=20, lat=-20) should be outside
        assert cont_cloud.n_points == 5

        # Step 2: assign plate IDs from continental polygons
        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )
        cont_cloud = rotator.assign_plate_ids(
            cont_cloud, at_age=0.0, remove_undefined=False,
            partitioning_features=continental_features,
        )

        # All should get plate 1 from the continental polygon
        assert np.all(cont_cloud.plate_ids == 1)

        # Step 3: rotate — plate 1 has identity rotation, so positions shouldn't change
        rotated = rotator.rotate(cont_cloud, from_age=0.0, to_age=100.0)
        np.testing.assert_allclose(rotated.xyz, cont_cloud.xyz, rtol=1e-6)

        # Properties should be preserved
        assert rotated.get_property("thickness") is not None


class TestPlateIdEdgeCases:
    """Test edge cases for the partitioning_features parameter."""

    def test_wrong_type_raises_type_error(self, mismatched_polygon_env):
        """Passing a string path instead of FeatureCollection raises TypeError."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        with pytest.raises(TypeError, match="pygplates.FeatureCollection"):
            rotator.assign_plate_ids(
                cloud, at_age=0.0,
                partitioning_features=continental_gpml,  # string, not FeatureCollection
            )

    def test_empty_features_assigns_plate_zero(self, mismatched_polygon_env):
        """Empty FeatureCollection assigns plate_id=0 (undefined) to all points."""
        _, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        empty_features = pygplates.FeatureCollection()
        with pytest.warns(UserWarning, match="undefined plate IDs"):
            result = rotator.assign_plate_ids(
                cloud, at_age=0.0, remove_undefined=False,
                partitioning_features=empty_features,
            )
        assert result.plate_ids[0] == 0

    def test_empty_features_with_remove_undefined(self, mismatched_polygon_env):
        """Empty FeatureCollection + remove_undefined=True removes all points."""
        _, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        empty_features = pygplates.FeatureCollection()
        with pytest.warns(UserWarning, match="undefined plate IDs"):
            result = rotator.assign_plate_ids(
                cloud, at_age=0.0, remove_undefined=True,
                partitioning_features=empty_features,
            )
        assert result.n_points == 0


class TestPlateIdNonzeroAge:
    """Test plate ID assignment at non-zero reconstruction times."""

    def test_assign_at_nonzero_age(self, mismatched_polygon_env):
        """Plate ID assignment works at non-zero reconstruction time."""
        continental_gpml, static_gpml, rot_path = mismatched_polygon_env

        latlon = np.array([[5.0, 5.0]])
        xyz = LatLon2XYZ(latlon)
        cloud = PointCloud(xyz=xyz)

        continental_features = pygplates.FeatureCollection(continental_gpml)
        rotator = PointRotator(
            rotation_files=rot_path,
            static_polygons=static_gpml,
        )

        # At 50 Ma with continental polygons (plate 1 = identity, polygon stays put)
        result = rotator.assign_plate_ids(
            cloud, at_age=50.0, remove_undefined=False,
            partitioning_features=continental_features,
        )
        assert result.plate_ids[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
