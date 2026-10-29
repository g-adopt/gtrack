"""Test the opt-in rigid rotation by plate ID of ``PolygonIndicatorSource``.

The unit tests use the tiny synthetic plate model of tests/data (plate 1 rotates
30 deg about the north pole by 100 Ma, plate 2 rotates -20 deg, plate 999 has no
rotation sequence). The oracle for every rotation is ``pygplates.reconstruct`` on
a feature that carries the same plate ID, which is the rule that the continent
polygons follow. One slow test repeats the check on the real Zahirovic 2022 data
when it is present.
"""

import logging
from pathlib import Path

import numpy as np
import pygplates
import pytest

from gtrack import PointCloud
from gtrack.geometry import LatLon2XYZ
from gtrack.point_rotation import _get_plate_ids
from gtrack.sources import (
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
    _rotate_rigid_by_plate_id,
    build_indicator_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_points(rotation_file, latlon, plate_ids, age):
    """Rotate points with ``pygplates.reconstruct``, one feature per point.

    Returns an array of (lat, lon) in input order. Every point must be valid at
    ``age``.
    """
    rotation_model = pygplates.RotationModel(rotation_file)
    features = []
    for i, ((lat, lon), plate_id) in enumerate(zip(latlon, plate_ids)):
        feature = pygplates.Feature()
        feature.set_geometry(pygplates.PointOnSphere(float(lat), float(lon)))
        feature.set_reconstruction_plate_id(int(plate_id))
        feature.set_valid_time(600.0, -999.0)
        feature.set_name(str(i))
        features.append(feature)
    reconstructed = []
    pygplates.reconstruct(features, rotation_model, reconstructed, float(age))
    by_index = {
        int(r.get_feature().get_name()): r.get_reconstructed_geometry().to_lat_lon()
        for r in reconstructed
    }
    return np.array([by_index[i] for i in range(len(latlon))])


# The comparisons go through a Cartesian round trip (lat/lon -> xyz -> lat/lon),
# which loses about 1e-6 deg near the poles. 1e-5 deg is about one metre.
TOLERANCE_DEG = 1e-5


def angular_distance_deg(latlon_a, latlon_b):
    """Great-circle distance in degrees between matching (lat, lon) rows."""
    a, b = LatLon2XYZ(latlon_a), LatLon2XYZ(latlon_b)
    # LatLon2XYZ returns Cartesian coordinates scaled by the Earth radius.
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


class LogCapture(logging.Handler):
    """Collect gtrack.sources records however the package configured propagation."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self._logger = logging.getLogger("gtrack.sources")
        self._old_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._old_level)

    def messages(self, level):
        return [r.getMessage() for r in self.records if r.levelno == level]


def write_polygon(path, corners, plate_id, begin_age=600.0):
    """Write one polygon feature with the given plate ID and begin age to a GPML file."""
    feature = pygplates.Feature()
    feature.set_geometry(pygplates.PolygonOnSphere(corners))
    feature.set_reconstruction_plate_id(plate_id)
    feature.set_valid_time(begin_age, -999.0)
    feature.set_name("test polygon")
    pygplates.FeatureCollection([feature]).write(str(path))
    return str(path)


@pytest.fixture
def rotation_model(synthetic_model):
    return pygplates.RotationModel(synthetic_model["rotation_files"])


# Points on plates 1 and 2 (in the rotation circuit) and plate 999 (not in it).
LATLON = np.array([[10.0, 10.0], [-20.0, 45.0], [5.0, -100.0], [0.0, 0.0]])
PLATES = np.array([1, 2, 1, 999])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_is_topological(self):
        assert PolygonIndicatorConfig().rotation == "topological"

    def test_accepts_rigid_plate_id(self):
        assert PolygonIndicatorConfig(rotation="rigid_plate_id").rotation == "rigid_plate_id"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="rotation must be one of"):
            PolygonIndicatorConfig(rotation="rigid")

    def test_missing_plate_default_is_in_place(self):
        assert PolygonIndicatorConfig().rigid_missing_plate == "in_place"

    def test_rejects_unknown_missing_plate_policy(self):
        with pytest.raises(ValueError, match="rigid_missing_plate must be one of"):
            PolygonIndicatorConfig(rotation="rigid_plate_id", rigid_missing_plate="drop")

    def test_missing_plate_policy_needs_the_rigid_mode(self):
        with pytest.raises(ValueError, match="only applies"):
            PolygonIndicatorConfig(rigid_missing_plate="topological_plate")

    def test_accepts_the_fallback_policy_with_the_rigid_mode(self):
        config = PolygonIndicatorConfig(
            rotation="rigid_plate_id", rigid_missing_plate="topological_plate"
        )
        assert config.rigid_missing_plate == "topological_plate"


# ---------------------------------------------------------------------------
# The rigid rotation helper
# ---------------------------------------------------------------------------

class TestRotateRigidByPlateId:
    def make_cloud(self):
        cloud = PointCloud.from_latlon(LATLON)
        cloud.add_property("tag", np.arange(len(LATLON), dtype=float))
        return cloud

    @pytest.mark.parametrize("age", [0.0, 40.0, 100.0, 250.0])
    def test_matches_pygplates_reconstruct(self, synthetic_model, rotation_model, age):
        """Every point, including the one whose plate has no sequence, agrees."""
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, age
        )
        expected = reconstruct_points(
            synthetic_model["rotation_files"], LATLON, PLATES, age
        )
        assert out.n_points == 4
        assert angular_distance_deg(out.latlon, expected).max() < TOLERANCE_DEG

    def test_plate_missing_from_the_tree_stays_in_place(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0
        )
        assert angular_distance_deg(out.latlon[3:], LATLON[3:]).max() < TOLERANCE_DEG

    def test_plate_with_a_sequence_moves(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0
        )
        # Plate 1 rotates 30 deg about the pole, so lon changes by 30 deg at the same lat.
        assert angular_distance_deg(out.latlon[:1], LATLON[:1]).max() > 1.0
        assert np.isclose(out.latlon[0, 0], LATLON[0, 0])

    def test_present_day_returns_the_same_points(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 0.0
        )
        assert out.n_points == 4
        assert angular_distance_deg(out.latlon, LATLON).max() < TOLERANCE_DEG

    def test_polygon_older_than_its_begin_age_is_removed(self, rotation_model):
        begin = np.array([50.0, np.inf, 100.0, 20.0])
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, begin, 0.0, 100.0
        )
        # Points 0 and 3 begin after 100 Ma, 2 begins at exactly 100 Ma (kept).
        assert out.n_points == 2
        assert list(out.get_property("tag")) == [1.0, 2.0]

    def test_begin_age_is_inclusive(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, 40.0), 0.0, 40.0
        )
        assert out.n_points == 4

    def test_begin_age_tolerates_float_round_off(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, 40.0), 0.0,
            40.0 + 1e-12,
        )
        assert out.n_points == 4

    def test_begin_age_beyond_the_tolerance_removes_the_points(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, 40.0), 0.0,
            40.0 + 1e-6,
        )
        assert out.n_points == 0

    def test_stats_and_log_line_count_the_points(self, rotation_model):
        stats = {}
        begin = np.array([np.inf, np.inf, np.inf, 50.0])
        with LogCapture() as logs:
            _rotate_rigid_by_plate_id(
                self.make_cloud(), rotation_model, PLATES, begin, 0.0, 100.0,
                stats=stats,
            )
        # Plate 999 has no sequence, but its polygon (begin 50) is not valid at
        # 100 Ma, so that point is removed and not counted as in place.
        assert stats == {"n_kept": 3, "n_in_place": 0, "n_fallback": 0, "n_removed": 1}
        (line,) = logs.messages(logging.INFO)
        assert "3 source points kept" in line and "1 removed" in line

    def test_stats_describe_the_latest_call_only(self, rotation_model):
        stats = {}
        _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0,
            stats=stats,
        )
        assert stats == {"n_kept": 4, "n_in_place": 1, "n_fallback": 0, "n_removed": 0}
        _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES,
            np.array([np.inf, np.inf, np.inf, 50.0]), 0.0, 100.0, stats=stats,
        )
        assert stats == {"n_kept": 3, "n_in_place": 0, "n_fallback": 0, "n_removed": 1}

    def test_points_of_a_missing_plate_are_counted_as_in_place(self, rotation_model):
        stats = {}
        _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0,
            stats=stats,
        )
        assert stats["n_in_place"] == 1 and stats["n_kept"] == 4

    def test_plate_that_appears_after_the_present_day_moves(self, tmp_path):
        """A plate with no sequence at 0 Ma must still follow its chain at 100 Ma.

        The three-argument stage form ``get_rotation(to, plate, from=0)`` returns
        the identity when the plate is missing from the tree at ``from``. The
        two-argument form reads the tree at ``to`` only, like pygplates.reconstruct.
        """
        rot = tmp_path / "late_plate.rot"
        rot.write_text(
            "0    0.0  0.0 0.0  0.0 000 !anchor\n"
            "0  600.0  0.0 0.0  0.0 000 !anchor\n"
            # An off-axis pole, so that latitudes change as well as longitudes.
            "3   50.0 40.0 30.0 25.0 000 !late plate, first pole at 50 Ma\n"
            "3  600.0 40.0 30.0 25.0 000 !late plate\n"
        )
        model = pygplates.RotationModel(str(rot))
        latlon = np.array([[10.0, 10.0], [-30.0, 120.0]])
        cloud = PointCloud.from_latlon(latlon)
        plates = np.array([3, 3])

        out = _rotate_rigid_by_plate_id(
            cloud, model, plates, np.full(2, np.inf), 0.0, 100.0
        )
        expected = reconstruct_points(str(rot), latlon, plates, 100.0)
        assert angular_distance_deg(out.latlon, expected).max() < TOLERANCE_DEG
        # The plate rotates about an off-axis pole, so the points really move and
        # their latitudes change too.
        assert angular_distance_deg(out.latlon, latlon).min() > 1.0
        assert np.abs(out.latlon[:, 0] - latlon[:, 0]).max() > 0.5
        # The stage form would have left them in place.
        stage = model.get_rotation(100.0, 3, 0.0)
        assert angular_distance_deg(
            np.array([[0.0, 0.0]]),
            np.array([list((stage * pygplates.PointOnSphere(0.0, 0.0)).to_lat_lon())]),
        ).max() < TOLERANCE_DEG

    def test_wrong_fallback_length_raises(self, rotation_model):
        with pytest.raises(ValueError, match="seed_fallback_plate_ids must have"):
            _rotate_rigid_by_plate_id(
                self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0,
                10.0, fallback_plate_ids=PLATES[:2],
            )

    def test_wrong_array_length_raises(self, rotation_model):
        with pytest.raises(ValueError, match="one entry per source point"):
            _rotate_rigid_by_plate_id(
                self.make_cloud(), rotation_model, PLATES[:2], np.full(4, np.inf), 0.0, 10.0
            )

    def test_nonzero_from_age_raises(self, rotation_model):
        with pytest.raises(ValueError, match="from_age must be 0"):
            _rotate_rigid_by_plate_id(
                self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 20.0, 40.0
            )

    def test_round_off_from_age_is_accepted(self, rotation_model):
        out = _rotate_rigid_by_plate_id(
            self.make_cloud(), rotation_model, PLATES, np.full(4, np.inf), 1e-14, 40.0
        )
        assert out.n_points == 4


class TestFallbackPlate:
    """A point whose plate is missing at the target age follows its fallback plate."""

    def cloud(self):
        return PointCloud.from_latlon(LATLON)

    def test_missing_plate_follows_the_fallback_plate(self, synthetic_model, rotation_model):
        # Plate 999 has no sequence. Its point (0, 0) follows plate 1 instead.
        fallback = np.array([1, 2, 1, 1])
        stats = {}
        out = _rotate_rigid_by_plate_id(
            self.cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0,
            fallback_plate_ids=fallback, stats=stats,
        )
        expected = reconstruct_points(
            synthetic_model["rotation_files"], LATLON, np.array([1, 2, 1, 1]), 100.0
        )
        assert angular_distance_deg(out.latlon, expected).max() < TOLERANCE_DEG
        assert stats["n_fallback"] == 1 and stats["n_in_place"] == 0

    def test_points_with_a_plate_ignore_the_fallback(self, synthetic_model, rotation_model):
        fallback = np.array([2, 1, 2, 2])
        out = _rotate_rigid_by_plate_id(
            self.cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0,
            fallback_plate_ids=fallback,
        )
        expected = reconstruct_points(
            synthetic_model["rotation_files"], LATLON[:3], PLATES[:3], 100.0
        )
        assert angular_distance_deg(out.latlon[:3], expected).max() < TOLERANCE_DEG

    def test_missing_fallback_plate_leaves_the_point_in_place(self, rotation_model):
        stats = {}
        out = _rotate_rigid_by_plate_id(
            self.cloud(), rotation_model, PLATES, np.full(4, np.inf), 0.0, 100.0,
            fallback_plate_ids=np.array([1, 2, 1, 999]), stats=stats,
        )
        assert angular_distance_deg(out.latlon[3:], LATLON[3:]).max() < TOLERANCE_DEG
        assert stats["n_in_place"] == 1 and stats["n_fallback"] == 0

    def test_plate_id_zero_counts_as_missing(self, synthetic_model, rotation_model):
        plates = np.array([1, 2, 1, 0])
        out = _rotate_rigid_by_plate_id(
            self.cloud(), rotation_model, plates, np.full(4, np.inf), 0.0, 100.0,
            fallback_plate_ids=np.array([1, 2, 1, 1]),
        )
        expected = reconstruct_points(
            synthetic_model["rotation_files"], LATLON, np.array([1, 2, 1, 1]), 100.0
        )
        assert angular_distance_deg(out.latlon, expected).max() < TOLERANCE_DEG

    def test_the_choice_is_made_again_at_every_age(self, tmp_path):
        """A late plate follows its own sequence after 50 Ma and the fallback before."""
        rot = tmp_path / "late_and_base.rot"
        rot.write_text(
            "0    0.0  0.0 0.0  0.0 000 !anchor\n"
            "0  600.0  0.0 0.0  0.0 000 !anchor\n"
            "1    0.0  0.0 0.0  0.0 000 !base plate\n"
            "1  600.0 60.0 20.0 40.0 000 !base plate\n"
            "3   50.0 40.0 30.0 25.0 000 !late plate, first pole at 50 Ma\n"
            "3  600.0 40.0 30.0 25.0 000 !late plate\n"
        )
        model = pygplates.RotationModel(str(rot))
        latlon = np.array([[10.0, 10.0], [-30.0, 120.0]])
        cloud = PointCloud.from_latlon(latlon)
        own, fallback = np.array([3, 3]), np.array([1, 1])
        for age, plate in ((30.0, 1), (100.0, 3)):
            out = _rotate_rigid_by_plate_id(
                cloud, model, own, np.full(2, np.inf), 0.0, age, fallback_plate_ids=fallback
            )
            expected = reconstruct_points(str(rot), latlon, np.array([plate, plate]), age)
            assert angular_distance_deg(out.latlon, expected).max() < TOLERANCE_DEG


# ---------------------------------------------------------------------------
# build_indicator_source
# ---------------------------------------------------------------------------

class StubRotator:
    """Provides only what the rigid branch needs, and fails if rotate is used."""

    def __init__(self, rotation_model):
        self.rotation_model = rotation_model

    def rotate(self, *args, **kwargs):
        raise AssertionError("the rigid branch must not call rotator.rotate")


class TestBuildIndicatorSourceRigid:
    def seeds(self):
        cloud = PointCloud.from_latlon(LATLON)
        cloud.add_property("masked_thickness", np.full(4, 50.0))
        return cloud

    def test_rigid_branch_rotates_and_adds_background(self, rotation_model):
        cloud = build_indicator_source(
            self.seeds(), StubRotator(rotation_model), 100.0,
            background_point_count=500,
            rotation="rigid_plate_id",
            seed_plate_ids=PLATES,
            seed_begin_ages=np.full(4, np.inf),
        )
        membership = cloud.get_property("membership")
        assert membership[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert membership[4:].max() == 0.0

        # Every background point lies outside the collision radius of every seed.
        from scipy.spatial import cKDTree

        unit = cloud.xyz / np.linalg.norm(cloud.xyz, axis=1, keepdims=True)
        radius = np.sqrt(4.0 * np.pi / 500)            # exclusion_factor 1.0
        chord = 2.0 * np.sin(radius / 2.0)
        distance, _ = cKDTree(unit[:4]).query(unit[4:], k=1)
        assert distance.min() > chord

    def test_removed_seeds_do_not_exclude_background(self, rotation_model):
        kept = build_indicator_source(
            self.seeds(), StubRotator(rotation_model), 100.0,
            background_point_count=500, rotation="rigid_plate_id",
            seed_plate_ids=PLATES, seed_begin_ages=np.full(4, np.inf),
        )
        removed = build_indicator_source(
            self.seeds(), StubRotator(rotation_model), 100.0,
            background_point_count=500, rotation="rigid_plate_id",
            seed_plate_ids=PLATES, seed_begin_ages=np.full(4, 0.0),
        )
        # With every seed removed the background is whole.
        assert removed.n_points == 500
        assert kept.n_points < 500 + 4

    def test_requires_the_plate_arrays(self, rotation_model):
        with pytest.raises(ValueError, match="requires seed_plate_ids"):
            build_indicator_source(
                self.seeds(), StubRotator(rotation_model), 100.0,
                rotation="rigid_plate_id",
            )

    def test_unknown_mode_raises(self, rotation_model):
        with pytest.raises(ValueError, match="rotation must be one of"):
            build_indicator_source(
                self.seeds(), StubRotator(rotation_model), 100.0, rotation="bogus",
            )


# ---------------------------------------------------------------------------
# PolygonIndicatorSource, with the synthetic static polygons as the continents
# ---------------------------------------------------------------------------

def make_source(synthetic_model, rotation, polygons=None, **config):
    return PolygonIndicatorSource(
        rotation_files=[synthetic_model["rotation_files"]],
        topology_files=[synthetic_model["topology_files"]],
        polygons=polygons or synthetic_model["static_polygons"],
        static_polygons=synthetic_model["static_polygons"],
        thickness_data=50.0,
        config=PolygonIndicatorConfig(
            background_point_count=2000, scalar_input_point_count=4000,
            rotation=rotation, **config,
        ),
    )


class TestSource:
    def test_default_mode_keeps_no_plate_arrays(self, synthetic_model):
        source = make_source(synthetic_model, "topological")
        source.at_age(50.0)
        assert source._seed_plate_ids is None
        assert source._seed_begin_ages is None

    def test_channels_are_unchanged(self, synthetic_model):
        cloud = make_source(synthetic_model, "rigid_plate_id").at_age(100.0)
        assert set(cloud.properties) == set(PolygonIndicatorSource.provides)

    def test_seeds_follow_their_polygon_plate_id(self, synthetic_model, rotation_model):
        source = make_source(synthetic_model, "rigid_plate_id")
        at_zero = source.at_age(0.0)
        at_age = source.at_age(100.0)
        n_seed = int((at_zero.get_property("membership") > 0.5).sum())
        assert n_seed > 0
        assert int((at_age.get_property("membership") > 0.5).sum()) == n_seed

        plate_ids = source._seed_plate_ids
        # Plate IDs from an independent path (partition_into_plates on the same
        # polygons), so a wrong assignment cannot pass by feeding the oracle.
        independent = _get_plate_ids(
            LatLon2XYZ(at_zero.latlon[:n_seed]),
            pygplates.FeatureCollection(synthetic_model["static_polygons"]),
            rotation_model, 0.0,
        )
        assert np.array_equal(plate_ids, independent)
        assert {1, 999} <= set(np.unique(plate_ids))
        # Reconstruct each seed with its own plate ID as the oracle.
        expected = reconstruct_points(
            synthetic_model["rotation_files"], at_zero.latlon[:n_seed], plate_ids, 100.0
        )
        assert angular_distance_deg(at_age.latlon[:n_seed], expected).max() < TOLERANCE_DEG

    def test_seeds_come_before_the_background(self, synthetic_model):
        source = make_source(synthetic_model, "rigid_plate_id")
        cloud = source.at_age(30.0)
        membership = cloud.get_property("membership")
        n_seed = len(source._seed_plate_ids)
        assert (membership[:n_seed] == 1.0).all()
        assert (membership[n_seed:] == 0.0).all()

    def test_the_begin_age_comes_from_the_polygon(self, synthetic_model, tmp_path):
        """A polygon that begins at 80 Ma holds its seeds up to 80 Ma, inclusive."""
        path = write_polygon(
            tmp_path / "begins_at_80.gpml",
            [(-20, -20), (-20, 20), (20, 20), (20, -20)], plate_id=1, begin_age=80.0,
        )
        source = make_source(synthetic_model, "rigid_plate_id", polygons=path)
        n_at_zero = int((source.at_age(0.0).get_property("membership") > 0.5).sum())
        assert n_at_zero > 0
        assert set(source._seed_begin_ages.tolist()) == {80.0}

        def n_seeds(age):
            return int((source.at_age(age).get_property("membership") > 0.5).sum())

        assert n_seeds(79.9) == n_at_zero
        assert n_seeds(80.0) == n_at_zero
        assert n_seeds(80.0 + 1e-12) == n_at_zero
        assert n_seeds(80.1) == 0

    def test_a_seed_in_no_polygon_raises(self, synthetic_model, monkeypatch):
        class NoPolygon:
            def __init__(self, *args, **kwargs):
                pass

            def partition_point(self, point):
                return None

        monkeypatch.setattr(pygplates, "PlatePartitioner", NoPolygon)
        source = make_source(synthetic_model, "rigid_plate_id")
        with pytest.raises(ValueError, match="lie in no polygon"):
            source.at_age(10.0)

    def test_points_that_stay_in_place_are_warned_about_once(self, synthetic_model):
        source = make_source(synthetic_model, "rigid_plate_id")
        with LogCapture() as logs:
            source.at_age(100.0)
            source.at_age(50.0)
            source.at_age(100.0)
        warnings = logs.messages(logging.WARNING)
        assert len(warnings) == 1
        assert "stay in place" in warnings[0] and "topological_plate" in warnings[0]
        # The per-call counts go to INFO, one line per call.
        assert len(logs.messages(logging.INFO)) == 3
        assert source._rigid_stats["n_in_place"] > 0

    def test_source_stats_follow_the_latest_age(self, synthetic_model, tmp_path):
        # Plate 999 has no sequence, so its seeds stay in place until the polygon begins.
        path = write_polygon(
            tmp_path / "plate_999.gpml",
            [(-20, -20), (-20, 20), (20, 20), (20, -20)], plate_id=999, begin_age=80.0,
        )
        source = make_source(synthetic_model, "rigid_plate_id", polygons=path)
        source.at_age(50.0)
        n_seed = len(source._seed_plate_ids)
        assert source._rigid_stats == {
            "n_kept": n_seed, "n_in_place": n_seed, "n_fallback": 0, "n_removed": 0}
        source.at_age(100.0)
        assert source._rigid_stats == {
            "n_kept": 0, "n_in_place": 0, "n_fallback": 0, "n_removed": n_seed}

    def test_the_warning_does_not_advise_the_policy_that_is_active(self, synthetic_model):
        source = make_source(
            synthetic_model, "rigid_plate_id", rigid_missing_plate="topological_plate"
        )
        source._rigid_stats = {"n_kept": 10, "n_in_place": 3}
        with LogCapture() as logs:
            source._warn_once_if_points_stay_in_place(100.0)
        (message,) = logs.messages(logging.WARNING)
        assert "fallback plate" in message
        assert "Use rigid_missing_plate" not in message

    def test_polygons_with_plate_id_zero_raise(self, synthetic_model, tmp_path):
        feature = pygplates.Feature()
        feature.set_geometry(pygplates.PolygonOnSphere(
            [(-20, -20), (-20, 20), (20, 20), (20, -20)]
        ))
        feature.set_reconstruction_plate_id(0)
        feature.set_valid_time(600.0, -999.0)
        path = str(tmp_path / "plate_zero.gpml")
        pygplates.FeatureCollection([feature]).write(path)

        source = make_source(synthetic_model, "rigid_plate_id", polygons=path)
        with pytest.raises(ValueError, match="plate ID 0"):
            source.at_age(10.0)


class TestSourceWithTheFallbackPlate:
    """rigid_missing_plate="topological_plate" on the synthetic model."""

    AGE = 100.0

    def test_seeds_of_a_plate_without_a_sequence_follow_their_topological_plate(
        self, synthetic_model
    ):
        source = make_source(
            synthetic_model, "rigid_plate_id", rigid_missing_plate="topological_plate"
        )
        at_zero = source.at_age(0.0)
        at_age = source.at_age(self.AGE)
        n_seed = len(source._seed_plate_ids)
        on_999 = source._seed_plate_ids == 999
        assert on_999.any()
        # Independent topological plate IDs at 0 Ma, from a topological snapshot
        # queried directly (the source uses PointRotator.assign_plate_ids).
        fallback = source._seed_fallback_plate_ids
        snapshot = pygplates.TopologicalModel(
            synthetic_model["topology_files"],
            pygplates.RotationModel(synthetic_model["rotation_files"]),
        ).topological_snapshot(0.0)
        locations = snapshot.get_point_locations([
            pygplates.PointOnSphere(float(lat), float(lon))
            for lat, lon in at_zero.latlon[:n_seed]
        ])
        independent = np.array([
            loc.located_in_resolved_boundary().get_feature().get_reconstruction_plate_id()
            for loc in locations
        ])
        assert fallback.shape == (n_seed,)
        assert np.array_equal(fallback, independent)
        assert (fallback[on_999] == 1).all()

        expected = reconstruct_points(
            synthetic_model["rotation_files"], at_zero.latlon[:n_seed][on_999],
            fallback[on_999], self.AGE,
        )
        assert angular_distance_deg(
            at_age.latlon[:n_seed][on_999], expected
        ).max() < TOLERANCE_DEG
        # They really move now, unlike with the in-place policy.
        assert angular_distance_deg(
            at_age.latlon[:n_seed][on_999], at_zero.latlon[:n_seed][on_999]
        ).min() > 1.0
        assert source._rigid_stats["n_fallback"] == int(on_999.sum())

    def test_no_warning_when_every_point_has_a_plate(self, synthetic_model):
        source = make_source(
            synthetic_model, "rigid_plate_id", rigid_missing_plate="topological_plate"
        )
        with LogCapture() as logs:
            source.at_age(self.AGE)
        assert logs.messages(logging.WARNING) == []

    def test_polygons_without_plate_ids_work(self, synthetic_model, tmp_path):
        """A plate ID 0 polygon, as in a shapefile, follows the topological plate."""
        path = write_polygon(
            tmp_path / "plate_zero.gpml",
            [(-20, -20), (-20, 20), (20, 20), (20, -20)], plate_id=0,
        )
        source = make_source(
            synthetic_model, "rigid_plate_id", polygons=path,
            rigid_missing_plate="topological_plate",
        )
        at_zero = source.at_age(0.0)
        at_age = source.at_age(self.AGE)
        n_seed = len(source._seed_plate_ids)
        assert (source._seed_plate_ids == 0).all()
        # The polygon lies inside topological plate A (plate 1), which rotates 30 deg.
        expected = reconstruct_points(
            synthetic_model["rotation_files"], at_zero.latlon[:n_seed],
            np.ones(n_seed, dtype=int), self.AGE,
        )
        assert angular_distance_deg(at_age.latlon[:n_seed], expected).max() < TOLERANCE_DEG

    def test_a_seed_in_no_topological_plate_raises(self, synthetic_model, tmp_path):
        # Longitudes 170 to 175 E are in a gap of the synthetic topology.
        path = write_polygon(
            tmp_path / "in_a_gap.gpml",
            [(-5, 170), (-5, 175), (5, 175), (5, 170)], plate_id=0,
        )
        source = make_source(
            synthetic_model, "rigid_plate_id", polygons=path,
            rigid_missing_plate="topological_plate",
        )
        with pytest.raises(ValueError, match="no topological plate"):
            source.at_age(10.0)


class TestBothModesOnTheSyntheticModel:
    """Run both rotation modes end to end on the same polygons and compare them.

    The static polygons label the point (0, 0) as plate 999, which has no rotation
    sequence, while the topological plate A around it rotates 30 deg by 100 Ma.
    The topological mode follows plate A. The rigid mode follows the polygon's
    plate ID, so those seeds stay in place. Seeds of plate 1 move in both modes.
    """

    AGE = 100.0

    @pytest.fixture
    def clouds(self, synthetic_model):
        out = {}
        for mode in ("topological", "rigid_plate_id"):
            source = make_source(synthetic_model, mode)
            at_zero = source.at_age(0.0)
            n_seed = int((at_zero.get_property("membership") > 0.5).sum())
            at_age = source.at_age(self.AGE)
            out[mode] = (source, at_zero, at_age, n_seed)
        return out

    def test_both_modes_start_from_the_same_seeds(self, clouds):
        topological = clouds["topological"]
        rigid = clouds["rigid_plate_id"]
        assert topological[3] == rigid[3] > 0
        assert angular_distance_deg(
            topological[1].latlon[:topological[3]], rigid[1].latlon[:rigid[3]]
        ).max() < TOLERANCE_DEG

    def test_both_modes_emit_the_same_channels(self, clouds):
        for _, _, at_age, _ in clouds.values():
            assert set(at_age.properties) == set(PolygonIndicatorSource.provides)

    def test_the_modes_differ_for_a_plate_without_a_sequence(self, clouds):
        rigid_source, at_zero, rigid_age, n_seed = clouds["rigid_plate_id"]
        _, _, topological_age, _ = clouds["topological"]
        on_999 = rigid_source._seed_plate_ids == 999
        assert on_999.any()
        start = at_zero.latlon[:n_seed][on_999]
        rigid_move = angular_distance_deg(rigid_age.latlon[:n_seed][on_999], start)
        topological_move = angular_distance_deg(
            topological_age.latlon[:n_seed][on_999], start
        )
        assert rigid_move.max() < TOLERANCE_DEG
        assert topological_move.min() > 1.0

    def test_the_modes_agree_for_most_seeds_of_a_plate_with_a_sequence(self, clouds):
        rigid_source, at_zero, rigid_age, n_seed = clouds["rigid_plate_id"]
        _, _, topological_age, _ = clouds["topological"]
        on_1 = rigid_source._seed_plate_ids == 1
        difference = angular_distance_deg(
            topological_age.latlon[:n_seed][on_1], rigid_age.latlon[:n_seed][on_1]
        )
        # Seeds near the edge of plate A can leave it topologically, so a few
        # differ. Most follow plate 1 in both modes.
        assert (difference < 1e-3).mean() > 0.8


# ---------------------------------------------------------------------------
# Real data (slow, skipped when the example data is absent)
# ---------------------------------------------------------------------------

ZAH = (Path(__file__).parent.parent / "examples" / "Cratons_M3_B"
       / "gplates_files" / "Zahirovic_2022")
ROTATION_FILES = [str(ZAH / "Zahirovic2022_CombinedRotations_fixed_crossovers.rot")]
TOPOLOGY_FILES = [str(ZAH / n) for n in (
    "Zahirovic2022_PlateBoundaries.gpmlz",
    "Zahirovic2022_ActiveDeformation.gpmlz",
    "Zahirovic2022_InactiveDeformation.gpmlz",
)]
CONTINENTS = str(ZAH / "shapes_continents.gpml")
STATIC = str(ZAH / "shapes_static_polygons_Merdith_et_al.gpml")
_REAL_FILES = [Path(p) for p in ROTATION_FILES + TOPOLOGY_FILES + [CONTINENTS, STATIC]]


@pytest.fixture(scope="module")
def source():
    """The real-data source, built once because loading the plate model is slow."""
    return PolygonIndicatorSource(
        ROTATION_FILES, TOPOLOGY_FILES, CONTINENTS, STATIC, 50.0,
        PolygonIndicatorConfig(
            background_point_count=40000, scalar_input_point_count=40000,
            rotation="rigid_plate_id",
        ),
    )


@pytest.mark.slow
@pytest.mark.skipif(not all(p.exists() for p in _REAL_FILES),
                    reason="Cratons_M3_B example data not present")
class TestRealData:

    @pytest.mark.parametrize("age", [100.0, 200.0, 250.0])
    def test_seeds_match_pygplates_reconstruct(self, source, age):
        source.at_age(0.0)
        seeds = source._source_points
        plate_ids, begin_ages = source._seed_plate_ids, source._seed_begin_ages
        valid = age <= begin_ages
        cloud = source.at_age(age)
        n_valid = int(valid.sum())
        assert int((cloud.get_property("membership") > 0.5).sum()) == n_valid

        expected = reconstruct_points(
            ROTATION_FILES[0], seeds.latlon[valid], plate_ids[valid], age
        )
        assert angular_distance_deg(cloud.latlon[:n_valid], expected).max() < TOLERANCE_DEG

    def test_no_seed_in_the_pacific_interior_at_200_ma(self, source):
        cloud = source.at_age(200.0)
        member = cloud.get_property("membership") > 0.5
        lon = np.mod(cloud.lonlat[member, 0], 360.0)
        lat = cloud.lonlat[member, 1]
        in_box = (lon > 170) & (lon < 230) & (lat > -30) & (lat < 50)
        assert int(in_box.sum()) == 0


CRATONS = str(ZAH.parent / "fast_velocity_anomalies_4pct.shp")


@pytest.fixture(scope="module")
def fallback_source():
    """The real-data source with the topological-plate fallback."""
    return PolygonIndicatorSource(
        ROTATION_FILES, TOPOLOGY_FILES, CONTINENTS, STATIC, 50.0,
        PolygonIndicatorConfig(
            background_point_count=40000, scalar_input_point_count=40000,
            rotation="rigid_plate_id", rigid_missing_plate="topological_plate",
        ),
    )


def oceanic_seed_count(latlon, age):
    """Seeds located in topological plates with IDs 900 to 999 (oceanic in this model)."""
    snapshot = pygplates.TopologicalModel(
        TOPOLOGY_FILES, pygplates.RotationModel(ROTATION_FILES)
    ).topological_snapshot(float(age))
    locations = snapshot.get_point_locations(
        [pygplates.PointOnSphere(float(a), float(b)) for a, b in latlon]
    )
    ids = np.array([
        loc.located_in_resolved_boundary().get_feature().get_reconstruction_plate_id()
        if loc.located_in_resolved_boundary() is not None else -1
        for loc in locations
    ])
    return int(((ids >= 900) & (ids < 1000)).sum())


@pytest.mark.slow
@pytest.mark.skipif(not all(p.exists() for p in _REAL_FILES),
                    reason="Cratons_M3_B example data not present")
class TestRealDataWithTheFallbackPlate:
    @pytest.mark.parametrize("age", [100.0, 200.0, 250.0])
    def test_seeds_match_pygplates_reconstruct_with_the_per_age_plate(self, fallback_source, age):
        fallback_source.at_age(0.0)
        seeds = fallback_source._source_points
        own = fallback_source._seed_plate_ids
        topological = fallback_source._seed_fallback_plate_ids
        valid = age <= fallback_source._seed_begin_ages
        # Independent choice: a plate is "in the tree" if the reconstruction tree
        # at this age has an edge for it.
        tree = pygplates.RotationModel(ROTATION_FILES).get_reconstruction_tree(age)
        in_tree = np.array([tree.get_edge(int(p)) is not None for p in own])
        plate = np.where(in_tree, own, topological)

        cloud = fallback_source.at_age(age)
        n_valid = int(valid.sum())
        assert int((cloud.get_property("membership") > 0.5).sum()) == n_valid
        expected = reconstruct_points(
            ROTATION_FILES[0], seeds.latlon[valid], plate[valid], age
        )
        assert angular_distance_deg(cloud.latlon[:n_valid], expected).max() < TOLERANCE_DEG

    def test_fewer_seeds_end_up_in_oceanic_plates_than_with_the_in_place_policy(
        self, fallback_source, source
    ):
        # Measured on Zahirovic 2022: 50 against 488 at 200 Ma.
        n_seed = int((fallback_source.at_age(200.0).get_property("membership") > 0.5).sum())
        with_fallback = oceanic_seed_count(fallback_source.at_age(200.0).latlon[:n_seed], 200.0)
        in_place = oceanic_seed_count(source.at_age(200.0).latlon[:n_seed], 200.0)
        assert with_fallback < 100
        assert with_fallback < in_place / 3

    def test_polygons_without_plate_ids_need_the_fallback(self):
        in_place = PolygonIndicatorSource(
            ROTATION_FILES, TOPOLOGY_FILES, CRATONS, STATIC, 200.0,
            PolygonIndicatorConfig(rotation="rigid_plate_id"),
        )
        with pytest.raises(ValueError, match="plate ID 0"):
            in_place.at_age(100.0)

        with_fallback = PolygonIndicatorSource(
            ROTATION_FILES, TOPOLOGY_FILES, CRATONS, STATIC, 200.0,
            PolygonIndicatorConfig(rotation="rigid_plate_id",
                                   rigid_missing_plate="topological_plate"),
        )
        cloud = with_fallback.at_age(100.0)
        assert int((cloud.get_property("membership") > 0.5).sum()) > 0

