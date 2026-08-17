"""Tests for LithosphereCloudSource's recipe and control flow.

The tracker, rotator and polygon filter are replaced by fakes. What is under
test is the recipe — which parameters reach the tracker, when the tracker is
rebuilt, when a checkpoint is written, what the forward-only contract
refuses, and what the returned cloud carries. None of that is pygplates
behaviour, so none of it needs plate data.

The fakes are installed by monkeypatching the names in gtrack.lithosphere,
which is where they are looked up.
"""

import numpy as np
import pytest

from gtrack import AgeCloudSource, CheckpointPolicy, PointCloud
from gtrack.config import TracerConfig
from gtrack.hpc_integration import SeafloorAgeTracker
from gtrack.lithosphere import (
    LithosphereCloudConfig,
    LithosphereCloudSource,
    OceanicLidWeight,
    quintic_smoothstep,
)
from gtrack.point_rotation import PointRotator
from gtrack.polygon_filter import PolygonFilter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTracker:
    """Records the calls the recipe makes, and hands back a two-point cloud."""

    instances = []

    def __init__(self, rotation_files, topology_files, continental_polygons, config):
        self.rotation_files = rotation_files
        self.topology_files = topology_files
        self.continental_polygons = continental_polygons
        self.config = config
        self.initialized_at = None
        self.steps = []
        self.reinits = []
        self.saved = []
        self.loaded = []
        self.load_should_fail = False
        self.save_should_fail = False
        self.current_age = None
        FakeTracker.instances.append(self)

    def initialize(self, starting_age):
        self.initialized_at = starting_age
        self.current_age = starting_age

    def step_to(self, target_age):
        self.steps.append(target_age)
        self.current_age = target_age
        cloud = PointCloud.from_latlon(np.array([[0.0, 0.0], [10.0, 10.0]]))
        cloud.add_property("age", np.array([10.0, 20.0]))
        return cloud

    def tracker_rebuild(self, n_points=None, **kwargs):
        self.reinits.append(n_points)

    def save_checkpoint(self, filepath):
        if self.save_should_fail:
            raise OSError("disk full")
        self.saved.append(filepath)

    def load_checkpoint(self, filepath):
        if self.load_should_fail:
            raise ValueError("incompatible checkpoint")
        self.loaded.append(filepath)
        self.current_age = 300.0


class FakeRotator:
    def __init__(self, rotation_files, topology_files, static_polygons):
        self.static_polygons = static_polygons
        self.calls = []

    def rotate(self, cloud, from_age, to_age):
        self.calls.append((from_age, to_age))
        rotated = PointCloud.from_latlon(np.array([[45.0, 45.0]]))
        rotated.add_property("thickness", np.array([150.0]))
        return rotated


class FakeFilter:
    def __init__(self, polygon_files, rotation_files):
        self.polygon_files = polygon_files
        self.calls = []

    def filter_inside(self, cloud, at_age):
        self.calls.append(at_age)
        return cloud


@pytest.fixture
def fakes(monkeypatch, bind_signature):
    FakeTracker.instances = []
    # Each stand-in uses the signature of the object that it replaces.
    # Thus, production accepts each call that this suite accepts.
    monkeypatch.setattr(
        "gtrack.lithosphere.SeafloorAgeTracker",
        bind_signature(SeafloorAgeTracker, FakeTracker),
    )
    monkeypatch.setattr(
        "gtrack.lithosphere.PointRotator", bind_signature(PointRotator, FakeRotator)
    )
    monkeypatch.setattr(
        "gtrack.lithosphere.PolygonFilter", bind_signature(PolygonFilter, FakeFilter)
    )
    return FakeTracker


def make_source(config=None, continental_data=100.0,
                plate_model_max_age_ma=400.0,
                oceanic_weight_from_thickness=None):
    return LithosphereCloudSource(
        rotation_files=["rot.rot"],
        topology_files=["topo.gpml"],
        continental_polygons="cont.shp",
        static_polygons="static.shp",
        continental_data=continental_data,
        oceanic_thickness_from_age=lambda ages: 2.32 * np.sqrt(ages),
        plate_model_max_age_ma=plate_model_max_age_ma,
        config=config,
        oceanic_weight_from_thickness=oceanic_weight_from_thickness,
    )


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

class TestSatisfiesProtocol:
    def test_isinstance(self, fakes):
        assert isinstance(make_source(), AgeCloudSource)

    def test_provides_excludes_coordinates(self):
        assert LithosphereCloudSource.provides == frozenset({"thickness", "age"})
        assert "xyz" not in LithosphereCloudSource.provides

    def test_declares_forward_only(self):
        assert LithosphereCloudSource.monotonic_backward is True


# ---------------------------------------------------------------------------
# Oceanic lateral weight
# ---------------------------------------------------------------------------

class TestOceanicLidWeight:
    @staticmethod
    def age_to_thickness(ages):
        return np.minimum(10.0 * np.sqrt(ages), 100.0)

    def test_quintic_smoothstep_clamps_and_reaches_midpoint(self):
        result = quintic_smoothstep(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_thresholds_follow_the_age_to_thickness_map(self):
        weight = OceanicLidWeight(
            self.age_to_thickness,
            zero_weight_age_myr=1.0,
            full_weight_age_myr=25.0,
        )
        assert weight.zero_weight_thickness_km == pytest.approx(10.0)
        assert weight.full_weight_thickness_km == pytest.approx(50.0)
        np.testing.assert_allclose(
            weight(np.array([0.0, 10.0, 30.0, 50.0, 100.0])),
            [0.0, 0.0, 0.5, 1.0, 1.0],
        )

    def test_rejects_reversed_age_thresholds(self):
        with pytest.raises(ValueError, match="zero_weight_age_myr must be less"):
            OceanicLidWeight(
                self.age_to_thickness,
                zero_weight_age_myr=25.0,
                full_weight_age_myr=1.0,
            )

    def test_rejects_full_weight_at_the_thickness_cap(self):
        with pytest.raises(ValueError, match="thickness cap"):
            OceanicLidWeight(
                self.age_to_thickness,
                zero_weight_age_myr=1.0,
                full_weight_age_myr=100.0,
            )


# ---------------------------------------------------------------------------
# Construction is cheap
# ---------------------------------------------------------------------------

class TestLazyConstruction:
    def test_nothing_is_built_until_first_use(self, fakes):
        make_source()
        assert fakes.instances == []

    def test_first_at_age_builds_once(self, fakes):
        source = make_source()
        source.at_age(300.0)
        source.at_age(200.0)
        assert len(fakes.instances) == 1

    def test_construction_validates_without_touching_data(self, fakes):
        with pytest.raises(ValueError, match="continental_polygons is required"):
            LithosphereCloudSource(
                rotation_files=[], topology_files=[],
                continental_polygons=None, static_polygons="s.shp",
                continental_data=1.0, oceanic_thickness_from_age=lambda a: a,
                plate_model_max_age_ma=400.0,
            )
        assert fakes.instances == []


# ---------------------------------------------------------------------------
# The tracer config reaches the tracker intact
# ---------------------------------------------------------------------------

class TestTracerConfigIsNotShadowed:
    def test_the_exact_object_is_handed_over(self, fakes):
        tracer = TracerConfig(tracker_step_myr=2.0, tracker_point_count=12345)
        source = make_source(LithosphereCloudConfig(tracer=tracer))
        source.at_age(300.0)
        assert fakes.instances[0].config is tracer

    def test_rebuild_uses_the_tracer_mesh_size(self, fakes):
        config = LithosphereCloudConfig(
            tracer=TracerConfig(tracker_point_count=7777),
            tracker_rebuild_interval_myr=50.0,
        )
        source = make_source(config)
        source.at_age(390.0)
        source.at_age(300.0)
        assert fakes.instances[0].reinits == [7777]

    def test_there_is_no_override_dict(self):
        # A passthrough dict is how the original bug was expressible. Its
        # absence is the fix, so its absence is worth asserting.
        fields = set(LithosphereCloudConfig.__dataclass_fields__)
        assert "gtrack_config" not in fields
        assert not any("kwargs" in name for name in fields)


# ---------------------------------------------------------------------------
# The returned cloud
# ---------------------------------------------------------------------------

class TestCloud:
    def test_carries_exactly_provides(self, fakes):
        source = make_source()
        cloud = source.at_age(300.0)
        for name in LithosphereCloudSource.provides:
            assert cloud.get_property(name) is not None
        assert set(cloud.properties) == set(LithosphereCloudSource.provides)

    def test_no_lateral_weight_by_default(self, fakes):
        source = make_source()
        assert "lateral_weight" not in source.provides
        cloud = source.at_age(300.0)
        assert "lateral_weight" not in cloud.properties

    def test_lateral_weight_maps_ocean_only(self, fakes):
        source = make_source(
            oceanic_weight_from_thickness=lambda thickness_km: thickness_km / 10.0
        )
        assert "lateral_weight" in source.provides
        cloud = source.at_age(300.0)
        weight = cloud.get_property("lateral_weight")
        # Two ocean seeds first, one continental seed last.
        thickness = cloud.get_property("thickness")
        np.testing.assert_allclose(
            weight[:2], np.clip(thickness[:2] / 10.0, 0.0, 1.0)
        )
        assert weight[-1] == 1.0

    def test_ocean_then_continental(self, fakes):
        source = make_source()
        cloud = source.at_age(300.0)
        # Two ocean seeds from the fake tracker, one continental from the
        # fake rotator, ocean first.
        assert cloud.n_points == 3
        np.testing.assert_allclose(
            cloud.get_property("thickness")[:2], 2.32 * np.sqrt([10.0, 20.0])
        )

    def test_continental_seeds_get_the_nominal_age(self, fakes):
        source = make_source(
            LithosphereCloudConfig(continental_material_age_myr=500.0)
        )
        cloud = source.at_age(300.0)
        assert cloud.get_property("age")[-1] == 500.0

    def test_thickness_comes_from_the_age_map(self, fakes):
        source = LithosphereCloudSource(
            rotation_files=[], topology_files=[],
            continental_polygons="c.shp", static_polygons="s.shp",
            continental_data=100.0,
            oceanic_thickness_from_age=lambda ages: ages * 3.0,
            plate_model_max_age_ma=400.0,
        )
        cloud = source.at_age(300.0)
        np.testing.assert_allclose(
            cloud.get_property("thickness")[:2], [30.0, 60.0]
        )

    def test_continental_cloud_is_clipped_at_present_day(self, fakes, monkeypatch):
        captured = {}

        class RecordingFilter(FakeFilter):
            def filter_inside(self, cloud, at_age):
                captured["at_age"] = at_age
                return super().filter_inside(cloud, at_age)

        monkeypatch.setattr("gtrack.lithosphere.PolygonFilter", RecordingFilter)
        make_source().at_age(300.0)
        assert captured["at_age"] == 0.0


# ---------------------------------------------------------------------------
# The forward-only contract
# ---------------------------------------------------------------------------

class TestForwardOnly:
    def test_walking_backwards_is_fine(self, fakes):
        source = make_source()
        source.at_age(300.0)
        source.at_age(200.0)
        source.at_age(200.0)
        assert fakes.instances[0].steps == [300, 200, 200]

    def test_rewinding_is_refused(self, fakes):
        source = make_source()
        source.at_age(200.0)
        with pytest.raises(ValueError, match="older than the last walked age"):
            source.at_age(300.0)

    def test_oldest_requested_age_fires_before_any_walking(self, fakes):
        source = make_source(LithosphereCloudConfig(oldest_requested_age_ma=250.0))
        with pytest.raises(ValueError, match="oldest_requested_age_ma"):
            source.validate_age(300.0)
        # Nothing was built, so the refusal really did precede the walk.
        assert fakes.instances == []

    def test_oldest_requested_age_does_not_permit_revisiting(self, fakes):
        source = make_source(LithosphereCloudConfig(oldest_requested_age_ma=400.0))
        source.at_age(200.0)
        with pytest.raises(ValueError, match="older than the last walked age"):
            source.at_age(300.0)

    def test_range_checks(self, fakes):
        source = make_source(plate_model_max_age_ma=400.0)
        with pytest.raises(ValueError, match="negative"):
            source.validate_age(-1.0)
        with pytest.raises(ValueError, match="older than the plate model"):
            source.validate_age(500.0)

    def test_oldest_requested_age_beyond_model_is_refused(self, fakes):
        with pytest.raises(ValueError, match="exceeds plate_model_max_age_ma"):
            make_source(
                LithosphereCloudConfig(oldest_requested_age_ma=500.0),
                plate_model_max_age_ma=400.0,
            )


# ---------------------------------------------------------------------------
# Tracker rebuild cadence
# ---------------------------------------------------------------------------

class TestTrackerRebuild:
    def test_fires_on_the_interval_and_resets_the_clock(self, fakes):
        source = make_source(
            LithosphereCloudConfig(tracker_rebuild_interval_myr=50.0)
        )
        source.at_age(400.0)   # Initialises without a tracker rebuild.
        source.at_age(360.0)   # This is 40 Myr after initialisation.
        assert fakes.instances[0].reinits == []
        source.at_age(350.0)   # This reaches the 50 Myr interval.
        assert len(fakes.instances[0].reinits) == 1
        source.at_age(320.0)   # This is 30 Myr after the rebuild at 350 Ma.
        assert len(fakes.instances[0].reinits) == 1
        source.at_age(300.0)   # 50 since 350
        assert len(fakes.instances[0].reinits) == 2


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_no_policy_means_no_saving(self, fakes):
        source = make_source()
        source.at_age(300.0)
        assert fakes.instances[0].saved == []

    def test_first_step_saves_then_cadence_applies(self, fakes, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        source = make_source(LithosphereCloudConfig(checkpoint=policy))
        source.at_age(400.0)
        assert len(fakes.instances[0].saved) == 1
        source.at_age(380.0)   # 20 < 50
        assert len(fakes.instances[0].saved) == 1
        source.at_age(350.0)   # 50 >= 50
        assert len(fakes.instances[0].saved) == 2

    def test_saved_path_follows_the_policy(self, fakes, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0, prefix="lith")
        source = make_source(LithosphereCloudConfig(checkpoint=policy))
        source.at_age(321.4)
        assert fakes.instances[0].saved[0].name == "lith_321Ma.npz"

    def test_resumes_from_a_checkpoint_when_one_fits(self, fakes, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        policy.path_for(300.0).touch()
        source = make_source(LithosphereCloudConfig(checkpoint=policy))
        source.at_age(250.0)
        tracker = fakes.instances[0]
        assert tracker.loaded[0].name == "ocean_checkpoint_300Ma.npz"
        assert tracker.initialized_at is None    # no full walk needed

    def test_falls_back_to_a_full_walk_when_the_checkpoint_is_unusable(
        self, fakes, tmp_path, monkeypatch
    ):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        policy.path_for(300.0).touch()

        class FailingLoad(FakeTracker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.load_should_fail = True

        monkeypatch.setattr("gtrack.lithosphere.SeafloorAgeTracker", FailingLoad)
        source = make_source(LithosphereCloudConfig(checkpoint=policy))
        source.at_age(250.0)          # must not raise
        assert fakes.instances[0].initialized_at == 400

    def test_a_failed_save_does_not_end_the_run(self, fakes, tmp_path, monkeypatch):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)

        class FailingSave(FakeTracker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.save_should_fail = True

        monkeypatch.setattr("gtrack.lithosphere.SeafloorAgeTracker", FailingSave)
        source = make_source(LithosphereCloudConfig(checkpoint=policy))
        cloud = source.at_age(300.0)   # must not raise
        assert cloud.n_points == 3


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = LithosphereCloudConfig()
        assert config.tracker_rebuild_interval_myr == 50.0
        assert config.checkpoint is None
        assert config.continental_material_age_myr == 500.0
        assert config.oldest_requested_age_ma is None
        assert isinstance(config.tracer, TracerConfig)

    def test_each_instance_gets_its_own_tracer(self):
        # default_factory, not a shared mutable default.
        assert LithosphereCloudConfig().tracer is not LithosphereCloudConfig().tracer

    def test_is_frozen(self):
        import dataclasses

        config = LithosphereCloudConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tracker_rebuild_interval_myr = 10.0

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="tracker_rebuild_interval_myr must be positive"):
            LithosphereCloudConfig(tracker_rebuild_interval_myr=0.0)
        with pytest.raises(ValueError, match="oldest_requested_age_ma must be non-negative"):
            LithosphereCloudConfig(oldest_requested_age_ma=-1.0)
        with pytest.raises(ValueError, match="scalar_continental_point_count must be positive"):
            LithosphereCloudConfig(scalar_continental_point_count=0)


def test_scalar_continental_data_uses_the_fallback_size(fakes):
    source = make_source(
        LithosphereCloudConfig(scalar_continental_point_count=512),
        continental_data=100.0,
    )
    source.at_age(300.0)
    assert source._continental_present.n_points == 512
