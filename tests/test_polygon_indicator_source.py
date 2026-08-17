"""Test the recipe and channel contract for ``PolygonIndicatorSource``.

The polygon filter, rotator and the union helper are replaced by fakes. What is
under test is the recipe, channel names, source-point filter, and stateless age
access. These tests do not require plate data.
"""

import numpy as np
import pytest

from gtrack import AgeCloudSource, PointCloud
from gtrack.point_rotation import PointRotator
from gtrack.polygon_filter import PolygonFilter
from gtrack.sources import (
    PolygonIndicatorConfig,
    PolygonIndicatorSource,
    build_indicator_source,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeFilter:
    instances = []

    def __init__(self, polygon_files, rotation_files):
        self.polygon_files = polygon_files
        self.rotation_files = rotation_files
        self.calls = []
        FakeFilter.instances.append(self)

    def filter_inside(self, cloud, at_age):
        self.calls.append(at_age)
        # Keep the first half, so "filtering happened" is observable.
        keep = max(1, cloud.n_points // 2)
        latlon = cloud.latlon[:keep]
        seeds = PointCloud.from_latlon(latlon)
        for name in cloud.properties:
            seeds.add_property(name, cloud.get_property(name)[:keep])
        return seeds


class FakeRotator:
    instances = []

    def __init__(self, rotation_files, topology_files, static_polygons):
        self.static_polygons = static_polygons
        FakeRotator.instances.append(self)


def fake_build_indicator_source(source_cloud, rotator, target_age, **kwargs):
    """Record the arguments and return the source channels."""
    fake_build_indicator_source.calls.append(
        {"source_cloud": source_cloud, "rotator": rotator,
         "target_age": target_age, **kwargs}
    )
    cloud = PointCloud.from_latlon(np.array([[0.0, 0.0], [20.0, 30.0]]))
    for name in source_cloud.properties:
        cloud.add_property(name, np.array([1.0, 0.0]))
    membership = kwargs.get("membership_property")
    if membership is not None:
        cloud.add_property(membership, np.array([1.0, 0.0]))
    return cloud


@pytest.fixture
def fakes(monkeypatch, bind_signature):
    signature_bound = bind_signature
    FakeFilter.instances = []
    FakeRotator.instances = []
    fake_build_indicator_source.calls = []
    # Every stand-in uses the signature of the object that it replaces.
    # Thus, production accepts each call that this suite accepts. Without it,
    # the **kwargs below swallows any keyword, including one the real helper
    # does not have. See conftest.signature_bound.
    monkeypatch.setattr(
        "gtrack.polygon_filter.PolygonFilter",
        signature_bound(PolygonFilter, FakeFilter),
    )
    monkeypatch.setattr(
        "gtrack.sources.PointRotator", signature_bound(PointRotator, FakeRotator)
    )
    monkeypatch.setattr(
        "gtrack.sources.build_indicator_source",
        signature_bound(build_indicator_source, fake_build_indicator_source),
    )
    return fake_build_indicator_source


def make_source(config=None, thickness_data=200.0):
    return PolygonIndicatorSource(
        rotation_files=["rot.rot"],
        topology_files=["topo.gpml"],
        polygons="cratons.shp",
        static_polygons="static.shp",
        thickness_data=thickness_data,
        config=config,
    )


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

class TestSatisfiesProtocol:
    def test_isinstance(self, fakes):
        assert isinstance(make_source(), AgeCloudSource)

    def test_provides_excludes_coordinates(self):
        assert PolygonIndicatorSource.provides == frozenset(
            {"masked_thickness", "membership"}
        )
        assert "xyz" not in PolygonIndicatorSource.provides

    def test_declares_no_walk(self):
        assert PolygonIndicatorSource.monotonic_backward is False


# ---------------------------------------------------------------------------
# The channel names — this is D4
# ---------------------------------------------------------------------------

class TestChannelNames:
    def test_thickness_channel_is_masked(self, fakes):
        # The distinct name prevents consumers from treating it as thickness.
        assert "thickness" not in PolygonIndicatorSource.provides
        assert PolygonIndicatorSource.PROPERTY_NAME == "masked_thickness"

    def test_emitted_cloud_carries_exactly_provides(self, fakes):
        cloud = make_source().at_age(100.0)
        assert set(cloud.properties) == set(PolygonIndicatorSource.provides)

    def test_seeds_are_labelled_with_the_masked_name(self, fakes):
        make_source().at_age(100.0)
        source_points = fakes.calls[0]["source_cloud"]
        assert "masked_thickness" in source_points.properties
        assert "thickness" not in source_points.properties

    def test_membership_is_passed_explicitly_not_defaulted(self, fakes):
        # F20: the string used to be coupled across the seam by two defaults
        # that happened to agree. It is now an argument.
        make_source().at_age(100.0)
        assert fakes.calls[0]["membership_property"] == "membership"


# ---------------------------------------------------------------------------
# Independent point counts
# ---------------------------------------------------------------------------

class TestSourceRecipeConfiguration:
    def test_config_reaches_the_source_recipe(self, fakes):
        config = PolygonIndicatorConfig(
            background_point_count=5000,
            scalar_input_point_count=800,
            exclusion_factor=2.5,
        )
        source = make_source(config, thickness_data=200.0)
        source.at_age(100.0)
        call = fakes.calls[0]
        assert call["background_point_count"] == 5000
        # The fake filter keeps half, so 800 in gives 400 seeds out.
        assert call["source_cloud"].n_points == 400
        assert call["exclusion_factor"] == 2.5
        assert call["background_value"] == 0.0

    def test_scalar_count_is_inert_for_non_scalar_data(self, fakes):
        latlon = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        source = make_source(
            PolygonIndicatorConfig(scalar_input_point_count=999999),
            thickness_data=(latlon, values),
        )
        source.at_age(100.0)
        assert fakes.calls[0]["source_cloud"].n_points == 2

# ---------------------------------------------------------------------------
# Seeds are filtered once, at present day
# ---------------------------------------------------------------------------

class TestSeeds:
    def test_filtered_once_and_reused(self, fakes):
        source = make_source()
        source.at_age(300.0)
        source.at_age(100.0)
        source.at_age(0.0)
        assert FakeFilter.instances[0].calls == [0.0]
        first = fakes.calls[0]["source_cloud"]
        assert all(call["source_cloud"] is first for call in fakes.calls)

    def test_the_rotator_is_handed_over(self, fakes):
        make_source().at_age(100.0)
        assert fakes.calls[0]["rotator"] is FakeRotator.instances[0]


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------

class TestStateless:
    def test_ages_may_be_requested_in_any_order(self, fakes):
        source = make_source()
        source.at_age(100.0)
        source.at_age(300.0)   # older than the previous: fine, no walk
        source.at_age(50.0)
        assert [c["target_age"] for c in fakes.calls] == [100.0, 300.0, 50.0]

    def test_repeating_an_age_repeats_the_request(self, fakes):
        source = make_source()
        source.at_age(100.0)
        source.at_age(100.0)
        assert len(fakes.calls) == 2

    def test_negative_ages_are_refused(self, fakes):
        with pytest.raises(ValueError, match="negative"):
            make_source().validate_age(-1.0)


# ---------------------------------------------------------------------------
# Lazy construction
# ---------------------------------------------------------------------------

class TestLazyConstruction:
    def test_nothing_is_built_until_first_use(self, fakes):
        make_source()
        assert FakeFilter.instances == []
        assert FakeRotator.instances == []

    def test_built_once_across_many_ages(self, fakes):
        source = make_source()
        source.at_age(300.0)
        source.at_age(100.0)
        assert len(FakeRotator.instances) == 1
        assert len(FakeFilter.instances) == 1

    def test_construction_validates_without_touching_data(self, fakes):
        with pytest.raises(ValueError, match="polygons is required"):
            PolygonIndicatorSource(
                rotation_files=[], topology_files=[], polygons=None,
                static_polygons="s.shp", thickness_data=1.0,
            )
        with pytest.raises(ValueError, match="static_polygons is required"):
            PolygonIndicatorSource(
                rotation_files=[], topology_files=[], polygons="p.shp",
                static_polygons=None, thickness_data=1.0,
            )
        assert FakeRotator.instances == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = PolygonIndicatorConfig()
        assert config.background_point_count == 20000
        assert config.scalar_input_point_count == 20000
        assert config.exclusion_factor == 1.0

    def test_is_frozen(self):
        import dataclasses

        config = PolygonIndicatorConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.background_point_count = 5000

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="background_point_count must be positive"):
            PolygonIndicatorConfig(background_point_count=0)
        with pytest.raises(ValueError, match="scalar_input_point_count must be positive"):
            PolygonIndicatorConfig(scalar_input_point_count=0)
        with pytest.raises(ValueError, match="exclusion_factor must be non-negative"):
            PolygonIndicatorConfig(exclusion_factor=-1.0)

    def test_there_is_no_single_n_points(self):
        # The conflation this class exists to remove: one knob doing both jobs.
        fields = set(PolygonIndicatorConfig.__dataclass_fields__)
        assert "n_points" not in fields
        assert {"background_point_count", "scalar_input_point_count"} <= fields
