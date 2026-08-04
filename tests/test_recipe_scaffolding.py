"""Tests for the age-source protocol, cloud dispatch, grid loader and checkpoints.

Nothing here needs plate data or a reconstruction: the point of this
scaffolding is that a consumer can be built and tested against it with numpy
alone, and these tests are the first demonstration of that.
"""

import numpy as np
import pytest

from gtrack import AgeCloudSource, CheckpointPolicy, PointCloud
from gtrack.io_formats import load_latlon_grid_hdf5


h5py = pytest.importorskip("h5py")


# ---------------------------------------------------------------------------
# AgeCloudSource
# ---------------------------------------------------------------------------

class NumpyDouble:
    """The whole point of the protocol: a source with no gtrack machinery."""

    provides = frozenset({"thickness"})
    monotonic_backward = False

    def __init__(self):
        self.seen = []

    def at_age(self, age):
        self.seen.append(age)
        latlon = np.array([[0.0, 0.0], [10.0, 20.0]])
        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property("thickness", np.array([100.0, 120.0]))
        return cloud

    def validate_age(self, age):
        pass


class ForwardOnlyDouble(NumpyDouble):
    monotonic_backward = True


class MissingAtAge:
    provides = frozenset({"thickness"})
    monotonic_backward = False

    def validate_age(self, age):
        pass


class AnnotatedOnly:
    """Annotations create no attributes, so this does NOT satisfy the protocol.

    Worth pinning: it is the natural way to write a class against a Protocol
    and it silently fails isinstance.
    """

    provides: frozenset
    monotonic_backward: bool

    def at_age(self, age):
        raise NotImplementedError

    def validate_age(self, age):
        pass


class TestAgeCloudSource:
    def test_numpy_double_satisfies_it(self):
        assert isinstance(NumpyDouble(), AgeCloudSource)

    def test_missing_method_is_rejected(self):
        assert not isinstance(MissingAtAge(), AgeCloudSource)

    def test_annotation_without_value_is_rejected(self):
        assert not isinstance(AnnotatedOnly(), AgeCloudSource)

    def test_monotonic_backward_is_readable_without_calling_anything(self):
        assert NumpyDouble().monotonic_backward is False
        assert ForwardOnlyDouble().monotonic_backward is True

    def test_provides_excludes_coordinates(self):
        source = NumpyDouble()
        cloud = source.at_age(0.0)
        assert "xyz" not in source.provides
        for name in source.provides:
            assert cloud.get_property(name) is not None


# ---------------------------------------------------------------------------
# PointCloud.from_data
# ---------------------------------------------------------------------------

class TestFromData:
    def test_passes_a_cloud_through_unchanged(self):
        original = PointCloud.from_latlon(np.array([[1.0, 2.0]]))
        original.add_property("thickness", np.array([7.0]))
        assert PointCloud.from_data(original, "thickness") is original

    def test_latlon_values_tuple(self):
        latlon = np.array([[0.0, 0.0], [45.0, 90.0], [-30.0, -60.0]])
        values = np.array([1.0, 2.0, 3.0])
        cloud = PointCloud.from_data((latlon, values), "thickness")
        assert cloud.n_points == 3
        np.testing.assert_array_equal(cloud.get_property("thickness"), values)

    def test_scalar_broadcasts_onto_a_sphere(self):
        cloud = PointCloud.from_data(50.0, "thickness", n_points_fallback=500)
        assert cloud.n_points == 500
        np.testing.assert_array_equal(
            cloud.get_property("thickness"), np.full(500, 50.0)
        )

    def test_integer_scalar_is_a_scalar_too(self):
        cloud = PointCloud.from_data(7, "thickness", n_points_fallback=200)
        assert cloud.n_points == 200
        assert cloud.get_property("thickness").dtype == float

    def test_fallback_size_is_ignored_for_everything_but_a_scalar(self):
        latlon = np.array([[0.0, 0.0], [45.0, 90.0]])
        cloud = PointCloud.from_data(
            (latlon, np.array([1.0, 2.0])), "thickness", n_points_fallback=999
        )
        assert cloud.n_points == 2

    def test_path_input(self, tmp_path):
        path = tmp_path / "grid.h5"
        with h5py.File(path, "w") as f:
            f["lon"] = np.array([0.0, 90.0])
            f["lat"] = np.array([0.0, 45.0])
            f["thickness"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        cloud = PointCloud.from_data(path, "thickness")
        assert cloud.n_points == 4

    def test_rejects_anything_else(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            PointCloud.from_data(object(), "thickness")
        with pytest.raises(TypeError, match="Unsupported data type"):
            PointCloud.from_data([1, 2, 3], "thickness")

    def test_rejects_a_bool(self):
        # bool is a subclass of int, so without the explicit guard True would
        # quietly become a thickness of 1 everywhere.
        with pytest.raises(TypeError, match="Unsupported data type"):
            PointCloud.from_data(True, "thickness")


# ---------------------------------------------------------------------------
# load_latlon_grid_hdf5
# ---------------------------------------------------------------------------

def write_grid(path, lon, lat, values, name="thickness"):
    with h5py.File(path, "w") as f:
        f["lon"] = np.asarray(lon)
        f["lat"] = np.asarray(lat)
        f[name] = np.asarray(values)


class TestLoadLatLonGridHdf5:
    def test_one_dimensional_axes_are_meshed(self, tmp_path):
        path = tmp_path / "axes.h5"
        write_grid(path, [0.0, 10.0, 20.0], [-5.0, 5.0], [[1.0, 2.0, 3.0],
                                                          [4.0, 5.0, 6.0]])
        cloud = load_latlon_grid_hdf5(path, "thickness")
        assert cloud.n_points == 6
        np.testing.assert_array_equal(
            cloud.get_property("thickness"), np.arange(1.0, 7.0)
        )

    def test_two_dimensional_grids_are_only_flattened(self, tmp_path):
        path = tmp_path / "grids.h5"
        lon = np.array([[0.0, 10.0], [0.0, 10.0]])
        lat = np.array([[-5.0, -5.0], [5.0, 5.0]])
        write_grid(path, lon, lat, np.array([[1.0, 2.0], [3.0, 4.0]]))
        cloud = load_latlon_grid_hdf5(path, "thickness")
        assert cloud.n_points == 4
        np.testing.assert_array_equal(
            cloud.get_property("thickness"), np.array([1.0, 2.0, 3.0, 4.0])
        )

    def test_longitudes_above_180_are_wrapped(self, tmp_path):
        path = tmp_path / "wrap.h5"
        write_grid(path, [0.0, 190.0, 350.0], [0.0], [[1.0, 2.0, 3.0]])
        cloud = load_latlon_grid_hdf5(path, "thickness")
        lons = cloud.latlon[:, 1]
        assert lons.min() >= -180.0
        assert lons.max() <= 180.0
        # 190 -> -170 and 350 -> -10, and both survive the round trip through
        # cartesian coordinates.
        np.testing.assert_allclose(sorted(lons), [-170.0, -10.0, 0.0], atol=1e-9)

    def test_z_is_accepted_as_a_fallback_name(self, tmp_path):
        path = tmp_path / "gmt.h5"
        write_grid(path, [0.0, 10.0], [0.0], [[1.0, 2.0]], name="z")
        cloud = load_latlon_grid_hdf5(path, "thickness")
        np.testing.assert_array_equal(
            cloud.get_property("thickness"), np.array([1.0, 2.0])
        )

    def test_missing_values_dataset_is_reported_with_what_is_there(self, tmp_path):
        path = tmp_path / "bad.h5"
        write_grid(path, [0.0], [0.0], [[1.0]], name="something_else")
        with pytest.raises(KeyError) as excinfo:
            load_latlon_grid_hdf5(path, "thickness")
        assert "something_else" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CheckpointPolicy
# ---------------------------------------------------------------------------

class TestCheckpointPolicy:
    def test_filename_round_trip(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        path = policy.path_for(123.4)
        assert path.name == "ocean_checkpoint_123Ma.npz"
        path.touch()
        assert policy.best_at_or_before(100.0) == path

    def test_age_is_rounded_not_truncated(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        assert policy.path_for(122.6).name == "ocean_checkpoint_123Ma.npz"

    def test_prefix_is_honoured_both_ways(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=10.0, prefix="craton")
        path = policy.path_for(80.0)
        assert path.name == "craton_80Ma.npz"
        path.touch()
        assert policy.best_at_or_before(50.0) == path
        # A file written under a different prefix is not a candidate.
        (tmp_path / "ocean_checkpoint_90Ma.npz").touch()
        assert policy.best_at_or_before(50.0) == path

    def test_picks_the_youngest_candidate_not_older_than_the_target(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        for age in (100, 200, 300):
            policy.path_for(age).touch()
        # 150 Ma: 100 is too young to resume from, 200 is the least stepping.
        assert policy.best_at_or_before(150.0).name == "ocean_checkpoint_200Ma.npz"
        # Exactly on a checkpoint, that checkpoint is the answer.
        assert policy.best_at_or_before(200.0).name == "ocean_checkpoint_200Ma.npz"

    def test_empty_directory(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        assert policy.best_at_or_before(100.0) is None

    def test_no_candidate_old_enough(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        policy.path_for(50.0).touch()
        assert policy.best_at_or_before(400.0) is None

    def test_ignores_files_that_are_not_checkpoints(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        (tmp_path / "ocean_checkpoint_100Ma.txt").touch()      # wrong suffix
        (tmp_path / "notes.npz").touch()                       # wrong stem
        (tmp_path / "ocean_checkpoint_abcMa.npz").touch()      # unparseable age
        assert policy.best_at_or_before(50.0) is None

    def test_missing_directory_raises_rather_than_saying_none(self, tmp_path):
        policy = CheckpointPolicy(tmp_path / "never_created", interval_myr=50.0)
        with pytest.raises(FileNotFoundError):
            policy.best_at_or_before(100.0)

    def test_is_due(self, tmp_path):
        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        assert policy.is_due(200.0, None) is True          # nothing saved yet
        assert policy.is_due(160.0, 200.0) is False        # 40 < 50
        assert policy.is_due(150.0, 200.0) is True         # exactly 50
        assert policy.is_due(100.0, 200.0) is True         # well past
        assert policy.is_due(250.0, 200.0) is True         # direction-agnostic

    def test_is_frozen(self, tmp_path):
        import dataclasses

        policy = CheckpointPolicy(tmp_path, interval_myr=50.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.interval_myr = 10.0

    def test_rejects_nonsense_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="interval_myr must be positive"):
            CheckpointPolicy(tmp_path, interval_myr=0.0)
        with pytest.raises(ValueError, match="prefix must not be empty"):
            CheckpointPolicy(tmp_path, interval_myr=50.0, prefix="")


def test_new_names_are_exported():
    import gtrack

    for name in ("AgeCloudSource", "CheckpointPolicy", "load_latlon_grid_hdf5"):
        assert name in gtrack.__all__
        assert hasattr(gtrack, name)
    assert hasattr(PointCloud, "from_data")
