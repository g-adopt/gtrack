"""Regression tests for the deforming-aware (topological) PointRotator engine.

These lock in the Phase 1-5 behaviour of CHANGE-PLAN.md against the tiny
synthetic topological model (see tests/data/make_synthetic_model.py and the
``synthetic_model`` fixture in conftest.py):

* no point is ever silently dropped (``deactivate_points=None`` -> n_out==n_in);
* an out-of-circuit point that the OLD rigid engine dropped is now kept and
  advected;
* round-trips (both directions) return to the origin;
* properties, plate_ids, and input ORDER survive rotation;
* topological advection equals rigid rotation using the topology's own plate id;
* deep-time (0 -> oldest) does not raise.
"""

import warnings

import numpy as np
import pygplates
import pytest

from gtrack import PointCloud, PointRotator
from gtrack.geometry import XYZ2LatLon
from gtrack.point_rotation import (
    _move_points_batched,
    _unreconstructable_plate_ids,
)


def _rotator(sm, **kw):
    return PointRotator(
        rotation_files=sm["rotation_files"],
        topology_files=sm["topology_files"],
        static_polygons=sm["static_polygons"],
        **kw,
    )


def _cloud(points, props=None, plate_ids=None):
    cloud = PointCloud.from_latlon(np.asarray(points, dtype=float))
    for name, vals in (props or {}).items():
        cloud.add_property(name, np.asarray(vals))
    if plate_ids is not None:
        cloud.plate_ids = np.asarray(plate_ids)
    return cloud


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_topology_files_required(self, synthetic_model):
        with pytest.raises(ValueError, match="topology_files is required"):
            PointRotator(rotation_files=synthetic_model["rotation_files"])

    def test_builds_topological_model(self, synthetic_model):
        r = _rotator(synthetic_model)
        assert isinstance(r._topological_model, pygplates.TopologicalModel)


# ---------------------------------------------------------------------------
# No silent drops
# ---------------------------------------------------------------------------
class TestNoSilentDrop:
    def test_n_out_equals_n_in(self, synthetic_model):
        sm = synthetic_model
        pts = [sm["plate_a_point"], sm["network_point"], sm["gap_point"],
               sm["out_of_circuit_point"]]
        cloud = _cloud(pts)
        r = _rotator(sm)
        out = r.rotate(cloud, from_age=0.0, to_age=100.0)
        assert out.n_points == cloud.n_points == 4

    def test_gap_point_kept_and_unmoved(self, synthetic_model):
        sm = synthetic_model
        cloud = _cloud([sm["gap_point"]])
        r = _rotator(sm)
        out = r.rotate(cloud, from_age=0.0, to_age=100.0)
        assert out.n_points == 1
        np.testing.assert_allclose(out.latlon[0], sm["gap_point"], atol=1e-6)

    def test_rotate_needs_no_plate_ids(self, synthetic_model):
        """Motion no longer requires plate_ids (contrast with the old engine)."""
        sm = synthetic_model
        cloud = _cloud([sm["plate_a_point"]])
        assert cloud.plate_ids is None
        r = _rotator(sm)
        out = r.rotate(cloud, from_age=0.0, to_age=100.0)  # must not raise
        assert out.n_points == 1


# ---------------------------------------------------------------------------
# Out-of-circuit drop regression (the decisive fix)
# ---------------------------------------------------------------------------
class TestOutOfCircuitRegression:
    def test_old_engine_would_drop_but_new_keeps(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["out_of_circuit_point"]])
        cloud = r.assign_plate_ids(cloud, at_age=0.0, source="static")

        # Precondition: the static polygon labels it plate 999 ...
        assert cloud.plate_ids[0] == sm["out_of_circuit_plate"]
        # ... which the OLD rigid engine flags as unreconstructable and drops.
        rm = pygplates.RotationModel([sm["rotation_files"]])
        bad = _unreconstructable_plate_ids(cloud.plate_ids, rm, 100.0)
        assert sm["out_of_circuit_plate"] in bad

        # The NEW engine keeps it and advects it by the topology (plate A: +30).
        out = r.rotate(cloud, from_age=0.0, to_age=100.0)
        assert out.n_points == 1
        np.testing.assert_allclose(out.latlon[0], [0.0, 30.0], atol=1e-3)
        # Label carried through unchanged, in place.
        assert out.plate_ids[0] == sm["out_of_circuit_plate"]


# ---------------------------------------------------------------------------
# Round trips (both directions)
# ---------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize("t", [50.0, 100.0])
    def test_present_to_past_to_present(self, synthetic_model, t):
        sm = synthetic_model
        cloud = _cloud([sm["plate_a_point"], sm["network_point"], sm["gap_point"]])
        r = _rotator(sm)
        fwd = r.rotate(cloud, from_age=0.0, to_age=t)
        back = r.rotate(fwd, from_age=t, to_age=0.0)
        # Measured max round-trip error is ~1e-3 deg (~0.1 km); pin at 1e-2 deg
        # (~1 km) to leave a little slack for network-deformation irreversibility.
        np.testing.assert_allclose(back.latlon, cloud.latlon, atol=1e-2)

    def test_past_to_present_direction(self, synthetic_model):
        """Older -> present must work as well as present -> older."""
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["plate_a_point"]])
        at100 = r.rotate(cloud, from_age=0.0, to_age=100.0)
        back = r.rotate(at100, from_age=100.0, to_age=0.0)
        np.testing.assert_allclose(back.latlon, cloud.latlon, atol=1e-3)


# ---------------------------------------------------------------------------
# Property / plate_id / ORDER preservation
# ---------------------------------------------------------------------------
class TestPreservation:
    def test_properties_and_plate_ids_preserved(self, synthetic_model):
        sm = synthetic_model
        pts = [sm["plate_a_point"], sm["gap_point"], sm["out_of_circuit_point"]]
        props = {"depth": [1.0, 2.0, 3.0], "temp": [10.0, 20.0, 30.0]}
        cloud = _cloud(pts, props=props, plate_ids=[11, 22, 33])
        r = _rotator(sm)
        out = r.rotate(cloud, from_age=0.0, to_age=50.0)
        np.testing.assert_array_equal(out.get_property("depth"), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out.get_property("temp"), [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out.plate_ids, [11, 22, 33])

    def test_order_preserved(self, synthetic_model):
        """get_geometry_points(return_inactive_points=True) preserves input order.

        Pin it explicitly: rotate each point alone and compare to the batched
        result at the same index. Any reordering would break the property/
        plate_id lockstep the engine relies on.
        """
        sm = synthetic_model
        rng = np.random.default_rng(0)
        lats = rng.uniform(-60, 60, 25)
        lons = rng.uniform(-180, 180, 25)
        cloud = _cloud(np.column_stack([lats, lons]))
        r = _rotator(sm)
        batched = r.rotate(cloud, from_age=0.0, to_age=40.0).latlon
        for i in range(len(lats)):
            single = r.rotate(_cloud([[lats[i], lons[i]]]),
                              from_age=0.0, to_age=40.0).latlon[0]
            np.testing.assert_allclose(batched[i], single, atol=1e-6,
                                       err_msg=f"order/position mismatch at {i}")


# ---------------------------------------------------------------------------
# Engine consistency: topological == rigid using the topology's own id
# ---------------------------------------------------------------------------
class TestEngineConsistency:
    def test_topological_equals_rigid_with_topology_id(self, synthetic_model):
        """On a rigid topological plate, advection == rigid rotation by that id.

        Mirrors the decisive Phase 0 test (median 0.000 deg). Uses points inside
        plate A's rigid boundary, whose topology id is 1.
        """
        sm = synthetic_model
        r = _rotator(sm)
        pts = np.array([[10.0, 10.0], [-20.0, -15.0], [30.0, 25.0], [5.0, -30.0]])
        cloud = _cloud(pts)
        # topology id for all of these is 1 (verify)
        ids = r.assign_plate_ids(cloud, at_age=0.0, source="topology").plate_ids
        assert np.all(ids == 1)

        topo = r.rotate(cloud, from_age=0.0, to_age=100.0).xyz
        rm = pygplates.RotationModel([sm["rotation_files"]])
        rigid = _move_points_batched(cloud.xyz, rm, ids, 0.0, 100.0)
        # sub-metre agreement
        d = np.linalg.norm(topo - rigid, axis=1)
        assert d.max() < 1.0, f"max disagreement {d.max()} m"

    def test_network_point_actually_deforms(self, synthetic_model):
        """A point in the deforming network is NOT reproduced by any single
        rigid pole (proves the network is exercised, not treated as rigid)."""
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["network_point"]])
        topo = r.rotate(cloud, from_age=0.0, to_age=100.0).latlon[0]
        rm = pygplates.RotationModel([sm["rotation_files"]])
        # Rigid by either bounding plate (1 or 2) should differ from the
        # deformed position.
        for pid in (1, 2):
            rigid_xyz = _move_points_batched(cloud.xyz, rm, np.array([pid]), 0.0, 100.0)
            rigid_ll = np.column_stack(XYZ2LatLon(rigid_xyz))[0]
            assert not np.allclose(topo, rigid_ll, atol=0.5)


# ---------------------------------------------------------------------------
# Plate-id assignment (labelling) modes
# ---------------------------------------------------------------------------
class TestAssignPlateIds:
    def test_topology_is_default(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        # The out-of-circuit point: static says 999, topology says 1 (plate A).
        cloud = _cloud([sm["out_of_circuit_point"]])
        topo = r.assign_plate_ids(cloud, at_age=0.0)  # default source
        static = r.assign_plate_ids(cloud, at_age=0.0, source="static")
        assert topo.plate_ids[0] == 1
        assert static.plate_ids[0] == sm["out_of_circuit_plate"]

    def test_assign_does_not_drop_by_default(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["gap_point"]])  # partitions to plate 0
        with pytest.warns(UserWarning, match="undefined plate IDs"):
            out = r.assign_plate_ids(cloud, at_age=0.0)
        assert out.n_points == 1
        assert out.plate_ids[0] == 0

    def test_remove_undefined_still_available(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["gap_point"], sm["plate_a_point"]])
        with pytest.warns(UserWarning, match="undefined plate IDs"):
            out = r.assign_plate_ids(cloud, at_age=0.0, remove_undefined=True)
        assert out.n_points == 1  # gap point removed


# ---------------------------------------------------------------------------
# Deactivation subset branch: inactive points removed, survivors in lockstep
# ---------------------------------------------------------------------------
class TestDeactivationSubset:
    """Exercise rotate() WITH a deactivation policy that actually consumes
    mid-array points, and pin the load-bearing assumption that
    get_geometry_points(to_age, return_inactive_points=True) returns a
    FULL-LENGTH list with None holes (not a compacted survivors-only list)."""

    # A grid of points inside the deforming network. The network is compressed
    # (left edge +30 deg, right edge -20 deg), so with a network-exit policy some
    # points drift out and deactivate while others survive — deterministically.
    def _network_grid(self):
        la = np.linspace(-25, 25, 6)
        lo = np.linspace(65, 115, 7)
        LA, LO = np.meshgrid(la, lo)
        return np.column_stack([LA.ravel(), LO.ravel()])

    def _policy(self):
        return pygplates.ReconstructedGeometryTimeSpan.DefaultDeactivatePoints(
            threshold_velocity_delta=1.0,
            threshold_distance_to_boundary=30.0,
            deactivate_points_that_fall_outside_a_network=True,
        )

    def test_get_geometry_points_is_full_length_with_none_holes(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        pts = self._network_grid()
        mp = pygplates.MultiPointOnSphere(list(zip(pts[:, 0], pts[:, 1])))
        ts = r._topological_model.reconstruct_geometry(
            mp, initial_time=0.0, oldest_time=100.0, youngest_time=0.0,
            time_increment=1.0, deactivate_points=self._policy())
        recon = ts.get_geometry_points(100.0, return_inactive_points=True)
        # Full-length list, with SOME (not all, not none) None holes.
        assert len(recon) == len(pts)
        n_none = sum(p is None for p in recon)
        assert 0 < n_none < len(pts)

    def test_subset_keeps_survivors_in_lockstep(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        pts = self._network_grid()
        n = len(pts)
        cloud = _cloud(pts,
                       props={"idx": np.arange(n, dtype=float)},
                       plate_ids=np.arange(100, 100 + n))

        # Independent active mask from the engine.
        _, _, active = r._advect_topological(
            pts[:, 0], pts[:, 1], 0.0, 100.0,
            time_step=1.0, deactivate_points=self._policy())
        assert 0 < active.sum() < n          # genuine mid-array deactivation

        out = r.rotate(cloud, from_age=0.0, to_age=100.0,
                       deactivate_points=self._policy())

        # Cloud is subset by the active mask ...
        assert out.n_points == int(active.sum())
        # ... and properties + plate_ids are the exact survivors, in order.
        survivors = np.where(active)[0]
        np.testing.assert_array_equal(out.get_property("idx"),
                                      survivors.astype(float))
        np.testing.assert_array_equal(out.plate_ids, 100 + survivors)
        assert np.all(np.isfinite(out.xyz))


# ---------------------------------------------------------------------------
# time_step validation
# ---------------------------------------------------------------------------
class TestTimeStepValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.5])
    def test_nonpositive_time_step_raises(self, synthetic_model, bad):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["plate_a_point"]])
        with pytest.raises(ValueError, match="time_step must be a positive"):
            r.rotate(cloud, from_age=0.0, to_age=50.0, time_step=bad)


# ---------------------------------------------------------------------------
# Deep-time edge case
# ---------------------------------------------------------------------------
class TestDeepTime:
    def test_deep_time_does_not_raise_and_keeps_points(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        pts = [sm["plate_a_point"], sm["network_point"], sm["gap_point"]]
        cloud = _cloud(pts)
        # 0 -> 600 Ma (oldest edge of the synthetic rotation file).
        out = r.rotate(cloud, from_age=0.0, to_age=600.0)
        assert out.n_points == len(pts)          # kept (deactivate_points=None)
        assert np.all(np.isfinite(out.xyz))


# ---------------------------------------------------------------------------
# rotate_incremental delegates to the topological engine
# ---------------------------------------------------------------------------
class TestRotateIncremental:
    def test_incremental_matches_rotate(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        cloud = _cloud([sm["plate_a_point"], sm["network_point"]])
        one = r.rotate(cloud, from_age=0.0, to_age=100.0, time_step=1.0)
        inc = r.rotate_incremental(cloud, from_age=0.0, to_age=100.0, time_step=1.0)
        np.testing.assert_allclose(inc.xyz, one.xyz, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
