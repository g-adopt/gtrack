"""Phase 4 pinning test: continental polygon reconstruction is RIGID.

CHANGE-PLAN.md Phase 4 audited the second rigid-reconstruction site,
``ContinentalPolygonCache`` (boundaries.py), which reconstructs continental
polygons with ``pygplates.reconstruct`` (per-feature Euler pole) rather than
topologically. The decision was to KEEP it rigid so the ocean tracker's
continental excision stays identical to GPlately's SeafloorGrid, and because the
craton/crust containment runs at at_age=0 (identity) and is unaffected.

This test pins that decision: it demonstrates, on a tiny synthetic model, that

  * an in-circuit continental polygon (plate with a rigid sequence) is moved to
    its reconstructed position at a non-zero age, while
  * an out-of-circuit continental polygon (plate id with NO rigid sequence) is
    identity-reconstructed, i.e. left at its present-day position — the very
    limitation the decision documents.

If continental reconstruction is ever made topology-consistent, this test should
be updated deliberately.
"""

import numpy as np
import pygplates
import pytest

from gtrack.boundaries import ContinentalPolygonCache


def _polygon_feature(vertices_latlon, plate_id):
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in vertices_latlon]
    feature = pygplates.Feature()
    feature.set_geometry(pygplates.PolygonOnSphere(points))
    feature.set_reconstruction_plate_id(plate_id)
    feature.set_valid_time(600.0, -999.0)
    return feature


@pytest.fixture
def rigid_env(tmp_path):
    """One polygon on plate 1 (moves) and one on plate 999 (no rigid sequence)."""
    in_circuit = _polygon_feature(
        [(10.0, 0.0), (10.0, 20.0), (-10.0, 20.0), (-10.0, 0.0)], plate_id=1)
    out_circuit = _polygon_feature(
        [(10.0, 100.0), (10.0, 120.0), (-10.0, 120.0), (-10.0, 100.0)], plate_id=999)

    gpml = str(tmp_path / "continents.gpml")
    pygplates.FeatureCollection([in_circuit, out_circuit]).write(gpml)

    rot = str(tmp_path / "rotations.rot")
    with open(rot, "w") as f:
        f.write("0    0.0  0.0 0.0  0.0 000 !anchor\n")
        f.write("0  600.0  0.0 0.0  0.0 000 !anchor\n")
        f.write("1    0.0  0.0 0.0  0.0 000 !plate1 @0\n")
        f.write("1  100.0 90.0 0.0 30.0 000 !plate1 @100 (+30 deg about N pole)\n")
        f.write("1  600.0 90.0 0.0 30.0 000 !plate1 @600\n")
        # plate 999 deliberately has no sequence -> out of circuit
    return gpml, rot


def _polygon_centroid_lon(poly):
    lons = [p.to_lat_lon()[1] for p in poly.get_points()]
    return float(np.mean(lons))


def test_continental_reconstruction_is_rigid_and_pins_limitation(rigid_env):
    gpml, rot = rigid_env
    rm = pygplates.RotationModel([rot])
    features = pygplates.FeatureCollection(gpml)
    cache = ContinentalPolygonCache(features, rm)

    polys_0 = cache.get_polygons(0.0)
    polys_100 = cache.get_polygons(100.0)
    assert len(polys_0) == len(polys_100) == 2

    # Match polygons across times by their present-day centroid longitude.
    cent0 = sorted(_polygon_centroid_lon(p) for p in polys_0)
    cent100 = sorted(_polygon_centroid_lon(p) for p in polys_100)

    # The in-circuit polygon (present centroid ~10 lon) moves ~+30 deg to ~40.
    # The out-of-circuit polygon (present centroid ~110 lon) stays put (~110),
    # because its plate id has no rigid sequence -> identity reconstruction.
    in_circuit_present, out_circuit_present = cent0  # ~10, ~110
    assert in_circuit_present == pytest.approx(10.0, abs=1.0)
    assert out_circuit_present == pytest.approx(110.0, abs=1.0)

    moved, stayed = cent100
    assert moved == pytest.approx(40.0, abs=1.5), "in-circuit polygon should rotate ~+30 deg"
    assert stayed == pytest.approx(110.0, abs=1.0), (
        "out-of-circuit polygon is identity-reconstructed (documented Phase 4 limitation)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
