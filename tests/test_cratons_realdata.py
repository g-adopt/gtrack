"""Slow, optional end-to-end check on the real Cratons_M3_B data.

Skipped automatically when the example data under examples/Cratons_M3_B is
absent. When present, it reproduces the Phase 0 ground-truth numbers with the
NEW topological engine (CHANGE-PLAN.md Phase 5.7 / 6.4):

  * the full 40,000-point masked craton grid survives 0 -> 50 Ma with ZERO holes
    (background coverage fraction == 1.0), whereas the old rigid path dropped
    ~2,700; and
  * all ~2,919 non-zero craton seeds survive (the old rigid path dropped ~95
    out-of-circuit seeds, e.g. the Kalahari craton on plate 77030).

This is the numeric replacement for eyeballing the source-cloud scatter.
"""

from pathlib import Path

import numpy as np
import pytest

CRATONS = Path(__file__).parent.parent / "examples" / "Cratons_M3_B"
GPLATES = CRATONS / "gplates_files"
ZAH = GPLATES / "Zahirovic_2022"
H5 = CRATONS / "continental_data" / "lithospheric_thickness_mesh.h5"

ROTATION_FILES = [str(ZAH / "Zahirovic2022_CombinedRotations_fixed_crossovers.rot")]
TOPOLOGY_FILES = [
    str(ZAH / "Zahirovic2022_ActiveDeformation.gpmlz"),
    str(ZAH / "Zahirovic2022_InactiveDeformation.gpmlz"),
    str(ZAH / "Zahirovic2022_PlateBoundaries.gpmlz"),
]
CRATON_POLYGONS = str(GPLATES / "fast_velocity_anomalies_4pct.shp")

_REQUIRED = [H5, *[Path(p) for p in ROTATION_FILES + TOPOLOGY_FILES],
             Path(CRATON_POLYGONS)]


def _data_present():
    return all(p.exists() for p in _REQUIRED)


pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not _data_present(),
                                 reason="Cratons_M3_B example data not present")]


def _build_masked_craton_cloud():
    import h5py
    from gtrack import PointCloud, PolygonFilter

    with h5py.File(H5, "r") as f:
        lonlat = f["lonlat"][:]                     # (N, 2) as (lon, lat)
        values = f["values"][:]
    latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
    cloud = PointCloud.from_latlon(latlon)

    pf = PolygonFilter(polygon_files=CRATON_POLYGONS, rotation_files=ROTATION_FILES)
    mask = pf.get_containment_mask(cloud, at_age=0.0)
    prop = values.copy()
    prop[~mask] = 0.0
    cloud.add_property("thickness", prop)
    return cloud


def test_full_masked_grid_has_zero_holes_0_to_50():
    from gtrack import PointRotator

    cloud = _build_masked_craton_cloud()
    n_in = cloud.n_points
    assert n_in == 40000

    rotator = PointRotator(
        rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)

    # Zero holes: every input point survives -> coverage fraction 1.0.
    assert rotated.n_points == n_in
    assert np.all(np.isfinite(rotated.xyz))


def test_all_craton_seeds_survive_0_to_50():
    from gtrack import PointRotator

    cloud = _build_masked_craton_cloud()
    seeds = cloud.subset(cloud.get_property("thickness") > 0)
    n_seeds = seeds.n_points
    # Phase 0 baseline: ~2919 non-zero craton seeds.
    assert 2800 < n_seeds < 3000

    rotator = PointRotator(
        rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    rotated = rotator.rotate(seeds, from_age=0.0, to_age=50.0)

    # Every seed kept (old rigid path dropped ~95 out-of-circuit seeds).
    assert rotated.n_points == n_seeds
    np.testing.assert_array_equal(
        rotated.get_property("thickness"), seeds.get_property("thickness"))


# Committed topological ground truth for the 2919 craton seeds at 50 Ma. This is
# the POSITIONAL regression lock: a bug that misplaces points (but keeps count/
# finiteness) fails here. Regenerate deliberately with make_craton_reference.py
# if the engine legitimately changes.
REF_SEEDS_50MA = Path(__file__).parent / "data" / "ref_craton_seeds_50ma_topological.npz"


def _gc_distance_deg(latlon_a, latlon_b):
    a = np.radians(latlon_a); b = np.radians(latlon_b)
    dlat = b[:, 0] - a[:, 0]; dlon = b[:, 1] - a[:, 1]
    h = (np.sin(dlat / 2) ** 2
         + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def test_craton_seed_positions_match_topological_ground_truth():
    """Per-point positional lock against the committed topological reference."""
    from gtrack import PointRotator

    if not REF_SEEDS_50MA.exists():
        pytest.skip("topological seed reference not found; regenerate it")

    cloud = _build_masked_craton_cloud()
    seeds = cloud.subset(cloud.get_property("thickness") > 0)
    rotator = PointRotator(
        rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    rotated = rotator.rotate(seeds, from_age=0.0, to_age=50.0)

    ref_xyz = np.load(REF_SEEDS_50MA)["xyz"]
    assert rotated.xyz.shape == ref_xyz.shape
    # Deterministic engine -> tight per-point lock.
    np.testing.assert_allclose(rotated.xyz, ref_xyz, rtol=1e-9, atol=1e-3)


def test_craton_seed_displacement_matches_phase0():
    """Aggregate displacement 0->50 Ma matches the Phase 0 topological numbers.

    Pins the *magnitude* of motion (median 9.42, 95pct 26.20, max 27.08 deg on
    this data) so a change in the reconstruction is caught even without the npz.
    """
    from gtrack import PointRotator

    cloud = _build_masked_craton_cloud()
    seeds = cloud.subset(cloud.get_property("thickness") > 0)
    rotator = PointRotator(
        rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    rotated = rotator.rotate(seeds, from_age=0.0, to_age=50.0)

    d = _gc_distance_deg(seeds.latlon, rotated.latlon)
    assert np.median(d) == pytest.approx(9.42, abs=0.1)
    assert np.percentile(d, 95) == pytest.approx(26.20, abs=0.2)
    assert d.max() == pytest.approx(27.08, abs=0.2)


def test_indicator_source_uniform_coverage_at_50ma():
    """build_indicator_source: craton seeds + fresh zeros give uniform coverage.

    On real data the fresh background must stay hole-free at 50 Ma and keep all
    ~2919 seeds; contrast with a back-rotated fixed grid, which would deform.
    """
    from gtrack import (
        PointCloud, PointRotator, build_indicator_source, create_sphere_mesh_latlon)
    from scipy.spatial import cKDTree

    cloud = _build_masked_craton_cloud()
    seeds = cloud.subset(cloud.get_property("thickness") > 0)
    n_seeds = seeds.n_points

    rotator = PointRotator(rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    bg_n = 40000
    src = build_indicator_source(
        seeds, rotator, target_age=50.0, background_point_count=bg_n
    )

    # All seeds present (first n_seeds rows), thickness intact.
    np.testing.assert_array_equal(
        src.get_property("thickness")[:n_seeds], seeds.get_property("thickness"))
    assert np.all(src.get_property("thickness")[n_seeds:] == 0.0)

    # Uniform, hole-free coverage: max nearest-neighbour distance from a dense
    # independent probe grid stays near the ideal background spacing.
    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    gl, go = create_sphere_mesh_latlon(20000)
    probe = unit(PointCloud.from_latlon(np.column_stack([gl, go])).xyz)
    dist, _ = cKDTree(unit(src.xyz)).query(probe, k=1)
    max_nn_deg = np.degrees(2.0 * np.arcsin(np.clip(dist / 2.0, 0.0, 1.0))).max()
    ideal_deg = np.degrees(np.sqrt(4.0 * np.pi / bg_n))
    assert max_nn_deg < 2.2 * ideal_deg, (
        f"coverage {max_nn_deg:.2f} deg exceeds {2.2 * ideal_deg:.2f} deg "
        "(background should be hole-free)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
