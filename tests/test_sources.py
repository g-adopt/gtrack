"""Tests for gtrack.build_indicator_source (seeds-only rotate + fresh background).

The whole point of the helper is a dense, hole-free background at ANY age so a
downstream nearest-neighbour interpolation is well-posed. The load-bearing test
is the coverage bound: the maximum nearest-neighbour distance from an
independent dense grid to the merged source stays close to the ideal background
grid spacing and does NOT grow with age (unlike a back-rotated fixed grid, which
deforms and opens gaps).
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gtrack import (
    PointCloud,
    PointRotator,
    build_indicator_source,
    create_sphere_mesh_latlon,
)


def _unit(xyz):
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def _ideal_spacing_deg(n):
    return np.degrees(np.sqrt(4.0 * np.pi / n))


def _max_nn_coverage_deg(source, n_probe=20000):
    """Max nearest-neighbour angular distance from a dense probe grid to source."""
    gl, go = create_sphere_mesh_latlon(n_probe)
    probe = _unit(PointCloud.from_latlon(np.column_stack([gl, go])).xyz)
    dist, _ = cKDTree(_unit(source.xyz)).query(probe, k=1)
    return np.degrees(2.0 * np.arcsin(np.clip(dist / 2.0, 0.0, 1.0))).max()


def _rotator(sm):
    return PointRotator(rotation_files=sm["rotation_files"],
                        topology_files=sm["topology_files"])


def _seed_cloud(sm, n=200, with_plate_ids=True, seed=1):
    rng = np.random.default_rng(seed)
    # Seeds inside plate A / the deforming network region of the synthetic model.
    lats = rng.uniform(-30, 30, n)
    lons = rng.uniform(-35, 115, n)
    cloud = PointCloud.from_latlon(np.column_stack([lats, lons]))
    cloud.add_property("thickness", rng.uniform(50.0, 200.0, n))
    cloud.add_property("tag", np.arange(n, dtype=float))
    if with_plate_ids:
        cloud.plate_ids = np.arange(1000, 1000 + n)
    return cloud


# The coverage bound the design exists to guarantee. Fibonacci max-NN runs
# ~1.15x the ideal spacing; 2.2x leaves headroom for grid randomness while a
# genuine hole (a deformed/torn background) would blow far past it.
COVERAGE_FACTOR = 2.2
BG_N = 8000


class TestCoverage:
    def test_background_is_hole_free(self, synthetic_model):
        sm = synthetic_model
        src = build_indicator_source(
            _seed_cloud(sm), _rotator(sm), target_age=50.0, background_n=BG_N)
        bound = COVERAGE_FACTOR * _ideal_spacing_deg(BG_N)
        assert _max_nn_coverage_deg(src) < bound

    def test_coverage_does_not_grow_with_age(self, synthetic_model):
        """Deep time: the bound still holds (a back-rotated fixed grid would tear
        open here; the fresh grid does not)."""
        sm = synthetic_model
        r = _rotator(sm)
        seeds = _seed_cloud(sm)
        bound = COVERAGE_FACTOR * _ideal_spacing_deg(BG_N)
        cov = []
        for age in (50.0, 200.0, 500.0):
            src = build_indicator_source(seeds, r, target_age=age, background_n=BG_N)
            cov.append(_max_nn_coverage_deg(src))
            assert cov[-1] < bound, f"coverage {cov[-1]:.2f} deg exceeds bound at {age} Ma"
        # And it does not blow up with age.
        assert max(cov) < bound


class TestCollisionRemoval:
    def test_no_background_within_exclusion_of_a_seed(self, synthetic_model):
        sm = synthetic_model
        n_seed = 150
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=40.0, background_n=BG_N, exclusion_factor=1.0)
        # Seeds are the first n_seed rows (concatenate puts rotated first).
        seeds_xyz = _unit(src.xyz[:n_seed])
        bg_xyz = _unit(src.xyz[n_seed:])
        theta = 1.0 * np.sqrt(4.0 * np.pi / BG_N)
        chord = 2.0 * np.sin(theta / 2.0)
        dist, _ = cKDTree(seeds_xyz).query(bg_xyz, k=1)
        assert dist.min() > chord * (1.0 - 1e-9)


class TestSeedPreservation:
    def test_all_seeds_present_at_rotated_position(self, synthetic_model):
        sm = synthetic_model
        r = _rotator(sm)
        seeds = _seed_cloud(sm, n=120)
        n_seed = seeds.n_points

        src = build_indicator_source(seeds, r, target_age=60.0, background_n=BG_N)
        rotated = r.rotate(seeds, from_age=0.0, to_age=60.0)

        # First n_seed rows are the rotated seeds, in order.
        np.testing.assert_allclose(src.xyz[:n_seed], rotated.xyz, atol=1e-9)
        np.testing.assert_array_equal(
            src.get_property("thickness")[:n_seed], seeds.get_property("thickness"))
        np.testing.assert_array_equal(
            src.get_property("tag")[:n_seed], seeds.get_property("tag"))
        np.testing.assert_array_equal(src.plate_ids[:n_seed], seeds.plate_ids)

    def test_background_carries_background_value(self, synthetic_model):
        sm = synthetic_model
        n_seed = 120
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=60.0, background_n=BG_N, background_value=0.0)
        bg_thickness = src.get_property("thickness")[n_seed:]
        bg_tag = src.get_property("tag")[n_seed:]
        assert np.all(bg_thickness == 0.0)
        assert np.all(bg_tag == 0.0)
        # Background plate_ids are 0 when seeds carry plate_ids.
        assert np.all(src.plate_ids[n_seed:] == 0)

    def test_background_value_nonzero(self, synthetic_model):
        sm = synthetic_model
        n_seed = 100
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=30.0, background_n=BG_N, background_value=-1.0)
        assert np.all(src.get_property("thickness")[n_seed:] == -1.0)


class TestPlateIdVariants:
    def test_seeds_without_plate_ids_give_background_without_plate_ids(self, synthetic_model):
        sm = synthetic_model
        seeds = _seed_cloud(sm, n=100, with_plate_ids=False)
        src = build_indicator_source(seeds, _rotator(sm), target_age=30.0, background_n=BG_N)
        assert src.plate_ids is None


class TestValidation:
    def test_bad_background_n(self, synthetic_model):
        sm = synthetic_model
        with pytest.raises(ValueError, match="background_n must be positive"):
            build_indicator_source(_seed_cloud(sm), _rotator(sm), 10.0, background_n=0)

    def test_bad_exclusion_factor(self, synthetic_model):
        sm = synthetic_model
        with pytest.raises(ValueError, match="exclusion_factor must be non-negative"):
            build_indicator_source(_seed_cloud(sm), _rotator(sm), 10.0, exclusion_factor=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
