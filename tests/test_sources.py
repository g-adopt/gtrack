"""Test ``gtrack.build_indicator_source`` with a new background at each age.

The new background keeps nearest-neighbour interpolation well-posed at all ages.
The maximum source distance stays near the ideal background spacing. A rotated
fixed background deforms and can open gaps at large ages.
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
# genuine hole from a deformed background exceeds it.
COVERAGE_FACTOR = 2.2
BG_N = 8000


class TestCoverage:
    def test_coverage_does_not_grow_with_age(self, synthetic_model):
        """Keep the coverage bound at large ages with each new background."""
        sm = synthetic_model
        r = _rotator(sm)
        seeds = _seed_cloud(sm)
        bound = COVERAGE_FACTOR * _ideal_spacing_deg(BG_N)
        cov = []
        for age in (50.0, 200.0, 500.0):
            src = build_indicator_source(
                seeds, r, target_age=age, background_point_count=BG_N
            )
            cov.append(_max_nn_coverage_deg(src))
            assert cov[-1] < bound, f"coverage {cov[-1]:.2f} deg exceeds bound at {age} Ma"


class TestCollisionRemoval:
    def test_no_background_within_exclusion_of_a_seed(self, synthetic_model):
        sm = synthetic_model
        n_seed = 150
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=40.0, background_point_count=BG_N, exclusion_factor=1.0)
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

        src = build_indicator_source(
            seeds, r, target_age=60.0, background_point_count=BG_N
        )
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
            target_age=60.0, background_point_count=BG_N, background_value=0.0)
        bg_thickness = src.get_property("thickness")[n_seed:]
        bg_tag = src.get_property("tag")[n_seed:]
        assert np.all(bg_thickness == 0.0)
        assert np.all(bg_tag == 0.0)
        # Background plate_ids are 0 when seeds carry plate_ids.
        assert np.all(src.plate_ids[n_seed:] == 0)


class TestMembership:
    """Keep lateral membership separate from physical thickness.

    A zero background makes interpolation smooth the product of membership and
    thickness. The separate membership channel lets a consumer recover thickness.
    """

    def test_membership_is_one_on_seeds_zero_on_background(self, synthetic_model):
        sm = synthetic_model
        n_seed = 120
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=60.0, background_point_count=BG_N)
        m = src.get_property("membership")
        assert np.all(m[:n_seed] == 1.0)
        assert np.all(m[n_seed:] == 0.0)

    def test_membership_ignores_background_value(self, synthetic_model):
        """background_value is about thickness-like channels; membership must
        not inherit it. A nonzero fill marks the exterior as
        partially in-region."""
        sm = synthetic_model
        n_seed = 100
        src = build_indicator_source(
            _seed_cloud(sm, n=n_seed), _rotator(sm),
            target_age=30.0, background_point_count=BG_N, background_value=-1.0)
        assert np.all(src.get_property("thickness")[n_seed:] == -1.0)
        assert np.all(src.get_property("membership")[n_seed:] == 0.0)

    def test_membership_can_be_disabled(self, synthetic_model):
        sm = synthetic_model
        src = build_indicator_source(
            _seed_cloud(sm, n=80), _rotator(sm),
            target_age=20.0, background_point_count=BG_N,
            membership_property=None)
        assert "membership" not in src.properties

    def test_membership_deblends_thickness(self, synthetic_model):
        """Recover physical thickness from the two interpolated channels.

        Constant source thickness gives an exact result where membership is
        nonzero. The masked thickness alone decreases across the boundary.
        """
        sm = synthetic_model
        n_seed = 200
        seeds = _seed_cloud(sm, n=n_seed)
        seeds.add_property("thickness", np.full(n_seed, 180.0))
        src = build_indicator_source(
            seeds, _rotator(sm), target_age=40.0,
            background_point_count=BG_N)

        thickness = src.get_property("thickness")
        membership = src.get_property("membership")

        # Gaussian-weighted kNN blend, as the downstream interpolator does it.
        gl, go = create_sphere_mesh_latlon(4000)
        probe = _unit(PointCloud.from_latlon(np.column_stack([gl, go])).xyz)
        dist, idx = cKDTree(_unit(src.xyz)).query(probe, k=40)
        w = np.exp(-dist**2 / (2 * 0.05**2))
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
        h_blend = np.sum(w * thickness[idx], axis=1)
        m_blend = np.sum(w * membership[idx], axis=1)

        covered = m_blend > 1e-3
        assert covered.sum() > 100
        np.testing.assert_allclose(
            h_blend[covered] / m_blend[covered], 180.0, rtol=1e-9)

        # Masked thickness is less than physical thickness across the boundary.
        partial = covered & (m_blend < 0.9)
        assert partial.sum() > 0
        assert np.all(h_blend[partial] < 180.0 * 0.9)


class TestPlateIdVariants:
    def test_seeds_without_plate_ids_give_background_without_plate_ids(self, synthetic_model):
        sm = synthetic_model
        seeds = _seed_cloud(sm, n=100, with_plate_ids=False)
        src = build_indicator_source(
            seeds, _rotator(sm), target_age=30.0,
            background_point_count=BG_N,
        )
        assert src.plate_ids is None


class TestValidation:
    def test_bad_background_point_count(self, synthetic_model):
        sm = synthetic_model
        with pytest.raises(ValueError, match="background_point_count must be positive"):
            build_indicator_source(
                _seed_cloud(sm), _rotator(sm), 10.0,
                background_point_count=0,
            )

    def test_bad_exclusion_factor(self, synthetic_model):
        sm = synthetic_model
        with pytest.raises(ValueError, match="exclusion_factor must be non-negative"):
            build_indicator_source(_seed_cloud(sm), _rotator(sm), 10.0, exclusion_factor=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
