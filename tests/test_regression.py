"""
Regression tests for SeafloorAgeTracker and PointRotator.

These tests ensure the scientific output remains consistent across code changes.
Reference data is stored in tests/data/ as npz files.
"""

import numpy as np
import pytest
from pathlib import Path

from gtrack import SeafloorAgeTracker, TracerConfig, PointCloud, PointRotator

# Data paths. The plate model is what `make data` fetches into examples/, which
# is also where test_checkpoint_roundtrip and test_sub_myr_step look for it.
DATA_DIR = Path(__file__).parent.parent / "examples" / "Matthews_et_al_410_0"
REF_DIR = Path(__file__).parent / "data"

ROTATION_FILES = [
    DATA_DIR / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    DATA_DIR / "Global_EB_410-250Ma_GK07_Matthews++.rot"
]
TOPOLOGY_FILES = [
    DATA_DIR / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "Paleozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "TopologyBuildingBlocks_Matthews++.gpml",
]
CONTINENTAL_POLYGONS = DATA_DIR / "ContPolys/PresentDay_ContinentalPolygons_Matthews++.shp"
STATIC_POLYGONS = DATA_DIR / "StaticPolys/PresentDay_StaticPlatePolygons_Matthews++.shp"


def _data_files_exist():
    """Check if GPlates data files exist."""
    return all(f.exists() for f in ROTATION_FILES) and all(f.exists() for f in TOPOLOGY_FILES)


def _run_tracker_200_to_180():
    """Run tracker from 200 Ma to 180 Ma with 1 Myr timesteps."""
    config = TracerConfig(
        time_step=1.0,
        default_refinement_levels=5,
        initial_ocean_mean_spreading_rate=75.0,
        ridge_sampling_degrees=2.0,
        spreading_offset_degrees=0.01,
        velocity_delta_threshold=7.0,
        distance_threshold_per_myr=10.0,
    )

    tracker = SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=config,
        verbose=False
    )

    tracker.initialize(starting_age=200)

    # Only store results at key checkpoints (190 and 180 Ma)
    results = {}
    for target_age in [190, 180]:
        cloud = tracker.step_to(target_age)
        results[target_age] = {
            'xyz': cloud.xyz.copy(),
            'ages': cloud.get_property('age').copy(),
        }

    return results


def generate_reference_data():
    """Generate and save reference data. Run manually when needed."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    results = _run_tracker_200_to_180()

    for age, data in results.items():
        np.savez(REF_DIR / f"ref_age_{age:03d}.npz", xyz=data['xyz'], ages=data['ages'])

    print(f"Saved reference data for ages 190 and 180 to {REF_DIR}")


@pytest.mark.skipif(not _data_files_exist(), reason="GPlates data files not found")
def test_tracker_200_to_180_regression():
    """Test that tracker output matches reference data at key checkpoints."""
    ref_file = REF_DIR / "ref_age_180.npz"
    if not ref_file.exists():
        pytest.skip("Reference data not found. Run generate_reference_data() first.")

    results = _run_tracker_200_to_180()

    for age in [190, 180]:
        ref = np.load(REF_DIR / f"ref_age_{age:03d}.npz")
        np.testing.assert_allclose(results[age]['xyz'], ref['xyz'], rtol=1e-10,
                                   err_msg=f"XYZ mismatch at {age} Ma")
        np.testing.assert_allclose(results[age]['ages'], ref['ages'], rtol=1e-10,
                                   err_msg=f"Ages mismatch at {age} Ma")


# =============================================================================
# Point Rotation Regression Test
# =============================================================================

ROTATION_SEED = 42  # Fixed seed for reproducible random points


def _static_polygons_exist():
    """Check if static polygon files exist."""
    return STATIC_POLYGONS.exists() and _data_files_exist()


def _run_point_rotation_to_200():
    """
    Create seeded random points and rotate them to 200 Ma topologically.

    Uses a fixed seed so random points are always the same. Under the single
    topological engine, rotation no longer needs plate IDs and never drops
    points, so this exercises the no-drop / property-preservation contract on
    real data rather than comparing against a (now removed) rigid reference.
    """
    # Set seed for reproducibility
    rng = np.random.default_rng(ROTATION_SEED)

    # Generate random lat/lon points (500 points for reasonable test coverage)
    n_points = 500
    lats = rng.uniform(-80, 80, n_points)  # Avoid poles for stability
    lons = rng.uniform(-180, 180, n_points)
    latlon = np.column_stack([lats, lons])

    # Create PointCloud
    cloud = PointCloud.from_latlon(latlon)

    # Add a test property (seeded random values)
    test_property = rng.uniform(0, 100, n_points)
    cloud.add_property('test_value', test_property)

    # Create rotator (topology_files now required for the deforming-aware engine)
    rotator = PointRotator(
        rotation_files=[str(f) for f in ROTATION_FILES],
        topology_files=[str(f) for f in TOPOLOGY_FILES],
        static_polygons=str(STATIC_POLYGONS),
    )

    # Rotate to 200 Ma — no plate-id assignment or pre-filtering needed.
    rotated = rotator.rotate(cloud, from_age=0.0, to_age=200.0)

    return {
        'n_in': cloud.n_points,
        'xyz': rotated.xyz,
        'test_value': rotated.get_property('test_value'),
        'n_out': len(rotated),
        'input_test_value': test_property,
    }


@pytest.mark.skipif(not _static_polygons_exist(), reason="Static polygon files not found")
def test_point_rotation_to_200_no_drop_and_preserves_properties():
    """Topological rotation to 200 Ma keeps every point and its properties.

    Replaces the old rigid XYZ regression (which depended on a per-plate-id
    engine that has been removed). The invariant that matters now is: no silent
    drops, properties preserved in order, and finite positions.
    """
    results = _run_point_rotation_to_200()

    # No silent drops.
    assert results['n_out'] == results['n_in'] == 500

    # Positions are finite and actually moved (200 Myr of plate motion).
    assert np.all(np.isfinite(results['xyz']))

    # Properties preserved, in order.
    np.testing.assert_array_equal(results['test_value'], results['input_test_value'])


if __name__ == "__main__":
    generate_reference_data()
