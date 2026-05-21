"""Regression tests for sub-Myr time_step support.

Prior to the fix, SeafloorAgeTracker.step_to() cast both endpoints and the
time_increment to int before calling pygplates.TopologicalModel.reconstruct_geometry,
which silently collapsed any 0 < time_step < 1 to a zero increment and
triggered "Time increment must be positive." inside pygplates.

These tests exercise:
  - A clean sub-Myr step (time_step=0.5 over an integer-Ma range).
  - A non-integer step that also leaves a truncated final step shorter than
    time_step, which the new code handles by passing the actual span as
    time_increment for that step.
"""

import pytest
from pathlib import Path

from gtrack import SeafloorAgeTracker, TracerConfig


DATA_DIR = Path(__file__).parent.parent / "examples" / "Matthews_et_al_410_0"

ROTATION_FILES = [
    DATA_DIR / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    DATA_DIR / "Global_EB_410-250Ma_GK07_Matthews++.rot",
]
TOPOLOGY_FILES = [
    DATA_DIR / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "Paleozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "TopologyBuildingBlocks_Matthews++.gpml",
]
CONTINENTAL_POLYGONS = (
    DATA_DIR / "ContPolys" / "PresentDay_ContinentalPolygons_Matthews++.shp"
)


def _data_available():
    return all(f.exists() for f in ROTATION_FILES) and all(
        f.exists() for f in TOPOLOGY_FILES
    )


def _make_tracker(time_step):
    config = TracerConfig(
        time_step=time_step,
        default_mesh_points=2000,
        initial_ocean_mean_spreading_rate=75.0,
        ridge_sampling_degrees=1.0,
        spreading_offset_degrees=0.01,
        velocity_delta_threshold=7.0,
        distance_threshold_per_myr=10.0,
    )
    return SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=config,
    )


@pytest.mark.skipif(not _data_available(), reason="GPlates data files not found")
def test_half_myr_step():
    """time_step=0.5 over an integer-Ma range must run without error."""
    tracker = _make_tracker(time_step=0.5)
    tracker.initialize(starting_age=12)
    cloud = tracker.step_to(10)

    assert tracker._current_age == pytest.approx(10.0)
    assert len(cloud.xyz) > 0


@pytest.mark.skipif(not _data_available(), reason="GPlates data files not found")
def test_truncated_final_step():
    """A non-integer time_step that leaves a partial final step must still work.

    starting_age=11.5, target_age=10.0, time_step=0.7 yields steps of
    0.7, 0.7, 0.1 — the last span is smaller than time_step, exercising
    the clamp branch of next_time = max(time - time_step, target_age).
    """
    tracker = _make_tracker(time_step=0.7)
    tracker.initialize(starting_age=11.5)
    cloud = tracker.step_to(10.0)

    assert tracker._current_age == pytest.approx(10.0)
    assert len(cloud.xyz) > 0
